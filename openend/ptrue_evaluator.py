from typing import List, Dict, Any, Tuple
from vllm import LLM, SamplingParams
import numpy as np
import json
import re
from utils import save_result


class PTrueEvaluator:
    def __init__(self, args):
        """初始化vLLM模型"""
        self.llm = LLM(
            model=f"/mnt/sharedata/ssd_large/common/LLMs/{args.model}",
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=0.5,
            max_model_len=args.max_model_len,
        )

        # 获取tokenizer用于编码Yes/No
        self.tokenizer = self.llm.get_tokenizer()
        #self.yes_token_id = self._get_first_token_id("Yes")
        #self.no_token_id = self._get_first_token_id("No")
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("A")
        self.no_token_id  = self.tokenizer.convert_tokens_to_ids("B")
        self.args = args

        print(f"Yes token ID: {self.yes_token_id}, No token ID: {self.no_token_id}")

    def _get_first_token_id(self, text: str) -> int:
        """获取文本的第一个token ID"""
        tokens = self.tokenizer.encode(text)
        return tokens[0] if tokens else -1

    def generate_answers(self, questions: List[str], sampling_params: SamplingParams) -> List[str]:
        """为问题生成初始答案"""
        outputs = self.llm.generate(questions, sampling_params)
        # 按输入顺序排序输出
        sorted_outputs = sorted(outputs, key=lambda x: x.request_id)
        return [output.outputs[0].text.strip() for output in sorted_outputs]

    def calculate_p_true(self, questions: List[str], answers: List[str]) -> List[float]:
        """计算P(True)置信度分数"""

        # 构建自我审查的提示
        prompts = []
        for question, answer in zip(questions, answers):
            prompt = self._build_verification_prompt(question, answer)
            prompts.append(prompt)
        for i in range(5):
            print(f"Prompt {i}")
            print(prompts[i])
            print()
        # 设置采样参数来获取logits
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=1,  # 我们只需要第一个token
            logprobs=20,  # 获取top-1 logprobs
            prompt_logprobs=0  # 不需要prompt的logprobs
        )

        # 生成并获取logits
        outputs = self.llm.generate(prompts, sampling_params)
        sorted_outputs = sorted(outputs, key=lambda x: x.request_id)

        p_true_scores = []
        for output in sorted_outputs:
            if output.outputs[0].logprobs:
                # 获取第一个生成token的logprobs
                first_token_logprobs = output.outputs[0].logprobs[0]

                # 提取Yes和No的logprob
                yes_logprob_obj = first_token_logprobs.get(self.yes_token_id)
                no_logprob_obj = first_token_logprobs.get(self.no_token_id)

                # Extract the actual logprob value
                yes_logprob = yes_logprob_obj.logprob if yes_logprob_obj else -float('inf')
                no_logprob = no_logprob_obj.logprob if no_logprob_obj else -float('inf')

                # 计算概率
                yes_prob = np.exp(yes_logprob) if yes_logprob != -float('inf') else 0.0
                no_prob = np.exp(no_logprob) if no_logprob != -float('inf') else 0.0
                total = yes_prob + no_prob
                if total > 0:
                    p_true = yes_prob / total
                else:
                    p_true = 0.0

                p_true_scores.append(p_true)
            else:
                # 如果没有logprobs，默认分数为0.5
                p_true_scores.append(0.5)

        return p_true_scores

    def _build_verification_prompt(self, question: str, answer: str) -> str:
        return f"""You are given a question and an answer.

Question:
{question}

We want to verify whether the following answer is correct:
'{answer}'

Please answer the following binary-choice question:

Which of the following is correct?
A: The answer above is correct.
B: The answer above is incorrect.

Response with A or B. No other words or explanation:\n"""



    def format_prompt(self, example, icl=True):
        icl_context = """Please answer the following multiple choice question and put your final answer in \\boxed{{}}.

Example:
Question: What is the capital of France?
A: London
B: Berlin  
C: Paris
D: Madrid

Final answer: \\boxed{{C}}

Now answer this question:

"""
        question = example['question']
        label = example['label']
        text = example['choices']

        prompt = ""
        prompt += ('Question: ' + question + '\n')

        for i in range(len(text)):
            prompt += (label[i] + ': ' + text[i] + '\n')

        if icl:
            prompt = icl_context + prompt
        return prompt

    def extract_boxed_answer(self, text):
        patterns = [
            r'\\boxed\{\{([ABCD])\}\}',  # \boxed{{C}} - 双花括号
            r'\\boxed\{([ABCD])\}',      # \boxed{C} - 单花括号
            r'boxed\{\{([ABCD])\}\}',    # boxed{{C}} (缺少反斜杠)
            r'boxed\{([ABCD])\}',        # boxed{C} (缺少反斜杠)
            r'final answer:\s*\\boxed\{\{([ABCD])\}\}',
            r'Final answer:\s*\\boxed\{\{([ABCD])\}\}',
            r'final answer:\s*\\boxed\{([ABCD])\}',
            r'Final answer:\s*\\boxed\{([ABCD])\}',
            r'\\boxed\{\{([ABCDabcd])\}\}',  # 大小写不敏感
            r'\\boxed\{([ABCDabcd])\}',      # 大小写不敏感
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()  # 确保大写

        # 如果上面的模式都没匹配到，尝试更宽松的匹配
        # 匹配 \boxed{ 和 } 之间的任何单个字母
        loose_pattern = r'\\boxed\{\{?([A-Da-d])\}?\}'
        match = re.search(loose_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return None

    def evaluate_dataset(self, dataset):
        """在整个数据集上运行P(True)评估"""

        # 提取问题
        questions = [self.format_prompt(item, icl=True) for item in dataset]
        stop_tokens = ["}}"]
        generation_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=1024,
            #stop = stop_tokens
        )
        initial_answers = self.generate_answers(questions, generation_params)

        # for i in range(5):
        #     print(f"Question {i}")
        #     print("Prompt")
        #     print(questions[i])
        #     print("Content of answer")
        #     print(initial_answers[i])
        print("Step 2: Calculating P(True) scores...")
        # 计算P(True)分数
        p_true_scores = self.calculate_p_true([self.format_prompt(item, icl=False) for item in dataset], initial_answers)

        # 组装结果
        results = {
            "Yhat": [],
            "Y": [],
            "confidence": [],
        }
        for i, item in enumerate(dataset):
            results['Yhat'].append(self.extract_boxed_answer(initial_answers[i]))
            results['Y'].append(item.get('answer', ''))
            results["confidence"].append(p_true_scores[i])
        save_result(self.args, results)
