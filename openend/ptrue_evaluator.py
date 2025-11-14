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
            gpu_memory_utilization=0.8,
            max_model_len=args.max_model_len,
        )

        # 获取tokenizer用于编码Yes/No
        self.tokenizer = self.llm.get_tokenizer()
        self.yes_token_id = self._get_first_token_id("Yes")
        self.no_token_id = self._get_first_token_id("No")
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

        # 设置采样参数来获取logits
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=1,  # 我们只需要第一个token
            logprobs=1,  # 获取top-1 logprobs
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
                yes_logprob = first_token_logprobs.get(self.yes_token_id, -float('inf'))
                no_logprob = first_token_logprobs.get(self.no_token_id, -float('inf'))

                # 计算概率
                yes_prob = np.exp(yes_logprob) if yes_logprob != -float('inf') else 0.0
                no_prob = np.exp(no_logprob) if no_logprob != -float('inf') else 0.0

                # 避免除零错误
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
        """构建验证提示"""
        return f"""Based on the following question: '{question}'
Is the answer '{answer}' correct?
You must answer with only a single word: 'Yes' or 'No'.
Answer:"""


    def format_prompt(self, example, icl=True):
        icl_context = """The following are multi choice questions. Please reason step by step, and conclude your final choice within \\boxed{}:

        Example 1:
        Question: What is the capital of France?
        A: London
        B: Berlin
        C: Paris
        D: Madrid

        Reasoning: The question asks for the capital of France. London is the capital of the United Kingdom, Berlin is the capital of Germany, Paris is the capital of France, and Madrid is the capital of Spain. Therefore, Paris is the correct answer.
        Final answer: \\boxed{C}

        Example 2:
        Question: Which gas do plants absorb during photosynthesis?
        A: Oxygen
        B: Carbon dioxide
        C: Nitrogen
        D: Hydrogen

        Reasoning: Photosynthesis is the process where plants convert light energy into chemical energy. They take in carbon dioxide and water, and produce glucose and oxygen. Therefore, plants absorb carbon dioxide during photosynthesis.
        Final answer: \\boxed{B}


        """
        if icl:
            prompt = icl_context + 'The following is a multi choice question. Please reason step by step, and conclude your final choice within \\boxed{}:'
        else:
            prompt = 'The following is a multi choice question. Please reason step by step, and conclude your final choice within \\boxed{}:'
        question = example['question']
        label = example['label']
        text = example['choices']

        prompt += ('Question: ' + question + '\n')

        for i in range(len(text)):
            prompt += label[i] + ': ' + text[i] + '\n'

        return prompt

    def extract_boxed_answer(self, text):
        """Extract the answer from \\boxed{} format"""
        # Pattern to match \boxed{X} where X is A, B, C, or D
        patterns = [
            r'\\boxed\{([ABCD])\}',  # \boxed{A}
            r'\\boxed{([ABCD]})',  # \boxed{A} (missing backslash)
            r'final answer:\s*\\boxed\{([ABCD])\}',
            r'Final answer:\s*\\boxed\{([ABCD])\}',
            r'\\boxed\{([ABCDabcd])\}',  # case insensitive
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()  # Ensure uppercase

        return None

    def evaluate_dataset(self, dataset: List[Dict], output_path: str = None):
        """在整个数据集上运行P(True)评估"""

        # 提取问题
        questions = [self.format_prompt(item, icl=True) for item in dataset]
        generation_params = SamplingParams(
            temperature=0.7,
            top_p=0.9
        )
        initial_answers = self.generate_answers(questions, generation_params)

        print("Step 2: Calculating P(True) scores...")
        # 计算P(True)分数
        p_true_scores = self.calculate_p_true([self.format_prompt(item, icl=False) for item in dataset], initial_answers)

        # 组装结果
        results = {
            "Yhat": [],
            "Y": [],
            "is_correct": [],
            "confidence": [],
        }
        for i, item in enumerate(dataset):
            results['Yhat'].append(self.extract_boxed_answer(initial_answers[i]))
            results['Y'].append(item.get('answer', ''))
            results["confidence"].append(p_true_scores[i])
        save_result(results, args=self.args)
