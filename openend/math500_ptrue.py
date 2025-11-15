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
        # 对于数学问题，我们使用A/B选项来验证答案正确性
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("A")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("B")
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
        """构建数学问题验证提示词"""
        return f"""You are a mathematics expert. You are given a math problem and a proposed solution.

Math Problem:
{question}

Proposed Solution:
{answer}

Please carefully verify whether this solution is correct. Check the reasoning step by step and verify the final answer.

Which of the following is correct?
A: The solution is correct and the final answer is accurate.
B: The solution contains errors or the final answer is incorrect.

Respond with only A or B. Do not provide any explanation:\n
Answer: 
"""

    def format_prompt(self, example, icl=True):
        """格式化数学问题提示词"""
        # MATH-500数据格式适配
        if isinstance(example, dict):
            question = example.get('problem', example.get('question', ''))
        else:
            question = str(example)

        icl_context = """Can you solve the following math problem?  Please reason step by step, and put your final answer within \\boxed{{}}.
Example:
Problem: What is 2 + 2?
Solution: 2 + 2 = 4. Therefore, the answer is 4.
Final answer: \\boxed{{4}}

Now solve this problem:

"""
        prompt = ""
        prompt += ('Problem: ' + question + '\n')
        prompt += "Solution: "

        if icl:
            prompt = icl_context + prompt
        return prompt

    def extract_boxed_answer(self, text):
        """提取数学问题中的boxed答案"""
        # 数学问题答案提取模式
        patterns = [
            r'\\boxed\{\{([^}]+)\}\}',  # \boxed{{answer}} - 双花括号
            r'\\boxed\{([^}]+)\}',  # \boxed{answer} - 单花括号
            r'boxed\{\{([^}]+)\}\}',  # boxed{{answer}} (缺少反斜杠)
            r'boxed\{([^}]+)\}',  # boxed{answer} (缺少反斜杠)
            r'final answer:\s*\\boxed\{\{([^}]+)\}\}',
            r'Final answer:\s*\\boxed\{\{([^}]+)\}\}',
            r'final answer:\s*\\boxed\{([^}]+)\}',
            r'Final answer:\s*\\boxed\{([^}]+)\}',
            r'answer:\s*\\boxed\{\{([^}]+)\}\}',
            r'Answer:\s*\\boxed\{\{([^}]+)\}\}',
            r'\\boxed\{\{([^}]+)\}\}',  # 通用匹配
            r'\\boxed\{([^}]+)\}',  # 通用匹配
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # 如果上面的模式都没匹配到，尝试匹配任何包含boxed的模式
        loose_pattern = r'\\boxed\{\{?(.+?)\}?\}'
        match = re.search(loose_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # 如果还是没有匹配到，尝试提取最后一行数字或表达式作为答案
        lines = text.strip().split('\n')
        for line in reversed(lines):
            # 匹配数字、分数、表达式等
            math_pattern = r'^(-?\d+\.?\d*|\\frac\{[^}]+\}\{[^}]+\}|[a-zA-Z]+\s*=\s*.+)$'
            match = re.search(math_pattern, line.strip())
            if match:
                return match.group(0).strip()

        return None

    def normalize_answer(self, answer):
        """标准化数学答案格式"""
        if answer is None:
            return ""

        # 移除多余空格
        answer = str(answer).strip()

        # 简化分数格式
        answer = re.sub(r'\s+', ' ', answer)

        # 处理常见数学符号
        answer = answer.replace('\\frac', '')
        answer = answer.replace('\\sqrt', 'sqrt')

        return answer

    def evaluate_dataset(self, dataset):
        """在整个数据集上运行P(True)评估"""

        # 提取问题
        questions = [self.format_prompt(item, icl=True) for item in dataset]

        # 生成参数设置
        generation_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=10240,  # 数学问题可能需要更多token
        )

        print("Step 1: Generating initial answers...")
        initial_answers = self.generate_answers(questions, generation_params)

        # 打印一些示例用于调试
        for i in range(min(3, len(questions))):
            print(f"\n--- Example {i} ---")
            print("Question:")
            print(questions[i][-200:])  # 显示最后200字符
            print("Generated Answer:")
            print(initial_answers[i][-200:])  # 显示最后200字符
            print("Extracted Answer:")
            print(self.extract_boxed_answer(initial_answers[i]))

        print("Step 2: Calculating P(True) scores...")
        # 计算P(True)分数 - 使用不包含示例的原始问题
        raw_questions = []
        for item in dataset:
            if isinstance(item, dict):
                raw_questions.append(item.get('problem', item.get('question', '')))
            else:
                raw_questions.append(str(item))

        p_true_scores = self.calculate_p_true(raw_questions, initial_answers)

        # 组装结果
        results = {
            "Yhat": [],
            "Y": [],
            "confidence": [],
            "raw_answers": [],
            "questions": []
        }

        for i, item in enumerate(dataset):
            extracted_answer = self.extract_boxed_answer(initial_answers[i])
            normalized_answer = self.normalize_answer(extracted_answer)

            # 获取真实答案
            if isinstance(item, dict):
                true_answer = item.get('solution', item.get('answer', ''))
                question_text = item.get('problem', item.get('question', ''))
            else:
                true_answer = ""
                question_text = str(item)

            results['Yhat'].append(normalized_answer)
            results['Y'].append(self.normalize_answer(true_answer))
            results["confidence"].append(p_true_scores[i])
            results["raw_answers"].append(initial_answers[i])
            results["questions"].append(question_text)

        # 保存结果
        save_result(self.args, results)

        # 打印统计信息
        print(f"\n--- Evaluation Summary ---")
        print(f"Total questions: {len(dataset)}")
        print(f"Average P(True) confidence: {np.mean(p_true_scores):.3f}")
        print(f"Answers extracted: {sum(1 for ans in results['Yhat'] if ans)}/{len(dataset)}")

        return results