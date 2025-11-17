import json
import re
import pandas as pd
from vllm import LLM, SamplingParams
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from collections import Counter
import torch

MODEL_PATH = "/mnt/sharedata/ssd_large/common/LLMs/Llama-3.1-8B-Instruct"

# ---------------------
# 载入模型（使用 vLLM 高效推理）
# ---------------------
llm = LLM(
    model=MODEL_PATH,
    dtype="float16",
    gpu_memory_utilization=0.95,  # 根据你的 GPU 调整
    tensor_parallel_size=torch.cuda.device_count(),  # 多 GPU 支持
)

# ---------------------
# 加载 MMLU (有正确答案)
# ---------------------
full_dataset = load_dataset(
    "parquet",
    data_files={
        "test": "/mnt/sharedata/ssd_large/common/datasets/mmlu/all/test-00000-of-00001.parquet",
        "validation": "/mnt/sharedata/ssd_large/common/datasets/mmlu/all/validation-00000-of-00001.parquet"}
)
dataset = concatenate_datasets([full_dataset["validation"], full_dataset["test"]])
label_list = ['A', 'B', 'C', 'D']
reformat = lambda x: {
    'question': x['question'],
    'choices': x['choices'],
    'answer': label_list[x['answer']],
    'label': label_list[:len(x["choices"])]
}
dataset = [reformat(data) for data in dataset]
dataset = [dataset[i] for i in range(10)]  # 只取前10个样本

# ---------------------
# 提取 verbalized confidence
# ---------------------
def extract_confidence(text):
    """
    从模型输出中抓取 like:
    "I am 72% confident" or "confidence: 0.81"
    """
    # 匹配百分比
    p = re.search(r"(\d{1,3})\s*%", text)
    if p:
        v = int(p.group(1))
        return min(max(v / 100, 0), 1)

    # 匹配 0.xx 的形式
    p2 = re.search(r"([01]\.\d+)", text)
    if p2:
        v = float(p2.group(1))
        return min(max(v, 0), 1)

    return None  # 没有 verbal confidence

def extract_prediction(text):
    """增强的答案提取函数"""
    patterns = [
        r"Answer\s*:\s*([A-D])",
        r"Final Answer\s*:\s*([A-D])",
        r"正确答案\s*:\s*([A-D])",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 备选方案
    m = re.search(r'\b([A-D])\b', text.split("Answer")[-1] if "Answer" in text else text)
    return m.group(1).upper() if m else None

# ---------------------
# prompt 设计（选择题）
# ---------------------
def build_prompt(question, choices):
    letters = ["A", "B", "C", "D"]
    formatted = "\n".join([f"{letters[i]}. {c}" for i, c in enumerate(choices)])

    return f"""You are a knowledgeable expert.

Answer the following multiple-choice question. 
Think step by step, then give your confidence in the final line as:
"I am XX% confident".

Question:
{question}

Options:
{formatted}

Final Answer (e.g., "Answer: A"):
"""

# ---------------------
# Self-Consistency 参数
# ---------------------
NUM_SAMPLES = 10  # 每个问题的采样次数
SAMPLING_PARAMS = SamplingParams(
    n=NUM_SAMPLES,  # vLLM 支持一次性生成多个样本
    temperature=0.8,  # 启用采样
    top_p=0.95,
    max_tokens=200,
)

# ---------------------
# 推理（Self-Consistency）
# ---------------------
results = []  # 用于存储所有结果

for item in tqdm(dataset):
    question = item["question"]
    choices = item["choices"]
    true_answer_letter = item["answer"]  # Already A-D from reformatting
    true_answer_idx = ord(true_answer_letter) - 65  # Convert A-D to 0-3 if needed

    prompt = build_prompt(question, choices)

    try:
        # 使用 vLLM 生成多个样本
        outputs = llm.generate([prompt], SAMPLING_PARAMS)  # 输入是 list of prompts
        generated_samples = outputs[0].outputs  # 每个 output.outputs 是 list of NUM_SAMPLES

        # ---------------------
        # 处理每个样本
        # ---------------------
        preds = []
        confidences = []
        responses = []

        for sample in generated_samples:
            response = sample.text.strip()  # 只提取模型新生成的部分
            responses.append(response)

            pred_letter = extract_prediction(response)
            if pred_letter:
                preds.append(pred_letter)

            confidence = extract_confidence(response)
            if confidence is not None:
                confidences.append(confidence)

        # ---------------------
        # Self-Consistency: 投票选最高频预测
        # ---------------------
        if preds:
            pred_counter = Counter(preds)
            pred_letter = pred_counter.most_common(1)[0][0]  # 最高频
            consistency_score = pred_counter[pred_letter] / NUM_SAMPLES  # 一致性分数作为额外指标
        else:
            pred_letter = None
            consistency_score = 0.0

        pred_idx = ord(pred_letter) - ord("A") if pred_letter else None

        # 平均置信度（如果有）
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        # 判断是否正确
        is_correct = (pred_letter == true_answer_letter) if pred_letter else False

        # 保存结果
        result = {
            "question": question,
            "options": choices,
            "Y": true_answer_letter,  # 真实答案 (A-D)
            "Y_true_idx": true_answer_idx,  # 真实答案索引 (0-3)
            "Yhat": pred_letter,  # 最终预测答案 (A-D, 来自一致性投票)
            "Yhat_idx": pred_idx,  # 预测答案索引 (0-3)
            "avg_confidence": avg_confidence,  # 平均置信度
            "consistency_score": consistency_score,  # 一致性分数 (0-1)
            "correct": is_correct,  # 是否正确
            "model_responses": responses  # 所有样本的完整回答 (list)
        }
        results.append(result)

    except Exception as e:
        print(f"Error processing item: {e}")
        # 记录错误情况
        result = {
            "question": question,
            "options": choices,
            "Y": true_answer_letter,
            "Y_true_idx": true_answer_idx,
            "Yhat": None,
            "Yhat_idx": None,
            "avg_confidence": None,
            "consistency_score": 0.0,
            "correct": False,
            "model_responses": [f"Error: {str(e)}"]
        }
        results.append(result)

# ---------------------
# 保存结果到CSV
# ---------------------
df = pd.DataFrame(results)

# 计算统计信息
correct_predictions = df['correct'].sum()
total_predictions = len(df)
accuracy = correct_predictions / total_predictions

valid_confidences = df['avg_confidence'].dropna()
confidence_extraction_rate = len(valid_confidences) / total_predictions
average_confidence = valid_confidences.mean() if len(valid_confidences) > 0 else None

consistency_scores = df['consistency_score']
average_consistency = consistency_scores.mean() if len(consistency_scores) > 0 else None

# 保存到CSV文件
csv_filename = "mmlu_self_consistency_results.csv"
df.to_csv(csv_filename, index=False, encoding='utf-8')

# ---------------------
# 输出统计结果
# ---------------------
print(f"\n{'=' * 50}")
print("实验结果统计 (Self-Consistency)")
print(f"{'=' * 50}")
print(f"总样本数: {total_predictions}")
print(f"正确预测: {correct_predictions}")
print(f"准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"置信度提取率: {confidence_extraction_rate:.4f} ({confidence_extraction_rate * 100:.2f}%)")
if average_confidence is not None:
    print(f"平均置信度: {average_confidence:.4f}")
if average_consistency is not None:
    print(f"平均一致性分数: {average_consistency:.4f}")
print(f"结果已保存到: {csv_filename}")

# 显示前几个样本的结果
print(f"\n前5个样本的结果:")
print(f"{'=' * 50}")
for i, row in df.head().iterrows():
    status = "✓" if row['correct'] else "✗"
    conf_display = f"{row['avg_confidence']:.3f}" if pd.notna(row['avg_confidence']) else "N/A"
    cons_display = f"{row['consistency_score']:.3f}"
    print(f"样本 {i}: 真实={row['Y']}, 预测={row['Yhat']}, 平均置信度={conf_display}, 一致性={cons_display} {status}")

# 可选：保存详细的统计报告
stats_report = {
    "total_samples": total_predictions,
    "correct_predictions": correct_predictions,
    "accuracy": accuracy,
    "confidence_extraction_rate": confidence_extraction_rate,
    "average_confidence": average_confidence,
    "average_consistency": average_consistency,
    "model_path": MODEL_PATH,
    "dataset": "abstract_algebra",  # 假设是这个子集，根据实际调整
    "num_samples_per_item": NUM_SAMPLES
}

with open("self_consistency_stats.json", "w", encoding='utf-8') as f:
    json.dump(stats_report, f, indent=2, ensure_ascii=False)

print(f"\n详细统计已保存到: self_consistency_stats.json")