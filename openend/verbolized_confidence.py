import json
import re
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from tqdm import tqdm

MODEL_PATH = "/mnt/sharedata/ssd_large/common/LLMs/Qwen3-4B-Thinking-2507"
DEVICE = "cuda"

# ---------------------
# 载入模型
# ---------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------
# 加载 MMLU (有正确答案)
# ---------------------
dataset = load_dataset("hendrycksTest", "abstract_algebra")["test"]


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
# 推理
# ---------------------
results = []  # 用于存储所有结果

for item in tqdm(dataset):
    question = item["question"]
    choices = item["choices"]
    true_answer_idx = item["answer"]  # 0~3
    true_answer_letter = chr(65 + true_answer_idx)  # 转换为A-D

    prompt = build_prompt(question, choices)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    try:
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # 只提取模型新生成的部分（去掉prompt）
        response = generated_text[len(prompt):].strip()

        # ---------------------
        # 提取模型回答
        # ---------------------
        pred_letter = extract_prediction(response)
        pred_idx = ord(pred_letter) - ord("A") if pred_letter else None

        # ---------------------
        # 提取 verbalized confidence
        # ---------------------
        confidence = extract_confidence(response)

        # 判断是否正确
        is_correct = (pred_idx == true_answer_idx) if pred_idx is not None else False

        # 保存结果
        result = {
            "question": question,
            "options": choices,
            "Y": true_answer_letter,  # 真实答案 (A-D)
            "Y_true_idx": true_answer_idx,  # 真实答案索引 (0-3)
            "Yhat": pred_letter,  # 预测答案 (A-D)
            "Yhat_idx": pred_idx,  # 预测答案索引 (0-3)
            "confidence": confidence,  # 置信度
            "correct": is_correct,  # 是否正确
            "model_response": response  # 模型完整回答
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
            "confidence": None,
            "correct": False,
            "model_response": f"Error: {str(e)}"
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

valid_confidences = df['confidence'].dropna()
confidence_extraction_rate = len(valid_confidences) / total_predictions
average_confidence = valid_confidences.mean() if len(valid_confidences) > 0 else None

# 保存到CSV文件
csv_filename = "mmlu_results.csv"
df.to_csv(csv_filename, index=False, encoding='utf-8')

# ---------------------
# 输出统计结果
# ---------------------
print(f"\n{'=' * 50}")
print("实验结果统计")
print(f"{'=' * 50}")
print(f"总样本数: {total_predictions}")
print(f"正确预测: {correct_predictions}")
print(f"准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"置信度提取率: {confidence_extraction_rate:.4f} ({confidence_extraction_rate * 100:.2f}%)")
if average_confidence is not None:
    print(f"平均置信度: {average_confidence:.4f}")
print(f"结果已保存到: {csv_filename}")

# 显示前几个样本的结果
print(f"\n前5个样本的结果:")
print(f"{'=' * 50}")
for i, row in df.head().iterrows():
    status = "✓" if row['correct'] else "✗"
    conf_display = f"{row['confidence']:.3f}" if pd.notna(row['confidence']) else "N/A"
    print(f"样本 {i}: 真实={row['Y']}, 预测={row['Yhat']}, 置信度={conf_display} {status}")

# 可选：保存详细的统计报告
stats_report = {
    "total_samples": total_predictions,
    "correct_predictions": correct_predictions,
    "accuracy": accuracy,
    "confidence_extraction_rate": confidence_extraction_rate,
    "average_confidence": average_confidence,
    "model_path": MODEL_PATH,
    "dataset": "abstract_algebra"
}

with open("experiment_stats.json", "w", encoding='utf-8') as f:
    json.dump(stats_report, f, indent=2, ensure_ascii=False)

print(f"\n详细统计已保存到: experiment_stats.json")