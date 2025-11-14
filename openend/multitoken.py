import argparse
from typing import List, Dict
import numpy as np
import json
from ptrue_evaluator import PTrueEvaluator
from utils import load_data

def load_dataset(data_path: str) -> List[Dict]:
    """加载数据集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        if data_path.endswith('.jsonl'):
            return [json.loads(line) for line in f]
        else:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Evaluate P(True) scores using vLLM')
    parser.add_argument('--model', type=str, default="Llama-3.1-8B-Instruct",
                        help='Path to the model or model name')
    parser.add_argument('--dataset', type=str, default="mmlu",
                        help='Path to the dataset file (JSON or JSONL)')
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)

    args = parser.parse_args()

    # 加载数据集
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} examples")

    # 初始化评估器
    print(f"Initializing model from {args.model}...")
    evaluator = PTrueEvaluator(args)

    # 运行评估
    results = evaluator.evaluate_dataset(dataset, args.output)

    # 打印统计信息
    p_true_scores = [result['p_true_score'] for result in results]
    print(f"\n=== Evaluation Summary ===")
    print(f"Average P(True) score: {np.mean(p_true_scores):.4f}")
    print(f"Std P(True) score: {np.std(p_true_scores):.4f}")
    print(f"Min P(True) score: {np.min(p_true_scores):.4f}")
    print(f"Max P(True) score: {np.max(p_true_scores):.4f}")

    # 高置信度和低置信度的比例
    high_confidence = sum(1 for score in p_true_scores if score > 0.7) / len(p_true_scores)
    low_confidence = sum(1 for score in p_true_scores if score < 0.3) / len(p_true_scores)
    print(f"High confidence (>0.7): {high_confidence:.2%}")
    print(f"Low confidence (<0.3): {low_confidence:.2%}")


if __name__ == "__main__":
    main()