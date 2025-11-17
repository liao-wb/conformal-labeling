import argparse
from typing import List, Dict
import numpy as np
import json
import os
#from math500_ptrue import PTrueEvaluator
from new_ptrue import PTrueEvaluator






def main():
    parser = argparse.ArgumentParser(description='Evaluate P(True) scores on MATH-500 using vLLM')
    parser.add_argument('--model', type=str, default="DeepSeek-Math-7B",
                        help='Path to the model or model name')
    parser.add_argument('--dataset', type=str, default="./math500.json",
                        help='Path to the MATH-500 dataset file (JSON or JSONL)')
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help='Maximum model context length')
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help='Tensor parallel size for model inference')
    parser.add_argument("--output_dir", type=str, default="math500_results",
                        help='Output directory for results')

    args = parser.parse_args()

    # 加载MATH-500数据集
    print(f"Loading MATH-500 dataset from {args.dataset}...")
    json_file = 'data/MATH500.json'
    dataset = json.load(open(json_file, "r"))
    print(f"Loaded {len(dataset)} math problems")

    evaluator = PTrueEvaluator(args)



    results = evaluator.evaluate_dataset(dataset)



if __name__ == "__main__":
    main()