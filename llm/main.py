"""
Main Inference Script for LLMs using vLLM.
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# Ensure we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_dataset, format_example, save_result, extract_logprobs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                        help="HuggingFace model path or local path")
    parser.add_argument("--dataset", type=str, default="mathqa")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing dataset folders")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--prompt_type", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Data
    dataset, label_list = get_dataset(args)
    print(f"Loaded {len(dataset)} examples from {args.dataset}")

    # 2. Initialize vLLM
    print(f"Initializing vLLM with model: {args.model}")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.6,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )

    # 3. Sampling Parameters
    # Constrain output to valid labels (A, B, C, D...)
    guided_decoding = GuidedDecodingParams(choice=label_list)
    sampling_params = SamplingParams(
        guided_decoding=guided_decoding, 
        logprobs=20,  # Ensure we get logprobs for our choices
        max_tokens=1,
        temperature=0.0 # Deterministic greedy decoding
    )

    # 4. Prepare Prompts
    prompts = []
    ground_truths = []
    
    for example in dataset:
        prompt, answer = format_example(example, prompt_type=args.prompt_type)
        prompts.append(prompt)
        ground_truths.append(answer)

    # 5. Generate
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    # 6. Process Results
    results = {
        "Yhat": [],
        "Y": [],
        "is_correct": [],
        "confidence": [],
        # "logits": [] # Optional: save full logits if needed
    }

    for i, output in tqdm(enumerate(outputs), total=len(outputs), desc="Processing"):
        pred_label, logits = extract_logprobs(output, label_list)
        true_label = ground_truths[i]
        
        # Calculate Softmax Confidence
        # Convert logits list to tensor for calculation
        logits_tensor = torch.tensor(logits)
        probs = torch.softmax(logits_tensor, dim=-1)
        confidence = torch.max(probs).item()

        results['Yhat'].append(pred_label)
        results['Y'].append(true_label)
        results['is_correct'].append(pred_label == true_label)
        results['confidence'].append(confidence)

    # 7. Statistics & Save
    acc = np.mean(results['is_correct'])
    print(f"Final Accuracy: {acc * 100:.2f}%")
    
    save_result(args, results)

if __name__ == "__main__":
    main()
