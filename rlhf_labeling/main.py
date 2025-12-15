"""
Main Inference Script for RLHF and Classification Tasks using vLLM.
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# Ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_dataset, format_example, save_result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model_dir", type=str, default=None, 
                        help="Optional: Base directory for models if not using HuggingFace hub")
    parser.add_argument("--dataset", type=str, default="mathqa")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Resolve Model Path
    model_path = args.model
    if args.model_dir:
        model_path = os.path.join(args.model_dir, args.model)
    
    print(f"Loading dataset: {args.dataset}")
    dataset, label_list = get_dataset(args)
    print(f"Dataset size: {len(dataset)}")

    print(f"Initializing vLLM: {model_path}")
    model = LLM(
        model=model_path,
        gpu_memory_utilization=0.7,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )

    # Setup Guided Decoding
    guided_decoding = GuidedDecodingParams(choice=label_list)
    sampling_params = SamplingParams(
        guided_decoding=guided_decoding, 
        logprobs=20, 
        max_tokens=1,
        temperature=0.0
    )

    # Prepare Prompts
    prompts, ground_truths = [], []
    for example in dataset:
        p, ans = format_example(example, dataset_name=args.dataset)
        prompts.append(p)
        ground_truths.append(ans)

    # Generate
    print("Generating responses...")
    outputs = model.generate(prompts, sampling_params)

    # Process Output
    results = {
        "Yhat": [],
        "Y": [],
        "is_correct": [],
        "confidence": [],
    }

    for i, output in tqdm(enumerate(outputs), total=len(outputs)):
        # Extract logprobs for valid labels only
        top_logprobs = output.outputs[0].logprobs[0]
        
        logits = []
        for label in label_list:
            # Find the logprob object matching the label
            # Note: This is sensitive to tokenization (e.g., " A" vs "A")
            # We iterate to match the decoded token.
            found_lp = -1e9
            for token_id, lp_obj in top_logprobs.items():
                if lp_obj.decoded_token.strip() == label:
                    found_lp = lp_obj.logprob
                    break
            logits.append(found_lp)
        
        # Softmax over the constrained choice set
        logits_tensor = torch.tensor(logits, device="cpu") # cpu is fine for scalar extract
        probs = torch.softmax(logits_tensor, dim=-1)
        
        pred_idx = torch.argmax(probs).item()
        pred_label = label_list[pred_idx]
        confidence = probs[pred_idx].item()
        
        results['Yhat'].append(pred_label)
        results['Y'].append(ground_truths[i])
        results['is_correct'].append(pred_label == ground_truths[i])
        results['confidence'].append(confidence)

    # Save
    acc = np.mean(results['is_correct'])
    print(f"Accuracy: {acc*100:.2f}%")
    save_result(args, results)

if __name__ == "__main__":
    main()
