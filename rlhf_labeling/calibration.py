"""
Temperature Scaling for RLHF/Classification Models.
Splits dataset, optimizes temperature T, and saves calibrated confidence scores.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_dataset, format_example, save_result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen3-8B")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="mmlu")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--calib_ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser.parse_args()

def extract_constrained_logits(outputs, label_list):
    """Extracts logits corresponding strictly to the label_list options."""
    all_logits = []
    
    for output in outputs:
        top_logprobs = output.outputs[0].logprobs[0]
        row_logits = []
        for label in label_list:
            val = -1e9
            for _, lp_obj in top_logprobs.items():
                if lp_obj.decoded_token.strip() == label:
                    val = lp_obj.logprob
                    break
            row_logits.append(val)
        all_logits.append(row_logits)
        
    return torch.tensor(all_logits, device="cuda")

def main():
    args = parse_args()
    
    # 1. Prepare Data
    full_dataset, label_list = get_dataset(args)
    label_map = {l: i for i, l in enumerate(label_list)}
    
    # Split
    split_idx = int(len(full_dataset) * args.calib_ratio)
    cal_data = full_dataset[:split_idx]
    test_data = full_dataset[split_idx:]
    
    print(f"Calibration size: {len(cal_data)}, Test size: {len(test_data)}")

    # 2. Init Model
    model_path = args.model
    if args.model_dir:
        model_path = os.path.join(args.model_dir, args.model)
        
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.6,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )
    
    sampling_params = SamplingParams(
        guided_decoding=GuidedDecodingParams(choice=label_list),
        logprobs=20, max_tokens=1, temperature=0.0
    )

    # 3. Optimize Temperature on Calibration Set
    print("Generating logits for calibration set...")
    cal_prompts = [format_example(ex, args.dataset)[0] for ex in cal_data]
    cal_labels_idx = [label_map[ex['answer']] for ex in cal_data]
    
    cal_outputs = llm.generate(cal_prompts, sampling_params)
    cal_logits = extract_constrained_logits(cal_outputs, label_list)
    cal_labels_tensor = torch.tensor(cal_labels_idx, device="cuda")
    
    # Optimization Loop
    T = torch.tensor(1.0, device="cuda", requires_grad=True)
    optimizer = torch.optim.Adam([T], lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    dataset_t = TensorDataset(cal_logits, cal_labels_tensor)
    loader_t = DataLoader(dataset_t, batch_size=256, shuffle=True)
    
    print("Optimizing temperature...")
    for _ in tqdm(range(args.max_epochs)):
        for batch_logits, batch_y in loader_t:
            optimizer.zero_grad()
            loss = loss_fn(batch_logits / T, batch_y)
            loss.backward()
            optimizer.step()
            
    optimal_T = T.item()
    print(f"Optimal Temperature: {optimal_T:.4f}")

    # 4. Evaluate on Test Set
    print("Generating logits for test set...")
    test_prompts = [format_example(ex, args.dataset)[0] for ex in test_data]
    test_ground_truths = [ex['answer'] for ex in test_data]
    
    test_outputs = llm.generate(test_prompts, sampling_params)
    test_logits = extract_constrained_logits(test_outputs, label_list)
    
    # Calculate Confidences
    probs_raw = torch.softmax(test_logits, dim=-1)
    probs_cal = torch.softmax(test_logits / optimal_T, dim=-1)
    
    results = {
        "Yhat": [], "Y": [], "is_correct": [],
        "confidence_raw": [], "confidence_calibrated": []
    }
    
    confs_raw, preds_idx = torch.max(probs_raw, dim=-1)
    confs_cal, _ = torch.max(probs_cal, dim=-1) # Preds shouldn't change with T scaling
    
    for i in range(len(test_data)):
        pred_label = label_list[preds_idx[i].item()]
        true_label = test_ground_truths[i]
        
        results['Yhat'].append(pred_label)
        results['Y'].append(true_label)
        results['is_correct'].append(pred_label == true_label)
        results['confidence_raw'].append(confs_raw[i].item())
        results['confidence_calibrated'].append(confs_cal[i].item())
        
    save_result(args, results)

if __name__ == "__main__":
    main()
