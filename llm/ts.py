"""
Temperature Scaling (TS) Calibration for LLMs.
Splits data into calibration/test sets, learns optimal T, and saves calibrated results.
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
from utils import get_dataset, format_example, save_result, extract_logprobs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="mmlu")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--calib_ratio", type=float, default=0.2, help="Ratio of data for calibration")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser.parse_args()

class TemperatureScaler(nn.Module):
    """Learns a scalar temperature value for calibration."""
    def __init__(self, device="cuda"):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) # Initialize > 1
        self.device = device
    
    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels, lr=0.01, epochs=100):
        optimizer = torch.optim.Adam([self.temperature], lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        
        dataset = TensorDataset(logits, labels)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        self.train()
        for _ in tqdm(range(epochs), desc="Optimizing T"):
            for batch_logits, batch_labels in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.forward(batch_logits), batch_labels)
                loss.backward()
                optimizer.step()
        
        return self.temperature.item()

def main():
    args = parse_args()
    
    # 1. Load & Split Data
    dataset, label_list = get_dataset(args)
    
    # Create label mapping (A->0, B->1...) for CrossEntropyLoss
    label_map = {label: i for i, label in enumerate(label_list)}
    
    split_idx = int(len(dataset) * args.calib_ratio)
    cal_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    print(f"Data Split: {len(cal_data)} Calibration, {len(test_data)} Test")

    # 2. Init Model
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.6,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )
    
    guided_decoding = GuidedDecodingParams(choice=label_list)
    sampling_params = SamplingParams(
        guided_decoding=guided_decoding, 
        logprobs=20, 
        max_tokens=1, 
        temperature=0.0
    )

    # 3. Run Inference on ALL Data (Pre-compute logits)
    # Merging prompts to run vLLM once is more efficient, then split results later
    all_prompts = []
    all_labels_idx = []
    
    # Process Calibration Prompts
    for ex in cal_data:
        p, _ = format_example(ex)
        all_prompts.append(p)
        all_labels_idx.append(label_map[ex['answer']])
        
    # Process Test Prompts
    for ex in test_data:
        p, _ = format_example(ex)
        all_prompts.append(p)
        all_labels_idx.append(label_map[ex['answer']])

    print(f"Running inference on {len(all_prompts)} total samples...")
    outputs = llm.generate(all_prompts, sampling_params)

    # 4. Extract Logits
    all_logits = []
    for output in outputs:
        _, logits = extract_logprobs(output, label_list)
        all_logits.append(logits)
    
    all_logits_tensor = torch.tensor(all_logits, device="cuda", dtype=torch.float32)
    all_labels_tensor = torch.tensor(all_labels_idx, device="cuda", dtype=torch.long)

    # Split back into Cal and Test
    cal_logits = all_logits_tensor[:len(cal_data)]
    cal_labels = all_labels_tensor[:len(cal_data)]
    
    test_logits = all_logits_tensor[len(cal_data):]
    test_labels = all_labels_tensor[len(cal_data):]
    test_ground_truths = [ex['answer'] for ex in test_data]

    # 5. Learn Temperature
    scaler = TemperatureScaler().cuda()
    optimal_t = scaler.fit(cal_logits, cal_labels, lr=args.lr, epochs=args.max_epochs)
    print(f"Optimal Temperature: {optimal_t:.4f}")

    # 6. Apply & Evaluate on Test Set
    results = {
        "Yhat": [],
        "Y": [],
        "is_correct": [],
        "confidence_raw": [],
        "confidence_calibrated": []
    }

    # No grad needed for evaluation
    with torch.no_grad():
        # Raw Softmax
        probs_raw = torch.softmax(test_logits, dim=-1)
        confs_raw, preds_idx = torch.max(probs_raw, dim=-1)
        
        # Calibrated Softmax
        probs_cal = torch.softmax(test_logits / optimal_t, dim=-1)
        confs_cal, _ = torch.max(probs_cal, dim=-1)

    # Convert indices back to labels
    for i in range(len(test_data)):
        pred_label = label_list[preds_idx[i].item()]
        true_label = test_ground_truths[i]
        
        results['Yhat'].append(pred_label)
        results['Y'].append(true_label)
        results['is_correct'].append(pred_label == true_label)
        results['confidence_raw'].append(confs_raw[i].item())
        results['confidence_calibrated'].append(confs_cal[i].item())

    save_result(args, results, suffix=f"_T{optimal_t:.2f}")

if __name__ == "__main__":
    main()
