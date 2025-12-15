"""
Temperature Scaling Calibration Script.
Learns an optimal temperature T to calibrate model confidence.
"""

import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_model, get_dataset, get_device

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet152") # Default match orig
    parser.add_argument("--calib_ratio", type=float, default=0.2, help="Ratio of val set for calibration")
    parser.add_argument("--epochs", type=int, default=10, help="Optimization epochs for T")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser.parse_args()

class ModelWithTemperature(torch.nn.Module):
    """Wraps a model to apply Temperature Scaling."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5) # Initialize > 1

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

def main():
    args = parse_args()
    device = get_device()
    
    # Load Model
    orig_model = load_model(args.model, device)
    
    # Load Data & Split
    full_dataset, _ = get_dataset(args.dataset, args.data_root)
    cal_size = int(len(full_dataset) * args.calib_ratio)
    test_size = len(full_dataset) - cal_size
    cal_set, test_set = random_split(full_dataset, [cal_size, test_size], 
                                     generator=torch.Generator().manual_seed(42))

    cal_loader = DataLoader(cal_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=4)

    # --- Step 1: Optimize Temperature ---
    print(f"Optimizing Temperature on {len(cal_set)} samples...")
    # Pre-compute logits to save time (since model is frozen)
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for input, label in tqdm(cal_loader, desc="Pre-computing Logits"):
            input = input.to(device)
            logits_list.append(orig_model(input))
            labels_list.append(label.to(device))
            
    logits_all = torch.cat(logits_list).to(device)
    labels_all = torch.cat(labels_list).to(device)

    # Optimization
    temperature = torch.nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([temperature], lr=args.lr, max_iter=50)

    def eval():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits_all / temperature, labels_all)
        loss.backward()
        return loss

    optimizer.step(eval)
    
    T_val = temperature.item()
    print(f"Optimal Temperature found: {T_val:.4f}")

    # --- Step 2: Evaluation ---
    confidences = []
    calib_confidences = []
    y_hat = []
    y_true = []

    print("Evaluating on Test Set...")
    with torch.no_grad():
        for input, target in tqdm(test_loader, desc="Testing"):
            input = input.to(device)
            
            logits = orig_model(input)
            
            # Uncalibrated
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            
            # Calibrated
            calib_probs = torch.softmax(logits / T_val, dim=-1)
            calib_conf = calib_probs.gather(1, pred.view(-1, 1)).squeeze()

            confidences.extend(conf.cpu().numpy())
            calib_confidences.extend(calib_conf.cpu().numpy())
            y_hat.extend(pred.cpu().numpy())
            y_true.extend(target.numpy())

    # Save
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df = pd.DataFrame({
        'Y': y_true,
        'Y_hat': y_hat,
        'confidence': confidences,
        'calibrated_confidence': calib_confidences,
    })
    
    out_file = os.path.join(args.output_dir, f'{args.model}_calibration_T{T_val:.2f}.csv')
    df.to_csv(out_file, index=False)
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
