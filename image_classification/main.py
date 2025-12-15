"""
Standard Image Classification Inference Script.
Computes predictions, confidence scores, and ground truth labels.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Adjust path to import utils from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_model, get_dataset, get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Image Classification Inference")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--dataset", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--data_root", type=str, required=True, 
                        help="Root directory containing datasets (e.g., /mnt/sharedata...)")
    parser.add_argument("--model", type=str, default="resnet34", help="Model architecture")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    return parser.parse_args()

def run_inference(model, dataloader, device, label_remap=None):
    all_confidences = []
    all_y_hat = []
    all_y_true = []

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Inference"):
            data = data.to(device)
            # Targets stay on CPU to avoid unnecessary transfers until needed
            
            logits = model(data)
            probs = torch.softmax(logits, dim=-1)
            conf, y_pred = torch.max(probs, dim=-1)

            all_confidences.extend(conf.cpu().numpy())
            all_y_hat.extend(y_pred.cpu().numpy())

            # Handle label remapping
            targets_np = target.numpy()
            if label_remap is not None:
                targets_np = np.array([label_remap[t] for t in targets_np])
            
            all_y_true.extend(targets_np)

    return all_y_true, all_y_hat, all_confidences

def main():
    args = parse_args()
    device = get_device()
    print(f"Running on device: {device}")

    # Load Resources
    model = load_model(args.model, device)
    dataset, label_remap = get_dataset(args.dataset, args.data_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Run Inference
    y_true, y_pred, confidences = run_inference(model, dataloader, device, label_remap)

    # Save Results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df = pd.DataFrame({
        'Y': y_true,
        'Yhat': y_pred,
        'confidence': confidences,
    })
    
    output_filename = f"{args.model}_{args.dataset}.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    df.to_csv(output_path, index=False)
    
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Results saved to {output_path}")
    print(f"Top-1 Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
