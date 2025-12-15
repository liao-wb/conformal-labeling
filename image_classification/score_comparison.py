
"""
Uncertainty Score Comparison Script.
Computes various uncertainty metrics: MSP, Energy, Entropy, Alpha-Score, React, Odin.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_model, get_dataset, get_device

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet34")
    parser.add_argument("--output_dir", type=str, default="./results")
    
    # Advanced Score Parameters
    parser.add_argument("--temperature", type=float, default=1000.0, help="Temp for ODIN")
    parser.add_argument("--epsilon", type=float, default=0.0014, help="Noise magnitude for ODIN")
    parser.add_argument("--enable_react", action="store_true", help="Enable ReAct score (requires hooks)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()
    
    # Load Model
    model = load_model(args.model, device)
    dataset, label_remap = get_dataset(args.dataset, args.data_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Optional ReAct Setup (Feature extraction)
    # Note: This is model-specific (ResNet structure assumed). 
    # For a general solution, hooks are better, but keeping your logic here for consistency.
    feature_extractor = None
    classifier_head = None
    if args.enable_react and "resnet" in args.model:
        feature_extractor = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4
        ).eval()
        classifier_head = nn.Sequential(
            model.avgpool, nn.Flatten(), model.fc
        ).eval()

    # Storage
    results = {
        "Y": [], "Yhat": [], 
        "msp": [], "energy": [], "entropy": [], "alpha": [],
        "odin": [] 
    }
    if args.enable_react:
        results["react"] = []

    print(f"Computing scores for {args.dataset} using {args.model}...")

    # Main Loop (Standard Scores)
    for data, target in tqdm(dataloader, desc="Scoring"):
        data = data.to(device)
        
        # 1. Standard Forward Pass (No Grad)
        with torch.no_grad():
            logits = model(data)
            probs = torch.softmax(logits, dim=-1)
            conf, y_pred = torch.max(probs, dim=-1)
            
            # MSP
            results["msp"].extend(conf.cpu().numpy())
            
            # Energy
            energy = torch.logsumexp(logits, dim=-1)
            results["energy"].extend(energy.cpu().numpy())
            
            # Entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1) # Added epsilon for stability
            results["entropy"].extend(entropy.cpu().numpy())

            # Alpha Score (Sum of squares)
            alpha_score = torch.sum(probs ** 2, dim=-1)
            results["alpha"].extend(alpha_score.cpu().numpy())
            
            results["Yhat"].extend(y_pred.cpu().numpy())
            
            # ReAct (If enabled)
            if args.enable_react and feature_extractor:
                feats = feature_extractor(data)
                feats = torch.clamp(feats, max=1.0) # Rectification
                r_logits = classifier_head(feats)
                r_probs = torch.softmax(r_logits, dim=-1)
                r_conf = r_probs.max(dim=-1)[0]
                results["react"].extend(r_conf.cpu().numpy())

        # 2. ODIN Forward Pass (Requires Grad for Input Perturbation)
        # We process ODIN separately to keep the logic clean, though it doubles compute for this batch
        data_odin = data.clone().detach().requires_grad_(True)
        logits_o = model(data_odin)
        logits_o = logits_o / args.temperature
        
        # Calculate pseudo-loss to get gradients
        pred_o = logits_o.max(1)[1]
        loss = torch.nn.functional.cross_entropy(logits_o, pred_o)
        loss.backward()
        
        # FGSM-style perturbation
        gradient = data_odin.grad.data
        data_p = data_odin - args.epsilon * torch.sign(-gradient)
        data_p = torch.clamp(data_p, 0, 1)
        
        with torch.no_grad():
            logits_p = model(data_p) / args.temperature
            probs_p = torch.softmax(logits_p, dim=-1)
            conf_odin = probs_p.max(dim=-1)[0]
            results["odin"].extend(conf_odin.cpu().numpy())

        # Labels
        targets_np = target.numpy()
        if label_remap:
            targets_np = np.array([label_remap[t] for t in targets_np])
        results["Y"].extend(targets_np)

    # Save
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    df = pd.DataFrame(results)
    out_file = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_scores.csv')
    df.to_csv(out_file, index=False)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
