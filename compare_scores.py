"""
Score Comparison Experiment.
Compares different uncertainty scores (MSP, Entropy, Energy) on OOD/Misclassification tasks.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.select_alg import selection
from algorithm.preprocess import get_ood_data # Assuming this exists based on original code

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="resnet34_imagenet_misclassificationscore")
    parser.add_argument("--calib_ratio", type=float, default=0.1)
    parser.add_argument("--random", action="store_true", default=True)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1, help="Target FDR level")
    parser.add_argument("--algorithm", type=str, default="cbh")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading data for {args.dataset}...")
    try:
        # Expecting get_ood_data to return arrays
        Y, Yhat, msp_conf, entropy_conf, energy_conf = get_ood_data(args.dataset)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    n_samples = len(Y)
    n_calib = int(n_samples * args.calib_ratio)
    
    metrics = {
        "MSP": {"fdr": [], "power": []},
        "Entropy": {"fdr": [], "power": []},
        "Energy": {"fdr": [], "power": []}
    }

    # Mock args for selection function
    sel_args = argparse.Namespace()
    sel_args.algorithm = args.algorithm
    sel_args._lambda = 0.9 # Default values for BH variants if needed
    sel_args.k_0 = 1000
    
    print(f"Running {args.num_trials} trials...")
    
    for _ in range(args.num_trials):
        cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)
        
        # 1. MSP
        fdr, power, _, _ = selection(
            Y, Yhat, msp_conf, cal_indices, args.alpha, 
            sel_args, calib_ratio=args.calib_ratio, random=args.random
        )
        metrics["MSP"]["fdr"].append(fdr)
        metrics["MSP"]["power"].append(power)

        # 2. Entropy (React in original code var name)
        fdr, power, _, _ = selection(
            Y, Yhat, entropy_conf, cal_indices, args.alpha, 
            sel_args, calib_ratio=args.calib_ratio, random=args.random
        )
        metrics["Entropy"]["fdr"].append(fdr)
        metrics["Entropy"]["power"].append(power)

        # 3. Energy (Alpha in original code var name?)
        # Original code mapped 'alpha_confidence' -> energy logic usually. 
        # Assuming energy_conf is passed correctly here.
        fdr, power, _, _ = selection(
            Y, Yhat, energy_conf, cal_indices, args.alpha, 
            sel_args, calib_ratio=args.calib_ratio, random=args.random
        )
        metrics["Energy"]["fdr"].append(fdr)
        metrics["Energy"]["power"].append(power)

    print("\n=== Results ===")
    for method, res in metrics.items():
        print(f"[{method}] FDR: {np.mean(res['fdr'])*100:.2f}% | Power: {np.mean(res['power'])*100:.2f}%")
        
    ai_acc = np.mean(Y == Yhat) * 100
    print(f"\nBase Model Error Rate: {100 - ai_acc:.2f}%")

if __name__ == "__main__":
    main()
