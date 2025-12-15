"""
Ablation Study Plotting.
Analyzes the effect of model size and calibration ratio on FDR and Power.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.select_alg import selection
# Assuming plot_utils.plot exists, otherwise we use local plotting
# from plot_utils.plot import plot_results 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--files", nargs='+', 
                        default=["Qwen3-8B_medmcqa.csv", "Qwen3-32B_medmcqa.csv"],
                        help="List of result files to compare")
    parser.add_argument("--calib_ratios", nargs='+', type=float, 
                        default=[0.05, 0.1, 0.2], 
                        help="Calibration ratios to test for each model")
    parser.add_argument("--num_trials", type=int, default=50)
    return parser.parse_args()

def main():
    args = parse_args()
    alpha_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # To store results for plotting
    summary = []

    for filename in args.files:
        path = os.path.join(args.data_dir, filename)
        if not os.path.exists(path):
            print(f"Skipping {filename}, not found.")
            continue
            
        df = pd.read_csv(path)
        Y = df["Y"].to_numpy()
        Yhat = df["Yhat"].to_numpy()
        confidence = df["confidence"].to_numpy()
        
        model_name = filename.split("_")[0] # e.g. Qwen3-8B

        for ratio in args.calib_ratios:
            print(f"Processing {model_name} with ratio {ratio}...")
            
            fdr_means = []
            power_means = []
            
            # Mock args for selection
            sel_args = argparse.Namespace()
            sel_args.random = "True"
            
            for alpha in tqdm(alpha_list, leave=False):
                trial_fdrs = []
                trial_powers = []
                
                for _ in range(args.num_trials):
                    n_samples = len(Y)
                    n_calib = int(n_samples * ratio)
                    cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)
                    
                    fdp, power, _, _ = selection(
                        Y, Yhat, confidence, cal_indices, alpha, 
                        sel_args, calib_ratio=ratio, random=True
                    )
                    trial_fdrs.append(fdp)
                    trial_powers.append(power)
                
                fdr_means.append(np.mean(trial_fdrs))
                power_means.append(np.mean(trial_powers))
            
            summary.append({
                "label": f"{model_name} (r={ratio})",
                "alpha": alpha_list,
                "fdr": fdr_means,
                "power": power_means
            })

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(alpha_list, alpha_list, 'k--', label="Target")
    for item in summary:
        ax1.plot(item["alpha"], item["fdr"], 'o-', label=item["label"], lw=2)
    
    ax1.set_xlabel("Target FDR")
    ax1.set_ylabel("Realized FDR")
    ax1.legend()
    
    for item in summary:
        ax2.plot(item["alpha"], item["power"], 'o-', label=item["label"], lw=2)
        
    ax2.set_xlabel("Target FDR")
    ax2.set_ylabel("Power")
    
    plt.tight_layout()
    plt.savefig("ablation_study.pdf")
    print("Saved ablation_study.pdf")

if __name__ == "__main__":
    main()
