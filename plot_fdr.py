"""
Plotting Script: FDR vs. Power Comparison.
Compares different selection algorithms (BH, Storey-BH, Quantile-BH, Ours).
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.select_alg import selection

# Constants for plotting
COLORS = ['#4682B4', '#E69F00', '#4DB99D', '#FF6200', "#6495ED"]
MARKERS = ['o', '^', 'D', 's']
ALGO_NAMES = {
    "bh": "BH",
    "sbh": "Storey-BH",
    "qbh": "Quantile BH",
    "cbh": "Conformal (Ours)"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Qwen3-32B_mmlu.csv")
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--calib_ratio", type=float, default=0.1)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--output_file", type=str, default="fdr_power_comparison.pdf")
    return parser.parse_args()

def run_simulation(args, algorithms, alpha_list):
    """Runs the Monte Carlo simulation for each algorithm."""
    
    # Load Data
    data_path = os.path.join(args.data_dir, args.dataset)
    df = pd.read_csv(data_path)
    Y = df["Y"].to_numpy()
    
    # Support different column names
    if "Yhat" in df.columns:
        Yhat = df["Yhat"].to_numpy()
    elif "Yhat (GPT4o)" in df.columns:
        Yhat = df["Yhat (GPT4o)"].to_numpy()
    else:
        raise KeyError("Could not find prediction column (Yhat)")
        
    confidence = df["confidence"].to_numpy()
    n_samples = len(Y)
    n_calib = int(n_samples * args.calib_ratio)

    results = {alg: {"fdr": [], "power": []} for alg in algorithms}

    print(f"Running simulation on {args.dataset}...")
    
    for alg in tqdm(algorithms, desc="Algorithms"):
        # Mock args object for the algorithm function
        alg_args = argparse.Namespace()
        alg_args.algorithm = alg
        # Add other defaults required by select_alg
        alg_args._lambda = 0.94 
        alg_args.k_0 = 10000 

        avg_fdrs = []
        avg_powers = []

        for alpha in alpha_list:
            trial_fdrs = []
            trial_powers = []

            for _ in range(args.num_trials):
                cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)
                
                fdp, power, _, _ = selection(
                    Y, Yhat, confidence, cal_indices, alpha, 
                    alg_args, calib_ratio=args.calib_ratio, random=True
                )
                trial_fdrs.append(fdp)
                trial_powers.append(power)
            
            avg_fdrs.append(np.mean(trial_fdrs))
            avg_powers.append(np.mean(trial_powers))

        results[alg]["fdr"] = avg_fdrs
        results[alg]["power"] = avg_powers

    return results

def plot_fdr_power(alpha_list, results, output_file):
    """Generates the dual-axis plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Common settings
    large_font = 24
    small_font = 20
    lw = 3
    ms = 12

    # --- Plot 1: Realized FDR vs Target FDR ---
    ax1.plot(alpha_list, alpha_list, label="Target FDR", linestyle="--", color='gray', lw=2)
    
    for i, (alg, metrics) in enumerate(results.items()):
        ax1.plot(
            alpha_list, metrics["fdr"],
            label=ALGO_NAMES.get(alg, alg),
            marker=MARKERS[i % len(MARKERS)], color=COLORS[i],
            linestyle='-', linewidth=lw, markersize=ms, alpha=0.8
        )
    
    ax1.set_xlabel(r"Target FDR Level ($\alpha$)", fontsize=large_font)
    ax1.set_ylabel("Realized FDR", fontsize=large_font)
    ax1.tick_params(labelsize=small_font)
    ax1.legend(fontsize=16, loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Plot 2: Power vs Target FDR ---
    for i, (alg, metrics) in enumerate(results.items()):
        ax2.plot(
            alpha_list, metrics["power"],
            label=ALGO_NAMES.get(alg, alg),
            marker=MARKERS[i % len(MARKERS)], color=COLORS[i],
            linestyle='-', linewidth=lw, markersize=ms, alpha=0.8
        )

    ax2.set_xlabel(r"Target FDR Level ($\alpha$)", fontsize=large_font)
    ax2.set_ylabel("Power", fontsize=large_font)
    ax2.tick_params(labelsize=small_font)
    ax2.grid(True, linestyle=':', alpha=0.6)
    # ax2.legend() # Optional, legend already in ax1

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.show()

def main():
    args = parse_args()
    algorithms = ["bh", "sbh", "qbh", "cbh"]
    alpha_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = run_simulation(args, algorithms, alpha_levels)
    plot_fdr_power(alpha_levels, results, args.output_file)

if __name__ == "__main__":
    main()
