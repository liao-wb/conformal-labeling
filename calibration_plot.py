"""
Calibration Effect Analysis Script.

This script evaluates the impact of confidence calibration on the performance
of the selective labeling algorithm. It compares False Discovery Rate (FDR)
and Power metrics before and after applying calibration.

Paper: Selective Labeling with False Discovery Rate Control
"""

import argparse
import os
import sys
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure the project root is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.select_alg import selection

# --- Constants ---
# Define column names here to avoid hardcoding deep in the logic
COL_LABEL = "Y"
COL_PRED = "Y_hat_resnet152"  # Update this if using a different model column
COL_CONF = "confidence"
COL_CALIB_CONF = "calibrated_confidence"

# Plotting Style
COLORS = [
    '#4682B4',  # Steel Blue
    '#E69F00',  # Orange
    '#4DB99D',  # Teal
    '#FF6200',  # Bright Orange
    "#6495ED"   # Cornflower Blue
]
MARKERS = ['o', 's', 'D', '^', 'v']


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Run calibration comparison experiments.")
    
    # Algorithm parameters
    parser.add_argument("--algorithm", type=str, default="cbh", 
                        choices=["bh", "sbh", "cbh", "qbh", "integrative"],
                        help="Selection algorithm to use.")
    parser.add_argument("--lam", type=float, default=0.94, dest="lambda_val",
                        help="Lambda parameter for Storey-BH.")
    parser.add_argument("--k_0", type=int, default=10000, 
                        help="k_0 parameter for Quantile-BH.")

    # Dataset parameters
    parser.add_argument("--datasets", type=str, default="calibration_imagenet",
                        help="Name of the dataset file (without extension).")
    parser.add_argument("--data_dir", type=str, default="./datasets",
                        help="Directory containing the dataset CSVs.")
    
    # Experiment control
    parser.add_argument("--calib_ratio", type=float, default=0.2, 
                        help="Ratio of data used for calibration.")
    parser.add_argument("--num_trials", type=int, default=100, 
                        help="Number of Monte Carlo trials.")
    parser.add_argument("--alpha", type=float, default=0.1, 
                        help="Target FDR level (for single run).")
    
    return parser.parse_args()


def load_data(data_dir: str, dataset_name: str) -> pd.DataFrame:
    """Loads the dataset from a CSV file."""
    file_path = os.path.join(data_dir, f"{dataset_name}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    
    return pd.read_csv(file_path)


def run_experiment(
    data: pd.DataFrame, 
    args: argparse.Namespace, 
    alpha_list: List[float]
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Runs the selection experiment for both uncalibrated and calibrated scores.

    Args:
        data: The dataset containing labels and predictions.
        args: Parsed command line arguments.
        alpha_list: List of target FDR levels to evaluate.

    Returns:
        A tuple containing (fdr_results, power_results).
        Each is a list of lists: [uncalibrated_metrics, calibrated_metrics].
    """
    # Extract arrays once
    Y = data[COL_LABEL].to_numpy()
    Yhat = data[COL_PRED].to_numpy()
    conf_raw = data[COL_CONF].to_numpy()
    conf_calib = data[COL_CALIB_CONF].to_numpy()
    
    n_samples = len(Y)
    n_calib = int(n_samples * args.calib_ratio)

    # Initialize storage: index 0 -> Raw, index 1 -> Calibrated
    avg_fdrs = [[], []]
    avg_powers = [[], []]

    print(f"Running experiments on {args.datasets} with {args.num_trials} trials...")

    for alpha in tqdm(alpha_list, desc="Target FDR Levels"):
        trial_fdrs = [[], []]
        trial_powers = [[], []]

        for _ in range(args.num_trials):
            # Random split for calibration/test set
            cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)
            
            # 1. Uncalibrated Selection
            fdp, power, _, _ = selection(
                Y, Yhat, conf_raw, cal_indices, alpha, args,
                calib_ratio=args.calib_ratio, random=True
            )
            trial_fdrs[0].append(fdp)
            trial_powers[0].append(power)

            # 2. Calibrated Selection
            fdp_c, power_c, _, _ = selection(
                Y, Yhat, conf_calib, cal_indices, alpha, args,
                calib_ratio=args.calib_ratio, random=True
            )
            trial_fdrs[1].append(fdp_c)
            trial_powers[1].append(power_c)

        # Average over trials
        avg_fdrs[0].append(np.mean(trial_fdrs[0]))
        avg_powers[0].append(np.mean(trial_powers[0]))
        
        avg_fdrs[1].append(np.mean(trial_fdrs[1]))
        avg_powers[1].append(np.mean(trial_powers[1]))

    return avg_fdrs, avg_powers


def plot_results(
    alpha_list: List[float], 
    fdr_results: List[List[float]], 
    power_results: List[List[float]],
    output_filename: str = "calibration_comparison.pdf"
):
    """Generates and saves the FDR and Power comparison plots."""
    
    labels = ["Before Calibration", "After Calibration"]
    
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # --- Plot 1: FDR Comparison ---
    # Plot diagonal reference line (Target FDR)
    ax1.plot(
        alpha_list, alpha_list,
        label="Target FDR",
        linestyle="--", color='gray', linewidth=1.5, alpha=0.7
    )

    for i in range(2):
        ax1.plot(
            alpha_list, fdr_results[i],
            marker=MARKERS[i],
            label=labels[i],
            markersize=8, linestyle='-', linewidth=3, color=COLORS[i], alpha=0.9
        )

    ax1.set_xlabel(r"Target FDR Level ($\alpha$)", fontsize=20)
    ax1.set_ylabel("Realized FDR", fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(loc='upper left', fontsize=16, framealpha=1, shadow=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Plot 2: Power Comparison ---
    for i in range(2):
        ax2.plot(
            alpha_list, power_results[i],
            marker=MARKERS[i],
            label=labels[i],
            markersize=8, linestyle='-', linewidth=3, color=COLORS[i], alpha=0.9
        )

    ax2.set_xlabel(r"Target FDR Level ($\alpha$)", fontsize=20)
    ax2.set_ylabel("Power", fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.legend(loc='lower right', fontsize=16, framealpha=1, shadow=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    print(f"Saving plot to {output_filename}")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    args = parse_arguments()
    
    # Configuration
    alpha_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    try:
        data = load_data(args.data_dir, args.datasets)
    except FileNotFoundError as e:
        print(e)
        return

    # Run Simulation
    fdr_res, power_res = run_experiment(data, args, alpha_levels)

    # Visualization
    plot_results(alpha_levels, fdr_res, power_res)


if __name__ == "__main__":
    main()
