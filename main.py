"""
Main Entry Point for Conformal Labeling Experiments.

This script reproduces the main experimental results (FDR, Power, Budget Save)
reported in the paper "Selective Labeling with False Discovery Rate Control".
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

# Ensure local modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithm.select_alg import selection
from algorithm.preprocess import get_data

# Predefined dataset groups for convenience
DATASET_GROUPS = {
    "vision": ["resnet34_imagenet", "resnet34_imagenetv2"], # Updated names to likely match files
    "text": ["bias", "stance", "misinfo"],
    "all": ["resnet34_imagenet", "resnet34_imagenetv2", "stance", "misinfo", "bias"]
}

def parse_args():
    parser = argparse.ArgumentParser(description="Conformal Labeling Main Experiment Runner")
    
    # Data arguments
    parser.add_argument("--datasets", nargs='+', default=["Qwen3-32B_mmlu"],
                        help="List of dataset names or a group name (vision, text, all).")
    parser.add_argument("--data_dir", type=str, default="./datasets",
                        help="Directory where dataset CSV files are stored.")
    
    # Experimental parameters
    parser.add_argument("--calib_ratio", type=float, default=0.1, 
                        help="Ratio of data used for calibration (default: 0.1).")
    parser.add_argument("--num_trials", type=int, default=1000, 
                        help="Number of Monte Carlo trials.")
    parser.add_argument("--alpha", type=float, default=0.1, 
                        help="Target FDR level (alpha).")
    
    # Algorithm parameters
    parser.add_argument("--algorithm", type=str, default="cbh", 
                        choices=["bh", "sbh", "cbh", "qbh", "integrative"],
                        help="Selection algorithm to use.")
    # Use store_true for boolean flags
    parser.add_argument("--no_random", action="store_false", dest="random",
                        help="Disable randomization in p-value construction (deterministic).")
    parser.set_defaults(random=True)
    
    # Other
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling factor.")
    
    return parser.parse_args()

def resolve_dataset_list(input_names: List[str]) -> List[str]:
    """Expands group names into actual dataset lists."""
    final_list = []
    for name in input_names:
        if name in DATASET_GROUPS:
            final_list.extend(DATASET_GROUPS[name])
        else:
            final_list.append(name)
    return final_list

def run_simulation(dataset_name: str, args) -> None:
    """Runs the experiment for a single dataset."""
    print(f"\n{'='*10} Processing: {dataset_name} {'='*10}")
    
    # 1. Load Data
    try:
        # Assuming get_data handles loading from args.data_dir internally
        # or we might need to modify get_data to accept a path.
        # For now, we assume standard usage from original code.
        Y, Yhat, confidence = get_data(dataset_name)
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        # Fallback: try loading CSV directly if get_data fails
        csv_path = os.path.join(args.data_dir, f"{dataset_name}.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            Y = df["Y"].to_numpy()
            Yhat = df["Yhat"].to_numpy()
            confidence = df["confidence"].to_numpy()
        else:
            print(f"Skipping {dataset_name} (File not found).")
            return

    n_samples = len(Y)
    n_calib = int(n_samples * args.calib_ratio)
    
    # 2. Monte Carlo Simulation
    fdr_list = []
    power_list = []
    selection_size_list = []
    
    # Pass args to the selection function
    # Note: selection function signature in original code expects 'args' object
    
    for _ in tqdm(range(args.num_trials), desc="Trials"):
        # Random split
        cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)
        
        # Run Selection Algorithm
        # Note: 'selection_indices' seems to be the raw indices/mask
        fdp, power, size, _ = selection(
            Y, Yhat, confidence, cal_indices, 
            args.alpha, 
            calib_ratio=args.calib_ratio, 
            random=args.random, 
            args=args
        )
        
        fdr_list.append(fdp)
        power_list.append(power)
        selection_size_list.append(size)

    # 3. Report Results
    mean_fdr = np.mean(fdr_list)
    mean_power = np.mean(power_list)
    avg_selected = np.mean(selection_size_list)
    budget_save = avg_selected / n_samples
    
    # Calculate base model error (Naive approach: trust all AI labels)
    base_acc = np.mean(Y == Yhat)
    base_error = 1.0 - base_acc

    print(f"\nResults for {dataset_name} (Target FDR = {args.alpha}):")
    print(f"  - Realized FDR:    {mean_fdr * 100:.2f}%")
    print(f"  - Statistical Power: {mean_power * 100:.2f}%")
    print(f"  - AI-Labeled Ratio:  {budget_save * 100:.2f}%")
    print(f"  - Base Model Error:  {base_error * 100:.2f}%")

def main():
    args = parse_args()
    
    datasets = resolve_dataset_list(args.datasets)
    
    if not datasets:
        print("No datasets specified.")
        return

    for ds in datasets:
        run_simulation(ds, args)

if __name__ == "__main__":
    main()
