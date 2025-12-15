"""
Score Comparison Plotting Script.
Compares the Statistical Power of different uncertainty scores (MSP, Energy, D_alpha)
under Conformal Labeling.
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

# Plotting Constants
COLORS = ['#1f77b4', '#d62728', '#4DB99D', '#FF6200']
MARKERS = ['o', '^', 'D', 's']

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Power of Different Uncertainty Scores")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the CSV file containing scores and labels.")
    # Optional: Allow loading a second file if scores are split (as in original code)
    parser.add_argument("--data_path_aux", type=str, default=None,
                        help="Optional second CSV file if scores are in different files.")
    
    parser.add_argument("--calib_ratio", type=float, default=0.1)
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--output_file", type=str, default="score_power_comparison.pdf")
    
    # Column names mapping
    parser.add_argument("--col_label", type=str, default="Y")
    parser.add_argument("--col_pred", type=str, default="Yhat")
    parser.add_argument("--scores", nargs='+', default=["msp", "energy", "Dalpha"],
                        help="List of score names to compare (must match keys in code or columns)")
    
    return parser.parse_args()

def load_data_vectors(args):
    """Loads data and extracts vectors for Y, Yhat, and various scores."""
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"{args.data_path} not found.")

    df_main = pd.read_csv(args.data_path)
    
    # Dictionary to store vectors: {'Y': ..., 'Yhat': ..., 'msp': ..., 'energy': ...}
    data = {
        'Y': df_main[args.col_label].to_numpy(),
        'Yhat': df_main[args.col_pred].to_numpy()
    }

    # Helper to safely get column
    def get_col(df, name):
        # Map common short names to likely CSV column names
        name_map = {
            "msp": "msp_confidence",
            "energy": "energy_confidence",
            "Dalpha": "alpha_confidence",
            "entropy": "entropy_confidence"
        }
        col = name_map.get(name, name) # Try map, else use raw name
        if col in df.columns:
            return df[col].to_numpy()
        return None

    # Load scores from main file
    for score_name in args.scores:
        vec = get_col(df_main, score_name)
        if vec is not None:
            data[score_name] = vec

    # Load scores from aux file if provided (handling the specific split in original code)
    if args.data_path_aux and os.path.exists(args.data_path_aux):
        df_aux = pd.read_csv(args.data_path_aux)
        # Ensure alignment! (Assuming same order)
        for score_name in args.scores:
            if score_name not in data: # Only look for missing ones
                vec = get_col(df_aux, score_name)
                if vec is not None:
                    data[score_name] = vec
                    # In case Y/Yhat are different in the aux file (e.g. OOD vs Misclassification datasets)
                    # The original code treated them as paired. We assume Y/Yhat from main is ground truth.

    return data

def main():
    args = parse_args()
    
    data_vectors = load_data_vectors(args)
    alpha_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    Y = data_vectors['Y']
    Yhat = data_vectors['Yhat']
    n_samples = len(Y)
    n_calib = int(n_samples * args.calib_ratio)

    results = {}

    # Simulation Loop
    for score_name in tqdm(args.scores, desc="Evaluating Scores"):
        if score_name not in data_vectors:
            print(f"Warning: Score '{score_name}' not found in data. Skipping.")
            continue
            
        confidence = data_vectors[score_name]
        avg_powers = []

        # Dummy args for selection function
        sel_args = argparse.Namespace()
        sel_args.random = "True"
        sel_args.algorithm = "cbh" # Conformal BH

        for alpha in alpha_list:
            powers = []
            for _ in range(args.num_trials):
                cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)
                
                _, power, _, _ = selection(
                    Y, Yhat, confidence, cal_indices, alpha, 
                    sel_args, calib_ratio=args.calib_ratio, random=True
                )
                powers.append(power)
            avg_powers.append(np.mean(powers))
        
        results[score_name] = avg_powers

    # Plotting
    plt.figure(figsize=(10, 8))
    
    large_font = 28
    small_font = 24
    
    for i, (score_name, powers) in enumerate(results.items()):
        label_map = {"Dalpha": r"$D_{\alpha}$", "msp": "MSP", "energy": "Energy"}
        label = label_map.get(score_name, score_name)
        
        plt.plot(
            alpha_list, powers,
            marker=MARKERS[i % len(MARKERS)],
            label=label,
            markersize=14, linestyle='-', linewidth=4, 
            color=COLORS[i % len(COLORS)], alpha=0.8
        )

    plt.xlabel(r"Target FDR Level ($\alpha$)", fontsize=large_font)
    plt.ylabel("Power", fontsize=large_font)
    plt.tick_params(axis='both', which='major', labelsize=small_font)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=small_font, framealpha=1, shadow=True, loc='lower right')

    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300)
    print(f"Plot saved to {args.output_file}")
    plt.show()

if __name__ == "__main__":
    main()
