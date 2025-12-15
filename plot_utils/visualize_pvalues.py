"""
Visualize P-value Distributions with Rejection Threshold.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.select_alg import get_p_values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--calib_ratio", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    # Load Data (Mocking helper)
    # Ideally import get_data from utils
    try:
        df = pd.read_csv(os.path.join(args.data_dir, f"resnet34_{args.dataset}.csv"))
        Y = df["Y"].to_numpy()
        Yhat = df["Yhat"].to_numpy()
        confidence = df["confidence"].to_numpy()
    except:
        print("Data load failed.")
        return

    n_samples = len(Y)
    n_calib = int(n_samples * args.calib_ratio)
    cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)
    
    dummy_args = argparse.Namespace()
    dummy_args.random = "True"

    # Get P-values
    p_values, y_test, y_test_hat, t = get_p_values(
        Y, Yhat, confidence, cal_indices, args.alpha, 
        dummy_args, calib_ratio=args.calib_ratio, random=True
    )
    
    p_incorrect = p_values[y_test != y_test_hat]
    p_correct = p_values[y_test == y_test_hat]

    # Plot
    sns.set_style("white")
    plt.figure(figsize=(6, 5))

    # KDE
    sns.kdeplot(p_incorrect, color='#7F7F7F', fill=True, alpha=0.2, linewidth=0, label="Incorrect ($H_0$)")
    sns.kdeplot(p_correct, color='#2E8B57', fill=True, alpha=0.2, linewidth=0, label="Correct ($H_1$)")
    
    # Outlines
    sns.kdeplot(p_incorrect, color='#7F7F7F', fill=False, linewidth=2)
    sns.kdeplot(p_correct, color='#2E8B57', fill=False, linewidth=2)

    # Threshold Line
    plt.axvline(t, color='#D62728', linestyle='--', linewidth=2, label=f"Threshold $\\tau$")
    
    # Styling
    plt.xlim(0, 1)
    plt.xlabel("Conformal p-value")
    plt.ylabel("Density")
    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.savefig("pvalue_distribution_viz.pdf", dpi=300)
    print("Saved plot.")
    plt.show()

if __name__ == "__main__":
    main()
