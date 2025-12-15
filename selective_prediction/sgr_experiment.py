"""
SGR Experiment Runner.
Runs the Selection with Guaranteed Risk (SGR) baseline on specified datasets.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selective_prediction.risk_control import RiskControl

# Note: You need to implement/import get_data from your algorithm.preprocess
# Since I don't have the full algorithm folder content yet, I will mock the import 
# or assume it's available.
try:
    from algorithm.preprocess import get_data
except ImportError:
    # Dummy placeholder if algorithm package is missing in this context
    def get_data(ds_name):
        print(f"Mock loading {ds_name}")
        # Return Y, Yhat, Confidence
        return np.random.randint(0, 2, 1000), np.random.randint(0, 2, 1000), np.random.rand(1000)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Qwen3-32B_mmlu")
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--calib_ratio", type=float, default=0.1, help="Ratio of data used for BOUND estimation")
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--alpha", default=0.1, type=float, help="Target Risk Bound (rstar)")
    parser.add_argument("--delta", default=0.1, type=float, help="Confidence parameter (1-delta)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Dataset handling
    # If using your get_data util, ensure it can find the files.
    # Here we assume get_data loads from args.dataset
    
    try:
        # Load data (Y: Ground Truth, Yhat: Prediction, Confidence: Score)
        # You might need to adjust this depending on how get_data is implemented
        Y, Yhat, confidence = get_data(args.dataset)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback for testing with CSV directly
        csv_path = os.path.join(args.data_dir, f"{args.dataset}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            Y = df["Y"].to_numpy()
            if "Yhat" in df.columns:
                Yhat = df["Yhat"].to_numpy()
            else: 
                Yhat = df["Yhat (GPT4o)"].to_numpy() # Handle LLM case
            confidence = df["confidence"].to_numpy()
        else:
            print("Could not load data.")
            return

    risk_list = []
    power_list = []
    size_list = []
    
    rc = RiskControl()
    
    print(f"Running SGR on {args.dataset} (Target Risk={args.alpha}, Delta={args.delta})...")
    
    for _ in tqdm(range(args.num_trials)):
        # Calculate residuals (1 if error, 0 if correct)
        residuals = (Yhat != Y).astype(int)
        
        # Determine split size
        # args.calib_ratio is the size of the calibration set (for bound)
        # So val_size (for test) = 1 - calib_ratio
        val_size = 1.0 - args.calib_ratio
        
        # Run Bound Calculation
        [theta, bound], test_risk, test_power, sel_size = rc.bound(
            rstar=args.alpha,
            delta=args.delta,
            kappa=confidence,
            residuals=residuals,
            split=True,
            val_size=val_size
        )
        
        risk_list.append(test_risk)
        power_list.append(test_power)
        size_list.append(sel_size)

    # Statistics
    mean_risk = np.mean(risk_list)
    quantile_90_risk = np.quantile(risk_list, 0.9)
    mean_power = np.mean(power_list)
    budget_save = np.mean(size_list) / (len(Y) * (1 - args.calib_ratio)) # Normalize by test set size

    print("\n=== SGR Results ===")
    print(f"Target Risk: {args.alpha}")
    print(f"Realized Risk (Mean): {mean_risk:.4f}")
    print(f"Realized Risk (90% Quantile): {quantile_90_risk:.4f}")
    print(f"Mean Power: {mean_power * 100:.2f}%")
    print(f"AI-Labeled Ratio (on Test Set): {budget_save * 100:.2f}%")
    
    base_error = 1.0 - (np.sum(Y == Yhat) / len(Y))
    print(f"Base Model Error Rate: {base_error * 100:.2f}%")

if __name__ == "__main__":
    main()
