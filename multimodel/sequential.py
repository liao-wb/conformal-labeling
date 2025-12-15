"""
Sequential (Cascading) Selection Experiment.

This script simulates a multi-stage selection process where:
1. Model A selects easy instances.
2. Remaining (unselected) instances are passed to Model B.
3. Model B selects from the remainder, and so on.

This approach simulates a "Human-AI-AI" or "Small Model -> Large Model" pipeline.
"""

import argparse
import os
import sys
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.select_alg import new_selection  # Assumes this exists in your repo

def parse_args():
    parser = argparse.ArgumentParser(description="Sequential Multi-Model Selection")
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--calib_ratio", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.1, help="Target FDR level")
    parser.add_argument("--num_trials", type=int, default=100)
    # Define the sequence of files to act as Model 1, Model 2, Model 3...
    parser.add_argument("--model_files", nargs='+', required=True,
                        help="List of CSV files representing the model sequence (e.g. model_small.csv model_large.csv)")
    return parser.parse_args()

class SequentialSelector:
    """Handles the state of sequential selection across multiple models."""
    
    def __init__(self, alpha: float, args):
        self.alpha = alpha
        self.args = args
        self.total_selected = 0
        self.cumulative_fdr_numerator = 0.0

    def run_stage(
        self, 
        y_true: np.ndarray, 
        y_hat: np.ndarray, 
        confidence: np.ndarray, 
        current_indices: np.ndarray,
        n_calib: int
    ) -> Tuple[np.ndarray, float, float, int]:
        """
        Runs selection on the current subset of data.
        
        Returns:
            remaining_indices: Indices of samples NOT selected (passed to next stage).
            fdr: Realized FDR for this stage.
            power: Power for this stage.
            selection_size: Number of samples selected.
        """
        # Slice data to current active set
        # Note: We assume calibration data is fixed/shared or split beforehand. 
        # Here we follow the logic of passing the subset to the algorithm.
        
        # Split Calibration and Test based on the original indices to maintain consistency
        # In this simulation, we assume the first n_calib samples are calibration, 
        # and we only filter the TEST set.
        
        y_calib = y_true[:n_calib]
        y_hat_calib = y_hat[:n_calib]
        conf_calib = confidence[:n_calib]

        # The test set is whatever remains in 'current_indices' that is > n_calib
        test_mask = current_indices >= n_calib
        active_test_indices = current_indices[test_mask]
        
        # Map global indices to local array positions
        y_test = y_true[active_test_indices]
        y_hat_test = y_hat[active_test_indices]
        conf_test = confidence[active_test_indices]

        if len(y_test) == 0:
            return np.array([]), 0.0, 0.0, 0

        # Run Conformal Selection
        fdr, power, size, selection_mask = new_selection(
            y_calib, y_hat_calib, conf_calib, 
            y_test, y_hat_test, conf_test, 
            self.alpha, random=True, args=self.args
        )
        
        # Identify which global indices were selected
        # selection_mask: 1 if selected, 0 if not (based on the user's code logic)
        # We need the indices where selection_mask == 0 (rejected)
        rejected_local_mask = (selection_mask == 0)
        remaining_indices = active_test_indices[rejected_local_mask]
        
        return remaining_indices, fdr, power, size

def main():
    args = parse_args()
    
    # Load all datasets first to ensure alignment
    # We assume all CSVs have the same length and order (aligned data)
    datasets = []
    for fname in args.model_files:
        path = os.path.join(args.data_dir, fname)
        df = pd.read_csv(path)
        datasets.append(df)
    
    total_samples = len(datasets[0])
    n_calib = int(total_samples * args.calib_ratio)
    
    results_fdr = []
    results_total_saved = []

    print(f"Running {args.num_trials} trials of sequential selection...")
    print(f"Sequence: {args.model_files}")

    for _ in tqdm(range(args.num_trials)):
        # Random shuffle indices for this trial (applied to all datasets uniformly)
        perm = np.random.permutation(total_samples)
        
        # Initialize state
        # We process calibration separately, so current_indices tracks the TEST set mainly
        # But for index mapping, we keep track of global indices.
        current_indices = np.arange(n_calib, total_samples) # Start with full test set
        
        trial_selections = []
        trial_fdrs = []
        
        for stage_idx, df in enumerate(datasets):
            # Get data for this model in the shuffled order
            # We don't shuffle the DF in place to save memory, just index it
            y_true = df["Y"].to_numpy()[perm]
            y_hat = df["Yhat"].to_numpy()[perm]
            # Handle different column names for confidence if necessary
            conf_col = "confidence" if "confidence" in df.columns else f"confidence_{stage_idx}"
            confidence = df[conf_col].to_numpy()[perm]
            
            selector = SequentialSelector(args.alpha, args)
            
            remaining, fdr, power, size = selector.run_stage(
                y_true, y_hat, confidence, current_indices, n_calib
            )
            
            trial_selections.append(size)
            trial_fdrs.append(fdr)
            
            # Update indices for next stage
            current_indices = remaining
            
            if len(current_indices) == 0:
                break

        # Calculate weighted average FDR (Global FDR)
        total_selected = sum(trial_selections)
        if total_selected > 0:
            weighted_fdr = sum(s * f for s, f in zip(trial_selections, trial_fdrs)) / total_selected
        else:
            weighted_fdr = 0.0
            
        results_fdr.append(weighted_fdr)
        results_total_saved.append(total_selected / (total_samples - n_calib))

    print("\n=== Final Results ===")
    print(f"Mean Global FDR: {np.mean(results_fdr):.4f} (Target: {args.alpha})")
    print(f"Mean Portion Selected: {np.mean(results_total_saved):.4f}")
    
    # Detailed stats per stage (based on last trial to keep it simple, or accumulate above)
    print("Variance of FDR:", np.var(results_fdr))

if __name__ == "__main__":
    main()
