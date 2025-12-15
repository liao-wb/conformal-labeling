"""
Selection with Guaranteed Risk (SGR) Implementation.
Reference: Geifman & El-Yaniv, 2017.
"""

import math
import random
from typing import Tuple, List, Optional

import numpy as np
import scipy.stats
from scipy.optimize import fsolve

class RiskControl:
    """
    Implements the risk control algorithm (Algorithm 1 from Geifman & El-Yaniv).
    """

    def calculate_bound(self, delta: float, m: int, erm: float) -> float:
        """
        Solves for the inverse of binomial CDF based on binary search.
        Finds the upper bound b such that P(Binom(m, b) <= m*erm) <= delta.
        """
        precision = 1e-7
        
        def func(b):
            # Binomial CDF: Probability of getting <= k successes in n trials with prob b
            k = int(m * erm)
            return (-1 * delta) + scipy.stats.binom.cdf(k, m, b)
        
        a = erm # Start binary search from the empirical risk
        c = 1.0 # The upper bound is 1
        b = (a + c) / 2 # Mid point
        
        funcval = func(b)
        
        # Binary search for the root
        while abs(funcval) > precision:
            if a >= 1.0 and c >= 1.0:
                b = 1.0
                break
                
            if funcval > 0:
                # Need to increase b to reduce CDF value (make event less likely under null)
                # Wait, scipy.binom.cdf(k, n, p) decreases as p increases.
                # If cdf > delta, we need a larger p (b) to make the tail sum smaller?
                # Actually, Geifman's bound is an upper confidence bound.
                a = b
            else:
                c = b
                
            b = (a + c) / 2
            funcval = func(b)
            
        return b

    def bound(
        self, 
        rstar: float, 
        delta: float, 
        kappa: np.ndarray, 
        residuals: np.ndarray, 
        split: bool = True,
        val_size: float = 0.5 # Modified to allow external control
    ) -> Tuple[List[float], float, float, int]:
        """
        Calculates the risk bound and selects a threshold.
        
        Args:
            rstar: The requested risk bound (target error rate).
            delta: The desired confidence level parameter (1 - confidence).
            kappa: Confidence scores (higher is more confident).
            residuals: 0 for correct prediction, 1 for error.
            split: Whether to split data into validation (for bound) and test (for evaluation).
            val_size: Fraction of data to use for TEST/EVALUATION if split is True. 
                      (Note: The original code used valsize=0.9 meaning 90% is TEST, 10% is CALIBRATION/BOUND).
                      
        Returns:
            [theta, bound]: The selected threshold and the calculated bound.
            test_risk: Realized risk on the test set.
            test_power: Realized power (recall) on the test set.
            selection_size: Number of selected samples in test set.
        """
        probs = kappa
        FY = residuals # Failure/Error indicator (1=Error)

        # Split data
        if split:
            idx = np.arange(len(FY))
            np.random.shuffle(idx)
            
            # Original code logic: split_point = len * (1 - valsize)
            # If valsize=0.9, split_point = 0.1 * len.
            # FY[:split_point] -> BOUND estimation (Calibration)
            # FY[split_point:] -> TEST evaluation
            
            # Using clearer naming
            n_samples = len(FY)
            n_calib = int(n_samples * (1 - val_size))
            
            # Calibration set (used to find theta)
            FY_cal = FY[idx[:n_calib]]
            probs_cal = probs[idx[:n_calib]]
            
            # Validation/Test set (used to report metrics)
            FY_val = FY[idx[n_calib:]]
            probs_val = probs[idx[n_calib:]]
        else:
            # Use all data for bound (no test evaluation)
            FY_cal = FY
            probs_cal = probs
            FY_val = None
            probs_val = None
            
        m = len(FY_cal)
        if m == 0:
             raise ValueError("Calibration set is empty. Check split ratio.")

        # Sort by confidence
        probs_idx_sorted = np.argsort(probs_cal)
        
        # Binary search for the optimal theta index
        a = 0
        b = m - 1
        
        # We test log(m) hypotheses to avoid union bound penalty being too large
        # But here we just iterate log(m) times to find the theta?
        # The algorithm in the paper is structural risk minimization over a grid.
        # The code implements a binary search over the sorted indices.
        
        # Correct delta for union bound (if we consider log(m) hypotheses)
        # In Geifman's code, they divide delta by log2(m).
        deltahat = delta / math.ceil(math.log2(m) + 1e-9)

        final_bound = 1.0
        final_theta = 1.0
        
        # Binary Search for the lowest theta that satisfies the bound
        # Range [a, b] are indices in the sorted array. 
        # Higher index -> Higher confidence threshold -> Fewer selected samples -> Lower Risk
        
        # The logic in the original code is a bit specific. It iterates log(m) times.
        # Let's preserve the logic: it updates a/b to narrow down the index "mid".
        
        iterations = math.ceil(math.log2(m)) + 1
        
        for q in range(iterations):
            mid = math.ceil((a + b) / 2)
            mid = min(mid, m - 1)
            
            # Candidate Threshold
            theta_candidate = probs_cal[probs_idx_sorted[mid]]
            
            # Samples with conf >= theta in calibration set
            # Since sorted, these are indices [mid:]
            selected_in_cal = FY_cal[probs_idx_sorted[mid:]]
            mi = len(selected_in_cal) # Number of selected samples
            
            if mi == 0:
                # If nothing selected, risk is 0 but coverage is 0. 
                # We want to lower threshold (move mid left)?
                # Actually if mi=0, we are too strict. b = mid - 1?
                risk = 0.0
            else:
                risk = sum(selected_in_cal) / mi

            # Calculate the UCB bound for this risk
            bound_val = self.calculate_bound(deltahat, mi, risk)
            
            # Check if this bound satisfies our requirement
            if bound_val > rstar:
                # Bound is too high (risk is too high).
                # We need to be more selective -> Increase threshold -> Increase index
                a = mid 
            else:
                # Bound is satisfied.
                # Try to be less selective (decrease threshold) to improve coverage?
                # Binary search usually finds the boundary.
                b = mid
                final_bound = bound_val
                final_theta = theta_candidate
                
            if a >= b - 1:
                break
                
        # Final evaluation on Test Set
        theta = probs_cal[probs_idx_sorted[b]] # Choose the safe side
        
        if split and FY_val is not None:
            selected_indices = probs_val >= theta
            n_selected = np.sum(selected_indices)
            
            if n_selected == 0:
                test_risk = 0.0
                test_cov = 0.0
                test_power = 0.0
            else:
                # Risk: Proportion of errors in selected set
                test_risk = np.sum(FY_val[selected_indices]) / n_selected
                test_cov = n_selected / len(FY_val)
                
                # Power: Proportion of correct samples retrieved
                # Correct predictions in full val set
                total_correct = np.sum(FY_val == 0)
                # Correct predictions in selected set (True Positives)
                correct_selected = np.sum((FY_val == 0) & selected_indices)
                
                if total_correct > 0:
                    test_power = correct_selected / total_correct
                else:
                    test_power = 0.0
            
            return [theta, final_bound], test_risk, test_power, n_selected
            
        else:
            return [theta, final_bound], 0.0, 0.0, 0
