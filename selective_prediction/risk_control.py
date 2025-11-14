import pickle
import numpy as np
from scipy.stats import binom
import scipy
import math
from scipy.optimize import fsolve
import random

class risk_control:

    def calculate_bound(self,delta,m,erm):
        #This function is a solver for the inverse of binomial CDF based on binary search.
        precision = 1e-7
        def func(b):
            return (-1*delta) + scipy.stats.binom.cdf(int(m*erm),m,b)
        a=erm #start binary search from the empirical risk
        c=1   # the upper bound is 1
        b = (a+c)/2 #mid point
        funcval  =func(b)
        while abs(funcval)>precision:
            if a == 1.0 and c == 1.0:
                b = 1.0
                break
            elif funcval>0:
                a=b
            else:
                c=b
            b = (a + c) / 2
            funcval = func(b)
        return b

    def bound(self, rstar, delta, kappa, residuals, split=True):
        # A function to calculate the risk bound proposed in the paper, the algorithm is based on algorithm 1 from the paper.
        # Input: rstar - the requested risk bound
        #       delta - the desired delta
        #       kappa - rating function over the points (higher values is more confident prediction)
        #       residuals - a vector of the residuals of the samples 0 is correct prediction and 1 corresponding to an error
        #       split - is a boolean controls whether to split train and test
        # Output - [theta, bound] (also prints latex text for the tables in the paper)

        # when spliting to train and test this represents the fraction of the validation size
        valsize = 0.9

        probs = kappa
        FY = residuals

        if split:
            idx = list(range(len(FY)))
            random.shuffle(idx)
            slice = round(len(FY) * (1 - valsize))
            FY_val = FY[idx[slice:]]
            probs_val = probs[idx[slice:]]
            FY = FY[idx[:slice]]
            probs = probs[idx[:slice]]
        m = len(FY)

        probs_idx_sorted = np.argsort(probs)

        a = 0
        b = m - 1
        deltahat = delta / math.ceil(math.log2(m))

        for q in range(math.ceil(math.log2(m)) + 1):
            # the for runs log(m)+1 iterations but actually the bound calculated on only log(m) different candidate thetas
            mid = math.ceil((a + b) / 2)

            mi = len(FY[probs_idx_sorted[mid:]])
            theta = probs[probs_idx_sorted[mid]]
            risk = sum(FY[probs_idx_sorted[mid:]]) / mi
            if split:
                selected_indices = probs_val >= theta
                if np.sum(selected_indices) == 0:
                    # Handle the case where no samples are selected
                    testrisk = 1.0  # or appropriate default value
                    testcov = 0.0
                else:
                    testrisk = sum(FY_val[selected_indices]) / np.sum(selected_indices)
                    testcov = np.sum(selected_indices) / len(FY_val)
            bound = self.calculate_bound(deltahat, mi, risk)
            coverage = mi / m
            if bound > rstar:
                a = mid
            else:
                b = mid

        # Calculate power on test dataset
        if split:
            # Convert residuals to boolean mask for correct predictions
            correct_predictions = (FY_val == 0)

            # Samples with confidence ≥ theta
            high_confidence = (probs_val >= theta)

            # True positives: High confidence AND correct predictions
            true_positives = np.sum(high_confidence & correct_predictions)

            # Total correct predictions in test data
            total_correct = np.sum(correct_predictions)

            selection_set_size = len(FY_val[probs_val >= theta])
            # Calculate power
            if total_correct > 0:
                testpower = true_positives / total_correct
            else:
                testpower = 0.0

            print(
                f"rstar: {rstar}, risk: {risk:.4f}, coverage: {coverage:.4f}, testrisk: {testrisk:.4f}, testcov: {testcov:.4f}, bound: {bound:.4f}, testpower: {testpower:.4f}")
            print(f"FDR / testrisk: {testrisk:.4f}, Power: {testpower:.4f}")
            # print("%.2f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f \\\\" % (rstar, risk, coverage, testrisk, testcov, bound, testpower))
        else:
            print("%.2f & %.4f & %.4f & %.4f   \\\\" % (rstar, risk, coverage, bound))
            testpower = None

        return [theta, bound], testrisk, testpower, selection_set_size
