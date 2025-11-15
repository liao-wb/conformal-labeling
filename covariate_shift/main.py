import numpy as np
from scipy.stats import multivariate_normal
from sklearn.ensemble import RandomForestClassifier
from utils import total_variation_monte_carlo
from algorithm.select_alg import selection
import argparse

def generate_data(mu, cov, n_samples, beta=None):
    """
    Generate multivariate data where P(y|x) is deterministic and fixed:
        y =
            0   if w^T x < -t
            1   if -t <= w^T x <= t
            2   if w^T x > t

    This creates a 3-class classification problem.

    Parameters:
    - mu: mean of P(x)
    - cov: covariance of P(x)
    - n_samples: number of samples
    - beta: ignored

    Returns:
    - x: shape (N, d)
    - y: integer labels in {0,1,2}
    """
    n_features = len(mu)

    # sample x
    x = np.random.multivariate_normal(mu, cov, n_samples)

    # fixed w, fixed P(y|x)
    w = np.ones(n_features) / np.sqrt(n_features)
    logits = x.dot(w)

    # ---- Key: thresholds that produce 3 classes ----
    # You can tune t to control task difficulty!
    t = 0.5

    # assign 3 classes deterministically
    y = np.zeros(n_samples, dtype=int)
    y[logits > t] = 2
    y[(logits >= -t) & (logits <= t)] = 1
    y[logits < -t] = 0

    return x, y



n_features = 10

mu_source = np.zeros(n_features)
cov_source = np.eye(n_features) # Isotropic covariance
# Target: shifted mean, variance 1 (sigma=1)
mu_target = np.zeros(n_features)
cov_target = np.eye(n_features) * 3

beta = np.ones(n_features) * 0.3  # Simple: all features contribute equally; adjust as needed
x_train, y_train = generate_data(mu_source, cov_source, n_samples=5000, beta=beta)

classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(x_train, y_train)

fdr_list = []
power_list = []

for i in range(10):
    # Example usage with 10 features

    # Define beta (coefficients for P(y|x) - fixed across domains)

    # Source: mean zero, variance 4 (sigma=2)

    # Generate data
    #x_train, y_train = generate_data(mu_source, cov_source, n_samples=5000, beta=beta)
    x_source, y_source = generate_data(mu_source, cov_source, n_samples=1000, beta=beta)
    x_target, y_target = generate_data(mu_target, cov_target, n_samples=1000, beta=beta)


    # Compute accuracy
    if i == 0:
        source_accuracy = classifier.score(x_source, y_source)
        target_accuracy = classifier.score(x_target, y_target)

        print(f"Calibration Accuracy: {source_accuracy:.4f}")
        print(f"Test Accuracy: {target_accuracy:.4f}")


    # Get predicted probabilities for each x
    source_probs = classifier.predict_proba(x_source)  # Shape: (n_samples, 2)
    target_probs = classifier.predict_proba(x_target)  # Shape: (n_samples, 2)

    cal_Yhat = np.argmax(source_probs, axis=1)
    cal_confidence = np.max(source_probs, axis=1)
    cal_Y = y_source

    test_Yhat = np.argmax(target_probs, axis=1)
    test_confidence = np.max(target_probs, axis=1)
    test_Y = y_target

    Y = np.concatenate((cal_Y, test_Y), axis=0)
    Yhat = np.concatenate((cal_Yhat, test_Yhat), axis=0)
    confidence = np.concatenate((cal_confidence, test_confidence), axis=0)


    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="imagenetv2")
    parser.add_argument("--calib_ratio", type=float, default=0.1, help="Calibration ratio")
    parser.add_argument("--random", default="True", choices=["True", "False"])
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials")
    parser.add_argument("--alpha", default=0.1, type=float, help="FDR threshold q")
    parser.add_argument("--algorithm", default="cbh", choices=["bh", "sbh", "cbh", "quantbh", "integrative"])
    parser.add_argument("--temperature", type=float, default=1, help="Temperature")
    args = parser.parse_args()

    n_samples = len(Y)
    n_calib = len(cal_Y)
    n_test = n_samples - n_calib
    # Create boolean mask
    cal_mask = np.zeros(len(Y), dtype=bool)
    cal_mask[:n_calib] = True


    fdr, power, selection_size, selection_indices = selection(Y, Yhat, confidence, cal_mask, args.alpha, args, calib_ratio=len(cal_Y) / len(Y), random=True)
    fdr_list.append(fdr)
    power_list.append(power)

print(f"TVD: {total_variation_monte_carlo(mu_source, cov_source, mu_target, cov_target,)}")
print(f"FDR: {np.mean(np.array(fdr_list))}")
print(f"Power: {np.mean(np.array(power_list))}")