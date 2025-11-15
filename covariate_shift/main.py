import numpy as np
from scipy.stats import multivariate_normal
from sklearn.ensemble import RandomForestClassifier
from utils import total_variation_monte_carlo
from algorithm.select_alg import selection
import argparse

def generate_data(mu, cov, n_samples, beta):
    """
    Generate multivariate data with logistic conditional P(y|x).

    Parameters:
    - mu: array-like, shape (n_features,), mean vector
    - cov: array-like, shape (n_features, n_features), covariance matrix
    - n_samples: int, number of samples
    - beta: array-like, shape (n_features,), coefficients for logistic

    Returns:
    - x: ndarray, shape (n_samples, n_features)
    - y: ndarray, shape (n_samples,)
    """
    n_features = len(mu)
    x = np.random.multivariate_normal(mu, cov, n_samples)
    # Linear predictor: beta dot x
    linear_pred = np.dot(x, beta)
    p_y1 = 1 / (1 + np.exp(-linear_pred))
    y = np.random.binomial(1, p_y1)
    return x, y


def compute_weights(x, mu_source, cov_source, mu_target, cov_target):
    """
    Compute importance weights w(x) = P_target(x) / P_source(x) for each point in x.

    Parameters:
    - x: ndarray, shape (n_samples, n_features), points to evaluate (e.g., target data)
    - mu_source, cov_source: source distribution parameters
    - mu_target, cov_target: target distribution parameters

    Returns:
    - weights: ndarray, shape (n_samples,)
    """
    dist_source = multivariate_normal(mean=mu_source, cov=cov_source)
    dist_target = multivariate_normal(mean=mu_target, cov=cov_target)
    density_source = dist_source.pdf(x)
    density_target = dist_target.pdf(x)
    weights = density_target / density_source
    # Optional: clip to avoid extremes (e.g., due to numerical issues)
    weights = np.clip(weights, 1e-6, 1e6)
    return weights


# Example usage with 10 features
n_features = 10

# Define beta (coefficients for P(y|x) - fixed across domains)
beta = np.ones(n_features) * 0.3  # Simple: all features contribute equally; adjust as needed

# Source: mean zero, variance 4 (sigma=2)
mu_source = np.zeros(n_features)
cov_source = np.eye(n_features) # Isotropic covariance

# Target: shifted mean, variance 1 (sigma=1)
mu_target = np.ones(n_features) * 0.1
cov_target = np.eye(n_features)

# Generate data
x_train, y_train = generate_data(mu_source, cov_source, n_samples=5000, beta=beta)
x_source, y_source = generate_data(mu_source, cov_source, n_samples=2000, beta=beta)
x_target, y_target = generate_data(mu_target, cov_target, n_samples=2000, beta=beta)

classifier = RandomForestClassifier(n_estimators=500)
classifier.fit(x_train, y_train)

# Compute accuracy
source_accuracy = classifier.score(x_source, y_source)
target_accuracy = classifier.score(x_target, y_target)


print(f"TVD: {total_variation_monte_carlo(mu_source, cov_source, mu_target, cov_target,)}")
print(f"Source Accuracy: {source_accuracy:.4f}")
print(f"Target Accuracy: {target_accuracy:.4f}")


# Get predicted probabilities for each x
source_probs = classifier.predict_proba(x_source)  # Shape: (n_samples, 2)
target_probs = classifier.predict_proba(x_target)  # Shape: (n_samples, 2)

cal_Yhat = np.argmax(source_probs, axis=1)
cal_confidence = np.max(target_probs, axis=1)
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

print(Yhat[:100])
print(confidence[:200])
fdr, power, selection_size, selection_indices = selection(Y, Yhat, confidence, cal_mask, args.alpha, args, calib_ratio=len(cal_Y) / len(Y), random=True)
print(fdr, power)