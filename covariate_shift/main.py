import numpy as np
from scipy.stats import multivariate_normal


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
beta = np.ones(n_features)  # Simple: all features contribute equally; adjust as needed

# Source: mean zero, variance 4 (sigma=2)
mu_source = np.zeros(n_features)
cov_source = np.eye(n_features) * 4  # Isotropic covariance

# Target: shifted mean, variance 1 (sigma=1)
mu_target = np.ones(n_features) * 3
cov_target = np.eye(n_features) * 1

# Generate data
x_source, y_source = generate_data(mu_source, cov_source, n_samples=1000, beta=beta)
x_target, y_target = generate_data(mu_target, cov_target, n_samples=500, beta=beta)

# Compute weights for target points
weights = compute_weights(x_target, mu_source, cov_source, mu_target, cov_target)

# Print a few example weights
print("Sample weights:", weights[:5])