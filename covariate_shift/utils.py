import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import nquad

import numpy as np
from scipy.stats import multivariate_normal


def total_variation_monte_carlo(mu1, cov1, mu2, cov2, n_samples=100000):
    """
    Estimate TVD using Monte Carlo sampling - works for high dimensions.
    """
    dist1 = multivariate_normal(mu1, cov1)
    dist2 = multivariate_normal(mu2, cov2)

    # Sample from a proposal distribution (mixture of both)
    samples1 = dist1.rvs(n_samples // 2)
    samples2 = dist2.rvs(n_samples // 2)
    samples = np.vstack([samples1, samples2])

    # Compute densities
    density1 = dist1.pdf(samples)
    density2 = dist2.pdf(samples)

    # TVD = 0.5 * ∫|p(x) - q(x)|dx ≈ 0.5 * mean(|p(x) - q(x)| / r(x))
    # where r(x) is the proposal density
    # Use mixture density as proposal
    density_proposal = 0.5 * dist1.pdf(samples) + 0.5 * dist2.pdf(samples)

    abs_diff = np.abs(density1 - density2)
    tvd = 0.5 * np.mean(abs_diff / density_proposal)

    return tvd


from scipy.linalg import det, inv


def calculate_kl_divergence_multivariate_normal(mu1, cov1, mu2, cov2):
    """
    Calculate KL divergence between two multivariate normal distributions.

    KL(P || Q) where P ~ N(mu1, cov1) and Q ~ N(mu2, cov2)

    Parameters:
    -----------
    mu1 : array-like, shape (n_features,)
        Mean vector of source distribution P
    cov1 : array-like, shape (n_features, n_features)
        Covariance matrix of source distribution P
    mu2 : array-like, shape (n_features,)
        Mean vector of target distribution Q
    cov2 : array-like, shape (n_features, n_features)
        Covariance matrix of target distribution Q

    Returns:
    --------
    kl_divergence : float
        KL(P || Q)
    """
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    cov1 = np.asarray(cov1)
    cov2 = np.asarray(cov2)

    n = len(mu1)

    # Difference in means
    mu_diff = mu2 - mu1

    # Inverse of target covariance
    cov2_inv = inv(cov2)

    # Term 1: trace term
    trace_term = np.trace(cov2_inv @ cov1)

    # Term 2: quadratic form with mean difference
    mean_term = mu_diff.T @ cov2_inv @ mu_diff

    # Term 3: log determinant ratio
    det_term = np.log(det(cov2) / det(cov1))

    # KL divergence formula for multivariate normals
    kl_divergence = 0.5 * (trace_term + mean_term + det_term - n)

    return kl_divergence
