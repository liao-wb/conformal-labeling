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




# Compute TVD between your source and target
