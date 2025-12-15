import numpy as np
from scipy.stats import bernoulli, norm


def pac_labeling(Y, Yhat, loss, error_hyperparameter, prob_hyperparameter, uncertainty, pi, num_draws, asymptotic=True):
    n = len(Y)

    # Initialize labels
    Y_tilde = Yhat.copy()
    pi_min = np.min(pi)
    lams = np.sort(uncertainty)
    labeled_inds = np.zeros(n)

    ik_samples = np.random.choice(range(n), size=num_draws)
    xi_samples = bernoulli.rvs([pi[ik] for ik in ik_samples])
    ik_samples_unique = np.unique(ik_samples)
    calibration_idx = np.zeros(n)

    for ik in ik_samples_unique:
        calibration_idx[ik] = 1
        labeled_inds[ik] = xi_samples[np.where(ik_samples == ik)].any()
        if labeled_inds[ik]:
            Y_tilde[ik] = Y[ik]

    n_unique = labeled_inds.mean()
    losses = loss(Y[:, None], Yhat[:, None])
    base_errors = losses[ik_samples] * xi_samples / pi[ik_samples]

    ub = np.max(losses) / pi_min
    lb_index = 0
    ub_index = len(lams) - 1
    while lb_index <= ub_index:
        j = (lb_index + ub_index) // 2
        lam = lams[j]
        errors = base_errors * (uncertainty[ik_samples] < lam)
        err_ci = mean_ci(errors, prob_hyperparameter, asymptotic=asymptotic, ub=ub)
        if err_ci[1] > error_hyperparameter:
            ub_index = j - 1
        else:
            lb_index = j + 1

    lam = lams[lb_index - 1]
    labeled_inds[uncertainty >= lam] = 1
    Y_tilde[uncertainty >= lam] = Y[uncertainty >= lam]
    selection_idx = (calibration_idx == 0) * (uncertainty < lam)
    selection_size = np.sum(selection_idx)
    Y_selected = Y[selection_idx]
    Y_hat_selected = Yhat[selection_idx]
    fdp = np.sum(Y_selected != Y_hat_selected) / selection_size
    power = np.sum(Y_selected == Y_hat_selected) / np.sum(Y == Yhat)
    return fdp, power, selection_size

def mean_ci(Z, alpha, asymptotic=True, ub=1):
    if asymptotic:
        ci = [Z.mean() - norm.isf(alpha)*Z.std()/np.sqrt(len(Z)),
                  Z.mean() + norm.isf(alpha)*Z.std()/np.sqrt(len(Z))]
    else:
        ci = wsr(Z, alpha, u=ub)
    return ci


def wsr(x_n, alpha, grid=np.linspace(1e-5, 1 - 1e-5, 100000), l=0, u=1, num_cpus=10, parallelize: bool = False,
        intersection: bool = True,
        theta: float = 0, c: float = 0.75):
    # Non-asymptotic confidence interval by Waudby-Smith and Ramdas
    x_n = (x_n - l) / (u - l)

    n = x_n.shape[0]
    t_n = np.arange(1, n + 1)
    muhat_n = (0.5 + np.cumsum(x_n)) / (1 + t_n)
    sigma2hat_n = (0.25 + np.cumsum(np.power(x_n - muhat_n, 2))) / (1 + t_n)
    sigma2hat_tminus1_n = np.append(0.25, sigma2hat_n[: -1])
    assert (np.all(sigma2hat_tminus1_n > 0))
    lambda_n = np.sqrt(2 * np.log(2 / alpha) / (n * sigma2hat_tminus1_n))

    def M(m):
        lambdaplus_n = np.minimum(lambda_n, c / m)
        lambdaminus_n = np.minimum(lambda_n, c / (1 - m))
        return np.maximum(
            theta * np.exp(np.cumsum(np.log(1 + lambdaplus_n * (x_n - m)))),
            (1 - theta) * np.exp(np.cumsum(np.log(1 - lambdaminus_n * (x_n - m))))
        )

    indicators_gxn = np.zeros([grid.size, n])
    found_lb = False
    for m_idx, m in enumerate(grid):
        m_n = M(m)
        indicators_gxn[m_idx] = m_n < 1 / alpha
        if not found_lb and np.prod(indicators_gxn[m_idx]):
            found_lb = True
        if found_lb and not np.prod(indicators_gxn[m_idx]):
            break  # since interval, once find a value that fails, stop searching
    if intersection:
        ci_full = grid[np.where(np.prod(indicators_gxn, axis=1))[0]]
    else:
        ci_full = grid[np.where(indicators_gxn[:, -1])[0]]
    if ci_full.size == 0:  # grid maybe too coarse
        idx = np.argmax(np.sum(indicators_gxn, axis=1))
        if idx == 0:
            return np.array([grid[0], grid[1]])
        return np.array([grid[idx - 1], grid[idx]])
    return [ci_full.min() * (u - l) + l, ci_full.max() * (u - l) + l]  # only output the interval