import numpy as np

def selection(Y, Yhat, confidence, cal_indices, alpha, args, calib_ratio=0.5, random=True):
    """"""
    all_indices = np.arange(len(Y))
    test_indices = np.setdiff1d(all_indices, cal_indices)

    y_calib, y_hat_calib, conf_calib = Y[cal_indices], Yhat[cal_indices], confidence[cal_indices]
    y_test, y_hat_test, conf_test = Y[test_indices], Yhat[test_indices], confidence[test_indices]
    n_calib = y_calib.shape[0]
    n_test = y_test.shape[0]

    # H0: y_hat != y
    cal0_idx = (y_hat_calib != y_calib)
    y_calib_0, y_hat_calib_0, conf_calib_0 = y_calib[cal0_idx], y_hat_calib[cal0_idx], conf_calib[cal0_idx]
    n_calib_0 = y_calib_0.shape[0]

    cal0_score = 1 - conf_calib_0
    test_score = 1 - conf_test

    if random:
        p_values = np.sum((test_score[:, None] > cal0_score), axis=-1) + np.random.rand(n_test) * (np.sum((test_score[:, None] == cal0_score), axis=-1) + 1)
        p_values /= (1 + n_calib_0)
    else:
        p_values = np.sum((test_score[:, None] >= cal0_score), axis=-1) + 1
        p_values /= (1 + n_calib_0)

    if args.algorithm == "bh":
        selection_indices, _ = bh(p_values, alpha)
    elif args.algorithm == "cbh":
        selection_indices, _ = bh(p_values, alpha * (1 + n_calib) / (1 + n_calib_0))
    elif args.algorithm == "sbh":
        selection_indices, _ = storeybh(p_values, alpha)
    elif args.algorithm == "qbh":
        selection_indices, _ = quantilebh(p_values, alpha)
    elif args.algorithm == "integrative":
        selection_indices = integrative_bh(Y, Yhat, confidence, alpha, args, calib_ratio, random)
    elif args.algorithm == "by":
        raise NotImplementedError
    elif args.algorithm == "dby":
        raise NotImplementedError
    else:
        raise NotImplementedError

    y_reject, y_hat_reject = y_test[selection_indices], y_hat_test[selection_indices]
    #y_accept, y_hat_accept = y_test[selection_indices == 0], y_hat_test[selection_indices == 0]

    selection_size = y_reject.shape[0]

    fdr = np.sum(y_reject != y_hat_reject) / selection_size if selection_size > 0 else 0
    power = np.sum(y_reject == y_hat_reject) / np.sum(y_test == y_hat_test)

    return fdr, power, selection_size, selection_indices


def new_selection(y_calib, y_hat_calib, conf_calib, y_test, y_hat_test, conf_test, alpha, args, random=True):
    """"""
    n_calib = y_calib.shape[0]
    n_test = y_test.shape[0]

    # H0: y_hat != y
    cal0_idx = (y_hat_calib != y_calib)
    y_calib_0, y_hat_calib_0, conf_calib_0 = y_calib[cal0_idx], y_hat_calib[cal0_idx], conf_calib[cal0_idx]
    n_calib_0 = y_calib_0.shape[0]

    p_values = compute_p_values(conf_calib_0, conf_test, random)
    #p_values = compute_e_2_p_values(conf_calib_0, conf_test, random)

    if args.algorithm == "bh":
        selection_indices, _ = bh(p_values, alpha)
    elif args.algorithm == "cbh":
        selection_indices, _ = bh(p_values, alpha * (1 + n_calib) / (1 + n_calib_0))
    elif args.algorithm == "sbh":
        selection_indices, _ = storeybh(p_values, alpha)
    elif args.algorithm == "qbh":
        selection_indices, _ = quantilebh(p_values, alpha, k_0=args.k_0)
    elif args.algorithm == "by":
        raise NotImplementedError
    elif args.algorithm == "dby":
        raise NotImplementedError
    else:
        raise NotImplementedError

    y_reject, y_hat_reject = y_test[selection_indices], y_hat_test[selection_indices]
    #y_accept, y_hat_accept = y_test[selection_indices == 0], y_hat_test[selection_indices == 0]

    selection_size = y_reject.shape[0]

    fdr = np.sum(y_reject != y_hat_reject) / selection_size if selection_size > 0 else 0
    power = np.sum(y_reject == y_hat_reject) / np.sum(y_test == y_hat_test)

    return fdr, power, selection_size, selection_indices

def compute_p_values(conf_calib_0, conf_test, random):
    n_calib_0 = conf_calib_0.shape[0]
    n_test = conf_test.shape[0]
    cal0_score = 1 - conf_calib_0
    test_score = 1 - conf_test

    if random:
        p_values = np.sum((test_score[:, None] > cal0_score), axis=-1) + np.random.rand(n_test) * (
                    np.sum((test_score[:, None] == cal0_score), axis=-1) + 1)
        p_values /= (1 + n_calib_0)
    else:
        p_values = np.sum((test_score[:, None] >= cal0_score), axis=-1) + 1
        p_values /= (1 + n_calib_0)
    return p_values

def compute_e_2_p_values(conf_calib_0, conf_test, random):
    n_calib_0 = conf_calib_0.shape[0]
    n_test = conf_test.shape[0]
    cal0_score = 1 - conf_calib_0
    test_score = 1 - conf_test

    upper = np.sum(conf_calib_0)
    p_values = upper / conf_test / (n_calib_0)
    return p_values

def reg_selection(y, y_hat, confidence, alpha, args, error, calib_ratio=0.5, random=True):
    """"""
    n_samples = len(y)
    n_calib = int(n_samples * calib_ratio)
    n_test = n_samples - n_calib

    # Randomly select calibration indices (without replacement)
    cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)

    # The remaining indices are for prediction
    test_indices = np.setdiff1d(np.arange(n_samples), cal_indices)

    # Split the dataset
    y_calib, y_hat_calib, conf_calib = y[cal_indices], y_hat[cal_indices], confidence[cal_indices]
    y_test, y_hat_test, conf_test = y[test_indices], y_hat[test_indices], confidence[test_indices]

    # H0: y_hat != y
    cal0_idx = ((y_hat_calib - y_calib)**2 > error)
    y_calib_0, y_hat_calib_0, conf_calib_0 = y_calib[cal0_idx], y_hat_calib[cal0_idx], conf_calib[cal0_idx]
    n_calib_0 = y_calib_0.shape[0]

    cal0_score = 1 - conf_calib_0
    test_score = 1 - conf_test

    if random:
        p_values = np.sum((test_score[:, None] > cal0_score), axis=-1) + np.random.rand(n_test) * (np.sum((test_score[:, None] == cal0_score), axis=-1) + 1)
        p_values /= (1 + n_calib_0)
    else:
        p_values = np.sum((test_score[:, None] >= cal0_score), axis=-1) + 1
        p_values /= (1 + n_calib_0)

    if args.algorithm == "bh":
        selection_indices, _ = bh(p_values, alpha)
    elif args.algorithm == "cbh":
        selection_indices, _ = bh(p_values, alpha * (1 + n_calib) / (1 + n_calib_0))
    elif args.algorithm == "sbh":
        selection_indices = storeybh(p_values, alpha)
    elif args.algorithm == "qbh":
        selection_indices = quantilebh(p_values, alpha, k_0=args.k_0)
    elif args.algorithm == "integrative":
        selection_indices = integrative_bh(y, y_hat, confidence, alpha, args, calib_ratio, random)
    elif args.algorithm == "by":
        raise NotImplementedError
    elif args.algorithm == "dby":
        raise NotImplementedError
    else:
        raise NotImplementedError

    y_reject, y_hat_reject = y_test[selection_indices], y_hat_test[selection_indices]
    #y_accept, y_hat_accept = y_test[selection_indices == 0], y_hat_test[selection_indices == 0]

    selection_size = y_reject.shape[0]

    fdr = np.sum((y_reject - y_hat_reject)**2 > error) / selection_size if selection_size > 0 else 0
    power = np.sum((y_reject - y_hat_reject)**2 <= error) / np.sum((y_test - y_hat_test)**2 <= error)

    return fdr, power, selection_size



def bh(p_values, alpha):
    m = p_values.shape[0]
    p_values_sorted = np.sort(p_values)

    largest_i = 0
    threshold = [k * alpha / m for k in range(1, m + 1)]
    for i in range(m - 1, -1, -1):
        if p_values_sorted[i] <= threshold[i]:
            largest_i = i
            break
    t = threshold[largest_i]

    selection_indices = (p_values <= t)
    return selection_indices, t

def dbh(p_values, alpha, gamma=0.9):
    """
    Dependence-adjusted Benjamini–Hochberg (simplified for independence case).

    Parameters:
        p_values: array of p-values
        alpha: desired FDR level
        gamma: adjustment factor (default 0.9, =1 for positive dependence)
    Returns:
        Boolean array: True = reject
    """
    m = len(p_values)

    # Step 0: baseline BH(γ α) rejection set
    baseline_rejects = bh(p_values, gamma * alpha)
    R_hat = np.sum(baseline_rejects)  # same for all i under independence

    # Step 1: Calibrate each hypothesis
    c_hat = np.zeros(m)
    for i in range(m):
        # Under independence, this is equivalent to finding largest c s.t.
        # p_i <= τ_BH(c) ⇒ E[1{p_i <= τ}/R_hat] ≤ α/m
        # Since τ_BH(c) = c * R_hat / m, solve for c directly
        if p_values[i] <= gamma * alpha * R_hat / m:
            c_hat[i] = alpha  # can go up to α safely under PRDS/independence
        else:
            c_hat[i] = gamma * alpha  # fallback

    # Step 2: Reject if p_i <= τ_BH(c_hat[i])
    tau_i = c_hat * R_hat / m
    rejects = p_values <= tau_i

    return rejects


def _pFDR_hat_from_pi0_gamma(p_values, pi0_hat, gamma):
    """
    Compute a simple plug-in pFDR estimate p̂FDR_lambda(γ) used for MSE selection.
    Based on equation idea in Storey (2002). We'll use the original-data style:
      p̂FDRλ(γ) = (π̂0 * γ) / ( (R_γ / m) * cond_prob )
    where cond_prob = 1 - (1 - γ)^m (prob at least one p <= gamma)
    and R_γ = #{p_i <= γ} (we use R_γ ∨ 1 in denominator to avoid zero).
    This matches the 'pFDR' plug-in flavor in the paper for selection purposes.
    """
    p = np.asarray(p_values)
    m = len(p)
    R_gamma = np.sum(p <= gamma)
    R_gamma = max(R_gamma, 1)   # finite-sample stabilization
    Pr_P_leq_gamma_hat = R_gamma / m
    # conditional probability that at least one p <= gamma
    cond_prob = 1.0 - (1.0 - gamma) ** m
    if Pr_P_leq_gamma_hat <= 0 or cond_prob <= 0:
        return 1.0
    pfdr_hat = (pi0_hat * gamma) / (Pr_P_leq_gamma_hat * cond_prob)
    return float(min(pfdr_hat, 1.0))

def storeybh(p_values, alpha, lambda_grid=None, B=200, gamma=None, random_state=None):
    """
    Storey-adaptive BH with bootstrap selection of lambda per Storey (2002) Algorithm 3.
    Returns: (rejected_mask, pi0_est, best_lambda, adjusted_alpha)
    Parameters:
      p_values: array-like of p-values
      alpha: nominal FDR level
      lambda_grid: iterable of lambda values in [0,1). default np.arange(0.1,1.0,0.1)
      B: number of bootstrap resamples for MSE selection
      gamma: rejection region gamma used in pFDR estimates for MSE selection; default = median(p_values)
      random_state: None or int for reproducibility
    """
    rng = np.random.default_rng(random_state)
    p = np.asarray(p_values)
    m = len(p)
    if lambda_grid is None:
        lambda_grid = np.arange(0.1, 1.0, 0.1)
    lambda_grid = np.asarray(lambda_grid)
    # safe gamma
    if gamma is None:
        gamma = float(np.median(p))
        # ensure gamma not 0 or 1
        gamma = np.clip(gamma, 1e-8, 1 - 1e-8)
    # 1) compute original-data pi0_hat and pFDR_hat for each lambda
    pi0_orig = []
    pfdr_orig = []
    for lam in lambda_grid:
        # Storey's estimator with small +1 stabilization: (1 + #p > lambda) / (m * (1 - lambda))
        # This is a typical stabilized version.
        denom = max(1e-12, (1.0 - lam))
        W_lambda = np.sum(p > lam)
        pi0_hat = (1.0 + W_lambda) / (m * denom)
        pi0_hat = min(pi0_hat, 1.0)
        pi0_orig.append(pi0_hat)
        pfdr_orig.append(_pFDR_hat_from_pi0_gamma(p, pi0_hat, gamma))
    pi0_orig = np.asarray(pi0_orig)
    pfdr_orig = np.asarray(pfdr_orig)
    # plug-in target = min over lambdas of original pFDR estimates (Algorithm 3)
    plug_in_target = float(np.min(pfdr_orig))

    # 2) For each lambda: bootstrap B times, compute pFDR* and MSE relative to plug-in target
    MSEs = np.empty(len(lambda_grid), dtype=float)
    for i, lam in enumerate(lambda_grid):
        boot_pfdr = np.empty(B, dtype=float)
        denom = max(1e-12, (1.0 - lam))
        for b in range(B):
            boot_sample = rng.choice(p, size=m, replace=True)
            Wb = np.sum(boot_sample > lam)
            pi0_b = (1.0 + Wb) / (m * denom)
            pi0_b = min(max(pi0_b, 0.0), 1.0)
            boot_pfdr[b] = _pFDR_hat_from_pi0_gamma(boot_sample, pi0_b, gamma)
        MSEs[i] = np.mean((boot_pfdr - plug_in_target) ** 2)

    # 3) pick best lambda minimizing MSE
    best_idx = int(np.argmin(MSEs))
    best_lambda = float(lambda_grid[best_idx])

    # 4) estimate pi0 using best lambda on original data
    denom = max(1e-12, (1.0 - best_lambda))
    W_best = np.sum(p > best_lambda)
    pi0_hat = (1.0 + W_best) / (m * denom)
    pi0_hat = float(np.clip(pi0_hat, 0.0, 1.0))

    # 5) adjust alpha and apply BH
    if pi0_hat <= 0:
        adjusted_alpha = float(alpha)
    else:
        adjusted_alpha = float(alpha / pi0_hat)
    adjusted_alpha = min(adjusted_alpha, 1.0)

    return bh(p, adjusted_alpha)


def quantilebh(p_values, alpha, k0_min_frac=0.60, k0_max_frac=0.95, B=200, random_state=None):
    """
    Quantile-BH with bootstrap selection of k0 (same plug-in MSE idea).
    Returns: dict with keys similar to storey_bh: rejected, pi0, best_k0, adjusted_alpha, ...
    Parameters:
      p_values: array-like
      alpha: target FDR
      k0_min_frac, k0_max_frac: fractions of m to set the search grid for k0 (defaults 0.60..0.95)
      B: bootstrap samples
    Notes:
      uses the quantile formula for pi0:
        pi0 = (m - k0 + 1) / ( m * (1 - p_(k0)) )
      where p_(k0) is the k0-th order statistic (1-indexed).
    """
    rng = np.random.default_rng(random_state)
    p = np.asarray(p_values)
    m = len(p)
    sorted_p = np.sort(p)

    k0_min = max(1, int(np.floor(k0_min_frac * m)))
    k0_max = max(1, int(np.floor(k0_max_frac * m)))
    # ensure valid grid
    k0_grid = np.arange(k0_min, k0_max + 1, int(0.05 * m))
    k0_grid = k0_grid[(k0_grid >= 1) & (k0_grid < m)]
    if len(k0_grid) == 0:
        k0_grid = np.array([max(1, m // 2)])

    # original-data pi0 estimates for each k0
    pi0_orig = []
    for k0 in k0_grid:
        p_k0 = sorted_p[k0 - 1]
        if p_k0 >= 1.0 - 1e-12:
            pi0_val = 1.0
        else:
            pi0_val = (m - k0 + 1) / (m * (1.0 - p_k0))
        pi0_val = float(np.clip(pi0_val, 0.0, 1.0))
        pi0_orig.append(pi0_val)
    pi0_orig = np.asarray(pi0_orig)
    plug_in_target = float(np.min(pi0_orig))  # use min pi0 over k0 as plug-in target

    # bootstrap: for each k0 compute bootstrap pi0* and MSE relative to plug-in target
    MSEs = np.empty(len(k0_grid), dtype=float)
    for i, k0 in enumerate(k0_grid):
        boot_pi0 = np.empty(B, dtype=float)
        for b in range(B):
            boot_sample = rng.choice(p, size=m, replace=True)
            boot_sorted = np.sort(boot_sample)
            p_k0_b = boot_sorted[k0 - 1]
            if p_k0_b >= 1.0 - 1e-12:
                pi0_b = 1.0
            else:
                pi0_b = (m - k0 + 1) / (m * (1.0 - p_k0_b))
            pi0_b = float(np.clip(pi0_b, 0.0, 1.0))
            boot_pi0[b] = pi0_b
        MSEs[i] = np.mean((boot_pi0 - plug_in_target) ** 2)

    best_idx = int(np.argmin(MSEs))
    best_k0 = int(k0_grid[best_idx])

    # estimate pi0 on original data with best_k0
    p_k0_orig = sorted_p[best_k0 - 1]
    if p_k0_orig >= 1.0 - 1e-12:
        pi0_hat = 1.0
    else:
        pi0_hat = (m - best_k0 + 1) / (m * (1.0 - p_k0_orig))
    pi0_hat = float(np.clip(pi0_hat, 0.0, 1.0))

    # adjust alpha and apply BH
    if pi0_hat <= 0:
        adjusted_alpha = float(alpha)
    else:
        adjusted_alpha = float(alpha / pi0_hat)
    adjusted_alpha = min(adjusted_alpha, 1.0)

    return bh(p, adjusted_alpha)



def integrative_bh(y, y_hat, confidence, alpha, args, calib_ratio=0.5, random=True):
    n_samples = len(y)
    n_calib = int(n_samples * calib_ratio)
    n_test = n_samples - n_calib

    # Randomly select calibration indices (without replacement)
    cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)

    # The remaining indices are for prediction
    test_indices = np.setdiff1d(np.arange(n_samples), cal_indices)

    # Split the dataset
    y_calib, y_hat_calib, conf_calib = y[cal_indices], y_hat[cal_indices], confidence[cal_indices]
    y_test, y_hat_test, conf_test = y[test_indices], y_hat[test_indices], confidence[test_indices]

    # H0: y_hat != y
    cal0_idx, cal1_idx = (y_hat_calib != y_calib), (y_hat_calib == y_calib)
    y_calib_0, y_hat_calib_0, conf_calib_0 = y_calib[cal0_idx], y_hat_calib[cal0_idx], conf_calib[cal0_idx]
    y_calib_1, y_hat_calib_1, conf_calib_1 = y_calib[cal1_idx], y_hat_calib[cal1_idx], conf_calib[cal1_idx]
    n_calib_0, n_calib_1 = y_calib_0.shape[0], y_calib_1.shape[0]

    cal0_score = 1 - conf_calib_0
    test_score = 1 - conf_test

    u_0 = np.sum(cal0_score[:, None] >= cal0_score, axis=-1) / (1 + n_calib_0)
    u_1 = (np.sum(conf_calib_0[:, None] >= conf_calib_1, axis=-1) + 1) / (1 + n_calib_1)

    u = np.zeros_like(y_test)
    for i in range(n_test):
        u_0_i = u_0 + (cal0_score >= test_score[i]).astype(int) / (1 + n_calib_0)
        u_0_test = (np.sum(test_score[i] >= cal0_score) + 1) / (1 + n_calib_0)
        u_0_i = np.concatenate((u_0_i, np.array([u_0_test])), axis=-1)

        u_1_test = (np.sum(conf_test[i] >= conf_calib_1) + 1) / (1 + n_calib_1)
        u_1_i = np.concatenate((u_1, np.array([u_1_test])), axis=-1)
        r = u_0_i / u_1_i
        u[i] = np.sum(r[-1] >= r) / (1 + n_calib_0)

    return bh(u, alpha)

def get_p_values(Y, Yhat, confidence, cal_indices, alpha, args, calib_ratio=0.5, random=True):
        all_indices = np.arange(len(Y))
        test_indices = np.setdiff1d(all_indices, cal_indices)

        y_calib, y_hat_calib, conf_calib = Y[cal_indices], Yhat[cal_indices], confidence[cal_indices]
        y_test, y_hat_test, conf_test = Y[test_indices], Yhat[test_indices], confidence[test_indices]
        n_calib = y_calib.shape[0]
        n_test = y_test.shape[0]

        # H0: y_hat != y
        cal0_idx = (y_hat_calib != y_calib)
        y_calib_0, y_hat_calib_0, conf_calib_0 = y_calib[cal0_idx], y_hat_calib[cal0_idx], conf_calib[cal0_idx]
        n_calib_0 = y_calib_0.shape[0]

        cal0_score = 1 - conf_calib_0
        test_score = 1 - conf_test

        if random:
            p_values = np.sum((test_score[:, None] > cal0_score), axis=-1) + np.random.rand(n_test) * (
                        np.sum((test_score[:, None] == cal0_score), axis=-1) + 1)
            p_values /= (1 + n_calib_0)

        selection_indices, t = bh(p_values, alpha * (1 + n_calib) / (1 + n_calib_0))

        return p_values, y_test, y_hat_test, t