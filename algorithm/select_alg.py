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
        selection_indices = bh(p_values, alpha)
    elif args.algorithm == "cbh":
        selection_indices = bh(p_values, alpha * (1 + n_calib) / (1 + n_calib_0))
    elif args.algorithm == "sbh":
        selection_indices = storeybh(p_values, alpha)
    elif args.algorithm == "qbh":
        selection_indices = quantbh(p_values, alpha, k_0=args.k_0)
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
        selection_indices = bh(p_values, alpha)
    elif args.algorithm == "cbh":
        selection_indices = bh(p_values, alpha * (1 + n_calib) / (1 + n_calib_0))
    elif args.algorithm == "sbh":
        selection_indices = storeybh(p_values, alpha)
    elif args.algorithm == "qbh":
        selection_indices = quantbh(p_values, alpha, k_0=args.k_0)
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
        selection_indices = bh(p_values, alpha)
    elif args.algorithm == "cbh":
        selection_indices = bh(p_values, alpha * (1 + n_calib) / (1 + n_calib_0))
    elif args.algorithm == "sbh":
        selection_indices = storeybh(p_values, alpha)
    elif args.algorithm == "qbh":
        selection_indices = quantbh(p_values, alpha, k_0=args.k_0)
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
    return selection_indices

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


def storeybh(p_values, alpha, B=100):
    """Adaptive Storey-BH method with auto-selection of lambda from 0.1 to 0.9.
    Returns boolean array of rejections based on adjusted alpha via pi_0."""
    m = len(p_values)

    # Define lambda grid (0.1, 0.2, ..., 0.9)
    lambda_grid = np.arange(0.1, 1.0, 0.1)

    # Proxy gamma for pFDR estimation (e.g., median p-value as in Section 9)
    gamma_proxy = np.median(p_values)

    # Bootstrap to select optimal lambda via MSE minimization (approximating Algorithm 3, Section 9)
    best_lambda = lambda_grid[0]
    min_mse = float('inf')
    for lambda_ in lambda_grid:
        boot_pFDR = []
        for _ in range(B):
            boot_p = np.random.choice(p_values, size=m, replace=True)
            pi0 = (1 + np.sum(boot_p > lambda_)) / (m * (1 - lambda_)) if (1 - lambda_) > 0 else 1.0
            R = np.sum(boot_p <= gamma_proxy)
            Pr_P_leq_gamma = max(R, 1) / m
            cond_prob = 1 - (1 - gamma_proxy) ** m
            pFDR = (pi0 * gamma_proxy) / (Pr_P_leq_gamma * cond_prob) if cond_prob > 0 else 1.0
            boot_pFDR.append(min(pFDR, 1.0))
        mse = np.mean((np.array(boot_pFDR) - np.min(boot_pFDR)) ** 2)
        if mse < min_mse:
            min_mse = mse
            best_lambda = lambda_

    # Estimate pi_0 with the optimal lambda
    pi_0 = (1 + np.sum(p_values > best_lambda)) / (m * (1 - best_lambda)) if (1 - best_lambda) > 0 else 1.0
    pi_0 = min(pi_0, 1.0)  # Clamp <=1

    # Adjust alpha based on pi_0 (Section 4 connection)
    adjusted_alpha = alpha / pi_0 if pi_0 > 0 else alpha

    # Apply BH procedure with adjusted alpha
    return bh(p_values, adjusted_alpha)

def quantbh(p_values, alpha, k_0):
    sorted_p_values = np.sort(p_values)
    m = len(p_values)
    pi_0 = (m - k_0 + 1) / (m * (1 - sorted_p_values[k_0 - 1]))
    alpha = alpha / pi_0
    return bh(p_values, alpha)

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

