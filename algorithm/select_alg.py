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
        selection_indices, _ = quantbh(p_values, alpha)
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
        selection_indices, _ = quantbh(p_values, alpha, k_0=args.k_0)
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
        selection_indices = quantbh(p_values, alpha)
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
    power = np.sum((y_reject - y_hat_reject)**2 <= error) / np.sum((y_test - y_hat_test)**2 <= error) if np.sum((y_test - y_hat_test)**2 <= error) > 0 else 0
    mean_l2 = np.mean((y_reject - y_hat_reject)**2) if selection_size > 0 else 0

    if np.isnan(mean_l2):
        mean_l2 = 0

    return fdr, power, selection_size, mean_l2



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


def storeybh(p_values, alpha, B=20):
    """Adaptive Storey-BH method with auto-selection of lambda from 0.1 to 0.9.
    Returns boolean array of rejections based on adjusted alpha via pi_0.
    """
    m = len(p_values)
    p_values = np.asarray(p_values)

    # Define lambda grid (0.1, 0.2, ..., 0.9)
    lambda_grid = np.arange(0.1, 1.0, 0.1)

    # Proxy gamma for pFDR estimation (e.g., median p-value as in Section 9)
    gamma_proxy = np.clip(np.median(p_values), 1e-8, 1 - 1e-8)

    # --- Step 1: compute original-data pFDR estimates for all lambdas ---
    pfdr_orig = []
    for lam in lambda_grid:
        denom = max(1e-12, 1 - lam)
        pi0_hat = (1 + np.sum(p_values > lam)) / (m * denom)
        pi0_hat = np.clip(pi0_hat, 0.0, 1.0)

        R = np.sum(p_values <= gamma_proxy)
        R = max(R, 1)
        Pr_P_leq_gamma = R / m
        cond_prob = 1 - (1 - gamma_proxy) ** m
        pfdr_hat = (pi0_hat * gamma_proxy) / (Pr_P_leq_gamma * cond_prob) if cond_prob > 0 else 1.0
        pfdr_orig.append(min(pfdr_hat, 1.0))

    plug_in_target = np.min(pfdr_orig)

    # --- Step 2: bootstrap MSE for each lambda ---
    best_lambda = lambda_grid[0]
    min_mse = float("inf")

    for lam in lambda_grid:
        denom = max(1e-12, 1 - lam)
        boot_pfdr = []
        for _ in range(B):
            boot_p = np.random.choice(p_values, size=m, replace=True)
            pi0_b = (1 + np.sum(boot_p > lam)) / (m * denom)
            pi0_b = np.clip(pi0_b, 0.0, 1.0)

            Rb = np.sum(boot_p <= gamma_proxy)
            Rb = max(Rb, 1)
            Pr_P_leq_gamma_b = Rb / m
            cond_prob_b = 1 - (1 - gamma_proxy) ** m
            pfdr_b = (pi0_b * gamma_proxy) / (Pr_P_leq_gamma_b * cond_prob_b) if cond_prob_b > 0 else 1.0
            boot_pfdr.append(min(pfdr_b, 1.0))

        mse = np.mean((np.array(boot_pfdr) - plug_in_target) ** 2)
        if mse < min_mse:
            min_mse = mse
            best_lambda = lam

    # --- Step 3: estimate pi0 using best lambda ---
    denom = max(1e-12, 1 - best_lambda)
    pi0 = (1 + np.sum(p_values > best_lambda)) / (m * denom)
    pi0 = np.clip(pi0, 0.0, 1.0)

    # --- Step 4: adjust alpha and apply BH ---
    adjusted_alpha = alpha / pi0 if pi0 > 0 else alpha
    adjusted_alpha = min(adjusted_alpha, 1.0)

    return bh(p_values, adjusted_alpha)


def quantbh(p_values, alpha, B=20):
    """
    Quantile BH with bootstrap-based automatic selection of k_0.
    Returns boolean array of rejections (same order as p_values).
    """
    m = len(p_values)
    p_values = np.asarray(p_values)
    sorted_p = np.sort(p_values)

    # --- Step 1: define candidate k0 grid ---
    k0_min = int(0.60 * m)
    k0_max = int(0.9 * m)
    k0_grid = np.arange(k0_min, k0_max + 1, int(0.1 * m))
    k0_grid = k0_grid[(k0_grid >= 1) & (k0_grid < m)]
    if len(k0_grid) == 0:
        k0_grid = np.array([m // 2])

    # --- Step 2: compute pi0 on original data for all k0 ---
    pi0_orig = []
    for k0 in k0_grid:
        p_k0 = sorted_p[k0 - 1]
        if p_k0 >= 1 - 1e-12:
            pi0_val = 1.0
        else:
            pi0_val = (m - k0 + 1) / (m * (1 - p_k0))
        pi0_orig.append(np.clip(pi0_val, 0.0, 1.0))
    pi0_orig = np.array(pi0_orig)

    # plug-in target = min pi0 over original data
    plug_in_target = np.min(pi0_orig)

    # --- Step 3: bootstrap MSE for each k0 ---
    best_k0 = k0_grid[0]
    min_mse = float("inf")

    for k0 in k0_grid:
        boot_pi0 = []
        for _ in range(B):
            boot_sample = np.random.choice(p_values, size=m, replace=True)
            boot_sorted = np.sort(boot_sample)
            p_k0_b = boot_sorted[k0 - 1]
            if p_k0_b >= 1 - 1e-12:
                pi0_b = 1.0
            else:
                pi0_b = (m - k0 + 1) / (m * (1 - p_k0_b))
            boot_pi0.append(np.clip(pi0_b, 0.0, 1.0))

        mse = np.mean((np.array(boot_pi0) - plug_in_target) ** 2)
        if mse < min_mse:
            min_mse = mse
            best_k0 = k0

    # --- Step 4: final pi0 estimate using best_k0 ---
    p_k0 = sorted_p[best_k0 - 1]
    if p_k0 >= 1 - 1e-12:
        pi0_est = 1.0
    else:
        pi0_est = (m - best_k0 + 1) / (m * (1 - p_k0))
    pi0_est = np.clip(pi0_est, 0.0, 1.0)

    # --- Step 5: adjust alpha and apply BH ---
    adjusted_alpha = alpha / pi0_est if pi0_est > 0 else alpha
    adjusted_alpha = min(adjusted_alpha, 1.0)

    return bh(p_values, adjusted_alpha)


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