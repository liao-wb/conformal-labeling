import numpy as np
from scipy.stats import bernoulli, norm


def selection(y, y_hat, confidence, alpha, calib_ratio=0.5, random=True):
    """"""
    n_samples = len(y)
    n_calib = int(n_samples * calib_ratio)

    # Randomly select calibration indices (without replacement)
    cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)

    # The remaining indices are for prediction
    test_indices = np.setdiff1d(np.arange(n_samples), cal_indices)

    # Split the data
    y_calib, y_hat_calib, conf_calib = y[cal_indices], y_hat[cal_indices], confidence[cal_indices]
    y_test, y_hat_test, conf_test = y[test_indices], y_hat[test_indices], confidence[test_indices]

    # H0: y_hat != y
    cal0_idx = (y_hat_calib != y_calib)
    y_calib_0, y_hat_calib_0, conf_calib_0 = y_calib[cal0_idx], y_hat_calib[cal0_idx], conf_calib[cal0_idx]

    cal0_score = 1 - conf_calib_0
    test_score = 1 - conf_test

    if random:
        p_values = np.sum((test_score[:, None] > cal0_score), axis=-1) + np.random.rand(n_samples - n_calib) * (np.sum((test_score[:, None] == cal0_score), axis=-1) + 1)
        p_values /= (1 + n_calib)
    else:
        p_values = np.sum((test_score[:, None] >= cal0_score), axis=-1) + 1
        p_values /= (1 + n_calib)

    selection_indices = bh_procedure(p_values, alpha)
    y_reject, y_hat_reject = y_test[selection_indices], y_hat_test[selection_indices]
    #y_accept, y_hat_accept = y_test[selection_indices == 0], y_hat_test[selection_indices == 0]

    selection_size = y_reject.shape[0]

    fdr = np.sum(y_reject != y_hat_reject) / selection_size if selection_size > 0 else 0
    power = np.sum(y_reject == y_hat_reject) / np.sum(y_test == y_hat_test)

    return fdr, power, selection_size


def bh_procedure(p_values, alpha):
    m = p_values.shape[0]
    p_values_sorted = np.sort(p_values)

    largest_i = 0
    threshold = [k * alpha / m for k in range(1, m + 1)]
    for i in range(m - 1, -1, -1):
        if p_values_sorted[i] <= threshold[i]:
            largest_i = i
            break
    t = threshold[largest_i]
    selection_indices = p_values <= t
    return selection_indices
