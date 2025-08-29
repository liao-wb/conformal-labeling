import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from algorithm.select_alg import bh
# %%
print("First")
n_simulations = 100
alpha = 0.1
fdps = []
powers = []
for iter in range(n_simulations):
    X, y = make_classification(n_samples=10000, n_features=10, n_informative=8, n_redundant=2, n_classes=3,
                               random_state=None)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, stratify=y)
    X_cal, X_train, y_cal, y_train = train_test_split(X_temp, y_temp, test_size=1 / 3, shuffle=True, stratify=y_temp)

    n_cal = len(X_cal)
    n_train = len(X_train)
    mask = np.min(cdist(X_test, np.vstack((X_cal, X_train))), axis=1) > 0.5
    X_test = X_test[mask]
    y_test = y_test[mask]
    m = len(X_test)

    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=None)
    model.fit(X_train, y_train)

    # 校准样本的置信度和异常性
    y_cal_pred = model.predict(X_cal)
    y_cal_prob = model.predict_proba(X_cal)
    s_cal = np.array([y_cal_prob[i, y_cal_pred[i]] for i in range(n_cal)])
    true_index = np.where(y_cal_pred != y_cal)
    s_cal = s_cal[true_index]
    n_cal = len(s_cal)

    # 测试样本的置信度和异常性
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)
    s_test = np.array([y_test_prob[i, y_test_pred[i]] for i in range(m)])
    n_test = m

    #print(f"Calibration and test dataset:{n_cal} and {n_test}")
    fdp, power, t = 0, 0, 0
    for i in range(1000):
        S = [j for j in range(m) if s_test[j] > t]
        y_test_selected = y_test[S]
        y_test_pred_selected = y_test_pred[S]
        false_positives_estimated = np.sum(s_cal > t) + 1
        false_positives = np.sum(y_test_selected != y_test_pred_selected)
        fdp = false_positives / len(S) if len(S) > 0 else 0
        fdp_estimated = false_positives_estimated / len(S) if len(S) > 0 else false_positives_estimated / 1
        t += 0.001
        if fdp_estimated <= alpha:
            break
    fdps.append(fdp)
    power = (len(S) - false_positives) / np.sum(y_test == y_test_pred)
    powers.append(power)
    #print(f"Estimated FDR: {fdp:.4f}")
    #print(f"Estimated Power: {power:.4f}")
print(f"Average FDR: {np.mean(fdps)}")
print(f"Average Power: {np.mean(powers):.4f}")

print("Second")

alpha = 0.1
fdps = []
powers = []
for iter in range(n_simulations):
    X, y = make_classification(n_samples=10000, n_features=10, n_informative=8, n_redundant=2, n_classes=3,
                               random_state=None)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, stratify=y)
    X_cal, X_train, y_cal, y_train = train_test_split(X_temp, y_temp, test_size=1 / 3, shuffle=True, stratify=y_temp)

    origin_n_cal = len(X_cal)
    n_train = len(X_train)
    mask = np.min(cdist(X_test, np.vstack((X_cal, X_train))), axis=1) > 0.5
    X_test = X_test[mask]
    y_test = y_test[mask]
    m = len(X_test)

    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=None)
    model.fit(X_train, y_train)

    # 校准样本的置信度和异常性
    y_cal_pred = model.predict(X_cal)
    y_cal_prob = model.predict_proba(X_cal)
    s_cal = 1 - np.array([y_cal_prob[i, y_cal_pred[i]] for i in range(origin_n_cal)])
    true_index = np.where(y_cal_pred != y_cal)
    s_cal = s_cal[true_index]
    n_cal = len(s_cal)

    # 测试样本的置信度和异常性
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)
    s_test = 1 - np.array([y_test_prob[i, y_test_pred[i]] for i in range(m)])
    n_test = m

    #print(f"Calibration and test dataset:{n_cal} and {n_test}")
    fdp, power = 0, 0
    t_list = np.concatenate((s_cal, s_test), axis=0)
    t_list = sorted(t_list, reverse=True)
    #t_list = sorted(s_test, reverse=True)
    threshold = 0
    for i in range(len(t_list)):
        fdp_t = n_test / (1 + origin_n_cal) * (1 + np.sum(s_cal <= t_list[i])) / np.sum(s_test <= t_list[i])
        if fdp_t <= alpha:
            threshold = t_list[i]
            break



    selection_indices = (s_test <= threshold)
    y_reject, y_hat_reject = y_test[selection_indices], y_test_pred[selection_indices]

    selection_size = y_reject.shape[0]

    fdp = np.sum(y_reject != y_hat_reject) / selection_size if selection_size > 0 else 0
    power = np.sum(y_reject == y_hat_reject) / np.sum(y_test == y_test_pred)

    fdps.append(fdp)
    powers.append(power)
    #print(f"Estimated FDR: {fdp:.4f}")
    #print(f"Estimated Power: {power:.4f}")
print(f"Average FDR: {np.mean(fdps)}")
print(f"Average Power: {np.mean(powers):.4f}")


print("Third")

alpha = 0.1
fdps = []
powers = []
for iter in range(n_simulations):
    X, y = make_classification(n_samples=10000, n_features=10, n_informative=8, n_redundant=2, n_classes=3,
                               random_state=None)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, stratify=y)
    X_cal, X_train, y_cal, y_train = train_test_split(X_temp, y_temp, test_size=1 / 3, shuffle=True, stratify=y_temp)
    n_cal = len(X_cal)
    n_train = len(X_train)
    mask = np.min(cdist(X_test, np.vstack((X_cal, X_train))), axis=1) > 0.5
    X_test = X_test[mask]
    y_test = y_test[mask]
    m = len(X_test)

    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=None)
    model.fit(X_train, y_train)

    # 校准样本的置信度和异常性
    y_cal_pred = model.predict(X_cal)
    y_cal_prob = model.predict_proba(X_cal)
    s_cal = 1 - np.array([y_cal_prob[i, y_cal_pred[i]] for i in range(n_cal)])
    true_index = np.where(y_cal_pred != y_cal)
    s_cal = s_cal[true_index]
    n_cal = len(s_cal)

    # 测试样本的置信度和异常性
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)
    s_test = 1 - np.array([y_test_prob[i, y_test_pred[i]] for i in range(m)])

    p_values = (np.sum(s_test[:, None] >= s_cal, axis=-1) + 1) / (1 + n_cal)
    selection_indices = bh(p_values, alpha)

    y_reject, y_hat_reject = y_test[selection_indices], y_test_pred[selection_indices]

    selection_size = y_reject.shape[0]

    fdp = np.sum(y_reject != y_hat_reject) / selection_size if selection_size > 0 else 0
    power = np.sum(y_reject == y_hat_reject) / np.sum(y_test == y_test_pred)
    fdps.append(fdp)
    #power = (len(S) - false_positives) / np.sum(y_test == y_test_pred)
    powers.append(power)

    #print(f"Estimated FDR: {fdp:.4f}")
    #print(f"Estimated Power: {power:.4f}")
print(f"Average FDR: {np.mean(fdps)}")
print(f"Average Power: {np.mean(powers):.4f}")
