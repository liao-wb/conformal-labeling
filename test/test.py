import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from algorithm.select_alg import bh
# %%
n_simulations = 10
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
    n_test = len(s_test)

    #fdp, power, t = 0, 0, 0
    threshold = None
    all_score = np.concatenate((s_cal, s_test), axis=0)
    sorted_all_score = np.sort(all_score)

    for i in range(sorted_all_score.shape[0]):
        fdp_t =  (n_test) / (1 + n_cal) * (1 + np.sum(s_cal >= sorted_all_score[i])) / (np.sum(s_test >= sorted_all_score[i]))
        if fdp_t <= alpha:
                threshold = sorted_all_score[i]
    selection_indices = s_test >= threshold
    y_reject, y_hat_reject = y_test[selection_indices], y_test_pred[selection_indices]
    # y_accept, y_hat_accept = y_test[selection_indices == 0], y_hat_test[selection_indices == 0]

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