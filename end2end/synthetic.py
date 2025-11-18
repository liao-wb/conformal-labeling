import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from algorithm.select_alg import selection
from sklearn.metrics import accuracy_score, classification_report

# Generate a synthetic classification dataset
X, y = make_classification(
    n_samples=15000,           # Total number of samples
    n_features=15,             # Number of features
    n_informative=12,  # 增加有信息特征的比重
    n_redundant=3,
    n_classes=10,               # Number of classes
)

# Split the data into train, calibration, and test sets
# First split: 60% train, 40% temporary (for calibration + test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, stratify=y
)

X_train1, X_train2, y_train1, y_train2 = train_test_split(
    X_train, y_train, test_size=1 / 2, stratify=y_train
)


print(f"Training set 1 shape: {X_train1.shape}")
print(f"Test set shape: {X_test.shape}")


# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,        # Number of trees in the forest
)

# Train the model on the training set
rf_classifier.fit(X_train1, y_train1)

y_test_pred = rf_classifier.predict(X_test)
origin_acc = np.sum(y_test_pred == y_test) / len(y_test_pred)
print(f"Training on training dataset1 Accuracy: {origin_acc}")





import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="imagenetv2")
parser.add_argument("--calib_ratio", type=float, default=0.1, help="Calibration ratio")
parser.add_argument("--random", default="True", choices=["True", "False"])
parser.add_argument("--num_trials", type=int, default=1, help="Number of trials")
parser.add_argument("--alpha", default=0.1, type=float, help="FDR threshold q")
parser.add_argument("--algorithm", default="cbh", choices=["bh", "sbh", "cbh", "quantbh", "integrative"])
parser.add_argument("--temperature", type=float, default=1, help="Temperature")
args = parser.parse_args()

Y = y_train2
Yhat = rf_classifier.predict(X_train2)
confidence = rf_classifier.predict_proba(X_train2)[np.arange(len(y_train2)), Yhat]

n_samples = len(Y)
n_calib = int(0.1 * len(Y))
n_test = n_samples - n_calib
# Create boolean mask
cal_mask = np.zeros(len(Y), dtype=bool)
cal_mask[:n_calib] = True

fdr, power, selection_size, selection_indices = selection(Y, Yhat, confidence, cal_mask, alpha=0.1, args=args, calib_ratio=0.1, random=True)
print(f"FDR: {fdr} Selection size: {selection_size}")
cal_x, cal_y = X_train2[cal_mask], y_train2[cal_mask]
all_indices = np.arange(len(Y))
test_indices = np.setdiff1d(all_indices, cal_mask)
unlabeled_x, unlabeled_y = X_train2[test_indices], y_train2[test_indices]

ai_labeled_x = unlabeled_x[selection_indices]
ai_labeled_y = rf_classifier.predict(ai_labeled_x)

non_selected_x, non_selected_y = unlabeled_x[selection_indices == 0], unlabeled_y[selection_indices == 0]


x_train_3, y_train_3 = np.concatenate([cal_x, ai_labeled_x], axis=0), np.concatenate([cal_y, ai_labeled_y], axis=0)

new_classifier = RandomForestClassifier(
    n_estimators=200,        # Number of trees in the forest
)

# Train the model on the training set
new_classifier.fit(x_train_3, y_train_3)
y_test_pred = new_classifier.predict(X_test)
origin_acc = np.sum(y_test_pred == y_test) / len(y_test_pred)
print(f"Training on training dataset3 Accuracy: {origin_acc}")

x_train_3, y_train_3 = np.concatenate([cal_x, ai_labeled_x, non_selected_x], axis=0), np.concatenate([cal_y, ai_labeled_y, non_selected_y], axis=0)

new_classifier = RandomForestClassifier(
    n_estimators=200,        # Number of trees in the forest
)

# Train the model on the training set
new_classifier.fit(x_train_3, y_train_3)
y_test_pred = new_classifier.predict(X_test)
origin_acc = np.sum(y_test_pred == y_test) / len(y_test_pred)
print(f"Training on training dataset3 Accuracy: {origin_acc}")

