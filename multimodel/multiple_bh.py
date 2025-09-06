import pandas as pd
import numpy as np
from algorithm.select_alg import new_selection
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="gpt-4-turbo" ,choices=["vision", "text", "all", 'stance', 'misinfo', 'bias', 'imagenet', 'imagenetv2'])
parser.add_argument("--calib_ratio", type=float, default=0.2, help="Calibration ratio")
parser.add_argument("--random", default="True", choices=["True", "False"])
parser.add_argument("--num_trials", type=int, default=100, help="Number of trials")
parser.add_argument("--alpha", default=0.05, type=float, help="FDR threshold q")
parser.add_argument("--algorithm", default="cbh", choices=["bh", "sbh", "cbh", "quantbh", "integrative"])
parser.add_argument("--temperature", type=float, default=1, help="Temperature")
args = parser.parse_args()

alpha = 0.1

data = pd.read_csv("./datasets/" + "model_predictions" + '.csv').sample(frac=1).reset_index(drop=True)
Y = data["Y"].to_numpy()
confidence_1 = data["confidence_resnet34"].to_numpy()
confidence_2 = data["confidence_resnet50"].to_numpy()
confidence_3 = data["confidence_resnet152"].to_numpy()
Yhat_1 = data["Y_hat_resnet34"].to_numpy()
Yhat_2 = data["Y_hat_resnet50"].to_numpy()
Yhat_3 = data["Y_hat_resnet152"].to_numpy()

n_calib = int(0.2 * len(Y))
y_calib = Y[:n_calib]
y_hat_calib1 = Yhat_1[:n_calib]
conf_calib1 = confidence_1[:n_calib]
y_hat_calib2 = Yhat_2[:n_calib]
conf_calib2 = confidence_2[:n_calib]
y_hat_calib3 = Yhat_3[:n_calib]
conf_calib3 = confidence_3[:n_calib]

y_test = Y[n_calib:]
y_hat1_test = Yhat_1[n_calib:]
conf1_test = confidence_1[n_calib:]
y_hat2_test = Yhat_2[n_calib:]
conf2_test = confidence_2[n_calib:]
y_hat3_test = Yhat_3[n_calib:]
conf3_test = confidence_3[n_calib:]

fdr1, power1, selection_size1, selection_indices1 = new_selection(y_calib, y_hat_calib1, conf_calib1, y_test, y_hat1_test, conf1_test, alpha, random=True, args=args)

print("1")
print(fdr1, power1, selection_size1)
idx = (selection_indices1 == 0)
y_test = y_test[idx]
y_hat2_test = y_hat2_test[idx]
conf2_test = conf2_test[idx]
y_hat3_test = y_hat3_test[idx]
conf3_test = conf3_test[idx]
fdr2, power2, selection_size2, selection_indices2 = new_selection(y_calib, y_hat_calib2, conf_calib2, y_test, y_hat2_test, conf2_test, alpha, random=True, args=args)

print()
print("2")
print(fdr2, power2, selection_size2)

idx = (selection_indices2 == 0)
y_test = y_test[idx]
y_hat3_test = y_hat3_test[idx]
conf3_test = conf3_test[idx]
fdr3, power3, selection_size3, selection_indices3 = new_selection(y_calib, y_hat_calib3, conf_calib3, y_test, y_hat3_test, conf3_test, alpha, random=True, args=args)

print()
print("3")
print(fdr3, power3, selection_size3)

print("total selection size")
print(selection_size1 + selection_size2 + selection_size3)