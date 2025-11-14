import numpy as np
from algorithm.select_alg import selection
from algorithm.preprocess import get_ood_data
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="resnet34_imagenet_misclassificationscore")
parser.add_argument("--calib_ratio", type=float, default=0.1, help="Calibration ratio")
parser.add_argument("--random", default="True", choices=["True", "False"])
parser.add_argument("--num_trials", type=int, default=10, help="Number of trials")
parser.add_argument("--alpha", default=0.1, type=float, help="Target FDR level")
parser.add_argument("--algorithm", type=str, default="cbh")
args = parser.parse_args()

dataset = args.dataset

Y, Yhat, msp_confidence, entropy_confidence, alpha_confidence = get_ood_data(dataset)

alpha = args.alpha
num_trials = args.num_trials
msp_fdr_list = []
msp_power_list = []

react_fdr_list = []
react_power_list = []

energy_fdr_list = []
energy_power_list = []

for j in range(num_trials):
    n_samples = len(Y)
    n_calib = int(n_samples * args.calib_ratio)
    n_test = n_samples - n_calib
    cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)

    msp_fdp, msp_power, selection_size, _ = selection(Y, Yhat, msp_confidence, cal_indices, alpha,
                                                      calib_ratio=args.calib_ratio, random=(args.random == "True"),
                                                      args=args)
    msp_fdr_list.append(msp_fdp)
    msp_power_list.append(msp_power)

    react_fdp, react_power, selection_size, _ = selection(Y, Yhat, entropy_confidence, cal_indices, alpha, calib_ratio=args.calib_ratio, random=(args.random == "True"), args=args)
    react_fdr_list.append(react_fdp)
    react_power_list.append(react_power)

    energy_fdp, energy_power, selection_size, _ = selection(Y, Yhat, alpha_confidence, cal_indices, alpha,
                                                            calib_ratio=args.calib_ratio, random=(args.random == "True"),
                                                            args=args)
    energy_fdr_list.append(energy_fdp)
    energy_power_list.append(energy_power)




print(f"MSP Realized FDR: {np.mean(msp_fdr_list) * 100}%")
print(f"MSP Mean Power: {np.mean(msp_power_list) * 100}%")
print()
print(f"React Realized FDR: {np.mean(react_fdr_list) * 100}%")
print(f"React Mean Power: {np.mean(react_power_list) * 100}%")
print()
print(f"energy Realized FDR: {np.mean(energy_fdr_list) * 100}%")
print(f"energy Mean Power: {np.mean(energy_power_list) * 100}%")
print()

print(f"AI only Error: {100 - np.sum(Yhat==Y) / len(Y) * 100}%")
