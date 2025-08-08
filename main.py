import pandas as pd
import numpy as np
from algorithm.alg import selection
import argparse
from plot_utils.plot import plot_results


parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="text" ,choices=["vision", "text", "all", 'stance', 'misinfo', 'bias', 'imagenet', 'imagenetv2'])
parser.add_argument("--calib_ratio", type=float, default=0.5, help="Calibration ratio")
parser.add_argument("--random", default="True", choices=["True", "False"])
parser.add_argument("--num_trials", type=int, default=100, help="Number of trials")
parser.add_argument("--alpha", default=0.05, type=float, help="FDR threshold q")

parser.add_argument("--temperature", type=float, default=1, help="Temperature")
args = parser.parse_args()

dataset = args.datasets
if dataset == "vision":
    ds_list = ["imagenet", "imagenetv2"]
elif dataset == "text":
    ds_list = ['stance', 'misinfo', 'bias']
elif dataset == "all":
    ds_list = ["imagenet", "imagenetv2", 'stance', 'misinfo', 'bias']
else:
    ds_list = [dataset]

fdr_array = np.zeros(shape=(len(ds_list), args.num_trials))
power_array = np.zeros(shape=(len(ds_list), args.num_trials))
selection_size_array = np.zeros(shape=(len(ds_list), args.num_trials))

for i, ds in enumerate(ds_list):
    data = pd.read_csv("./datasets/" + ds + '.csv')
    Y = data["Y"].to_numpy()
    Yhat = None
    if ds in ['stance', 'misinfo', 'bias', 'sentiment']:
        Yhat = data["Yhat (GPT4o)"].to_numpy()
    elif ds in ['imagenet', 'imagenetv2', 'alphafold']:
        Yhat = data["Yhat"].to_numpy()

    confidence = data["confidence"].to_numpy()
    n = len(Y)

    alpha = args.alpha
    num_trials = args.num_trials
    fdr_list = []
    power_list = []
    selection_size_list = []

    for j in range(num_trials):
        fdp, power, selection_size = selection(Y, Yhat, confidence, alpha, calib_ratio=0.5, random=(args.random == "True"))
        fdr_list.append(fdp)
        power_list.append(power)
        selection_size_list.append(selection_size)

    fdr_array[i] = np.array(fdr_list)
    power_array[i] = np.array(power_list)
    selection_size_array[i] = np.array(selection_size_list)

    print(f"Results of {ds} dataset. q = {args.alpha}")
    print(f"Mean FDR: {np.mean(fdr_list)}")
    print(f"Mean Power: {np.mean(power_list)}")
    print(f"Mean Selection Size: {np.mean(selection_size_list)}")
