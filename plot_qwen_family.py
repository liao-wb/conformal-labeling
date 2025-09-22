import pandas as pd
import numpy as np
from algorithm.select_alg import selection
from algorithm.preprocess import get_data
import argparse
from plot_utils.plot import plot_results, plot_results_with_budget_save

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", default="cbh", choices=["bh", "sbh", "cbh", "quantbh", "integrative"])
parser.add_argument("--_lambda", type=float, default=0.5)
parser.add_argument("--k_0", type=int, default=3)

parser.add_argument("--datasets", type=str, default="gpt-4-turbo",)
parser.add_argument("--calib_ratio", type=float, default=0.1, help="Calibration ratio")
parser.add_argument("--random", default="True", choices=["True", "False"])
parser.add_argument("--num_trials", type=int, default=10 , help="Number of trials")

parser.add_argument("--model", default=None, type=str)
parser.add_argument("--temperature", type=float, default=1, help="Temperature")
args = parser.parse_args()

alpha_list = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

dataset = args.datasets

if dataset == "vision":
    ds_list = ["imagenet", "imagenetv2"]
elif dataset == "text":
    ds_list = ['stance', 'misinfo', 'bias']
elif dataset == "all":
    ds_list = ["imagenet", "imagenetv2", 'stance', 'misinfo', 'bias']
else:
    if args.model is not None:
        ds_list = [args.model + "_" + args.datasets]
    else:
        ds_list = [dataset]

ds_list = ["Qwen3-8B_medmcqa","Qwen3-8B_medmcqa","Qwen3-8B_medmcqa","Qwen3-32B_medmcqa","Qwen3-32B_medmcqa","Qwen3-32B_medmcqa"]

fdr_array = np.zeros(shape=(len(ds_list), len(alpha_list)))
power_array = np.zeros(shape=(len(ds_list), len(alpha_list)))
std_fdr_array = np.zeros(shape=(len(ds_list), len(alpha_list)))
std_power_array = np.zeros(shape=(len(ds_list), len(alpha_list)))

budget_save_array = np.zeros(shape=(len(ds_list), len(alpha_list)))
std_budget_save_array = np.zeros(shape=(len(ds_list), len(alpha_list)))

error_array = np.zeros(shape=(len(ds_list), len(alpha_list)))
std_error_array = np.zeros(shape=(len(ds_list), len(alpha_list)))

cal_ratio_list = [0.05, 0.1, 0.2, 0.05, 0.1, 0.2]


for i, ds in enumerate(ds_list):
    Y, Yhat, confidence = get_data(ds)
    args.calib_ratio = cal_ratio_list[i]

    n = len(Y)
    for j, alpha in enumerate(alpha_list):
        num_trials = args.num_trials
        fdr_list = []
        power_list = []
        budget_save_list = []
        error_list = []
        for z in range(num_trials):
            n_samples = len(Y)
            n_calib = int(n_samples * args.calib_ratio)
            n_test = n_samples - n_calib
            cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)
            fdp, power, selection_size, _ = selection(Y, Yhat, confidence, cal_indices, alpha, calib_ratio=args.calib_ratio, random=(args.random == "True"), args=args)
            fdr_list.append(fdp)
            power_list.append(power)
            budget_save_list.append(selection_size / len(Y))
            error_list.append(fdp * selection_size / len(Y))

        fdr_array[i,  j] = np.mean(fdr_list)
        power_array[i, j] = np.mean(power_list)
        budget_save_array[i, j] = np.mean(budget_save_list)
        error_array[i, j] = np.mean(error_list)
        std_fdr_array[i, j] = np.std(fdr_list)
        std_power_array[i, j] = np.std(power_list)
        std_budget_save_array[i, j] = np.std(budget_save_list)
        std_error_array[i, j] = np.std(error_list)

            #print(f"Results of {ds} dataset. q = {args.alpha}")
            #print(f"Mean FDR: {np.mean(fdr_list)}")
            #print(f"Mean Power: {np.mean(power_list)}")
            #print(f"Mean Selection Size: {np.mean(selection_size_list)}")


ds_list = ["Qwen3-8B: cal_ratio=0.05", "Qwen3-8B: cal_ratio=0.1", "Qwen3-8B: cal_ratio=0.2", "Qwen3-32B: cal_ratio=0.05", "Qwen3-32B: cal_ratio=0.1", "Qwen3-32B: cal_ratio=0.2"]
plot_results(models=ds_list, target_fdr_list=np.array([alpha_list for _ in range(len(ds_list))]), fdr_list=fdr_array, power_list=power_array,fdr_std_list=std_fdr_array, power_std_list=std_power_array,)
#plot_results_with_budget_save(models=ds_list, target_fdr_list=np.array([alpha_list for _ in range(len(ds_list))]), fdr_list=fdr_array, power_list=power_array,fdr_std_list=std_fdr_array, power_std_list=std_power_array, budget_save_list=budget_save_array, budget_save_std_list=std_budget_save_array)
