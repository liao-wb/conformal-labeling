import pandas as pd
import numpy as np
from algorithm.alg import selection
import argparse
from plot_utils.plot import plot_results, plot_results_with_budget_save

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="text",
                    choices=["vision", "text", "all", 'stance', 'misinfo', 'bias', 'imagenet', 'imagenetv2', "mathqa"])
parser.add_argument("--calib_ratio", type=float, default=0.1, help="Calibration ratio")
parser.add_argument("--random", default="True", choices=["True", "False"])
parser.add_argument("--num_trials", type=int, default=100, help="Number of trials")
parser.add_argument("--alpha", default=0.1, type=float, help="FDR threshold q")

parser.add_argument("--model", default=None, type=str)
parser.add_argument("--temperature", type=float, default=1, help="Temperature")
args = parser.parse_args()

alpha_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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

fdr_array = np.zeros(shape=(len(ds_list), len(alpha_list)))
power_array = np.zeros(shape=(len(ds_list), len(alpha_list)))
std_fdr_array = np.zeros(shape=(len(ds_list), len(alpha_list)))
std_power_array = np.zeros(shape=(len(ds_list), len(alpha_list)))

budget_save_array = np.zeros(shape=(len(ds_list), len(alpha_list)))
std_budget_save_array = np.zeros(shape=(len(ds_list), len(alpha_list)))

for i, ds in enumerate(ds_list):
    print(ds)
    data = pd.read_csv("./datasets/" + ds + '.csv', sep=",")
    print("haha")
    print(data.columns)
    Y = data["Y"].to_numpy()
    Yhat = None
    if ds in ['stance', 'misinfo', 'bias', 'sentiment']:
        Yhat = data["Yhat (GPT4o)"].to_numpy()
    else:
        Yhat = data["Yhat"].to_numpy()

    confidence = data["confidence"].to_numpy()
    n = len(Y)

    for j, alpha in enumerate(alpha_list):
        num_trials = args.num_trials
        fdr_list = []
        power_list = []
        budget_save_list = []

        for z in range(num_trials):
            fdp, power, selection_size = selection(Y, Yhat, confidence, alpha, calib_ratio=args.calib_ratio,
                                                   random=(args.random == "True"))
            fdr_list.append(fdp)
            power_list.append(power)
            budget_save_list.append(selection_size / (len(Y) * (1 - args.calib_ratio)))

        fdr_array[i,  j] = np.mean(fdr_list)
        power_array[i, j] = np.mean(power_list)
        budget_save_array[i, j] = np.mean(budget_save_list)
        std_fdr_array[i, j] = np.std(fdr_list)
        std_power_array[i, j] = np.std(power_list)
        std_budget_save_array[i, j] = np.std(budget_save_list)

        #print(f"Results of {ds} dataset. q = {args.alpha}")
        #print(f"Mean FDR: {np.mean(fdr_list)}")
        #print(f"Mean Power: {np.mean(power_list)}")
        #print(f"Mean Selection Size: {np.mean(selection_size_list)}")

plot_results(models=ds_list, target_fdr_list=np.array([alpha_list for _ in range(len(ds_list))]), fdr_list=fdr_array, power_list=power_array,fdr_std_list=std_fdr_array, power_std_list=std_power_array,)
#plot_results_with_budget_save(models=ds_list, target_fdr_list=np.array([alpha_list for _ in range(len(ds_list))]), fdr_list=fdr_array, power_list=power_array,fdr_std_list=std_fdr_array, power_std_list=std_power_array, budget_save_list=budget_save_array, budget_save_std_list=std_budget_save_array)
