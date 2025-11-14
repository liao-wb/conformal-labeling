
import pandas as pd
import numpy as np
from algorithm.select_alg import selection
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", default="cbh", choices=["bh", "sbh", "cbh", "qbh", "integrative"])
parser.add_argument("--_lambda", type=float, default=0.94)
parser.add_argument("--k_0", type=int, default=10000)

parser.add_argument("--datasets", type=str, default="Qwen3-32B_mmlu",)
parser.add_argument("--calib_ratio", type=float, default=0.1, help="Calibration ratio")
parser.add_argument("--random", default="True", choices=["True", "False"])
parser.add_argument("--num_trials", type=int, default=2, help="Number of trials")
parser.add_argument("--alpha", default=0.1, type=float, help="FDR threshold q")

parser.add_argument("--model", default="Qwen3-32B", type=str)
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

algorithm_list = ["msp", "energy", "Dalpha"]
algorithm_fdr_list = [[] for _ in range(len(algorithm_list))]
algorithm_power_list = [[] for _ in range(len(algorithm_list))]
j = 0
for i, algorithm in tqdm(enumerate(algorithm_list)):
    print(i)
    args.algorithm = "cbh"
    ood_data = pd.read_csv("./datasets/resnet34_imagenet_oodscore_latest.csv")
    ood_Y = ood_data["Y"].to_numpy()
    msp_confidence = ood_data["msp_confidence"].to_numpy()
    energy_confidence = ood_data["energy_confidence"].to_numpy()
    ood_Yhat = ood_data["Yhat"].to_numpy()

    mis_data = pd.read_csv("./datasets/resnet34_imagenet_misclassificationscore.csv")
    alpha_confidence = mis_data["alpha_confidence"].to_numpy()
    mis_Y = mis_data["Y"].to_numpy()
    mis_Yhat = mis_data["Yhat"].to_numpy()

    n = len(ood_Y)
    for j, alpha in enumerate(alpha_list):
        num_trials = args.num_trials
        fdr_list = []
        power_list = []
        budget_save_list = []

        for z in range(num_trials):
            n_samples = len(ood_Y)
            n_calib = int(n_samples * args.calib_ratio)
            n_test = n_samples - n_calib
            cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)

            if algorithm == "msp":
                fdp, power, selection_size, _ = selection(ood_Y, ood_Yhat, msp_confidence, cal_indices, alpha, args, calib_ratio=args.calib_ratio, random=True)
            elif algorithm == "energy":
                fdp, power, selection_size, _ = selection(ood_Y, ood_Yhat, energy_confidence, cal_indices, alpha, args,
                                                          calib_ratio=args.calib_ratio, random=True)
            else:
                fdp, power, selection_size, _ = selection(mis_Y, mis_Yhat, alpha_confidence, cal_indices, alpha, args,
                                                          calib_ratio=args.calib_ratio, random=True)
            fdr_list.append(fdp)
            power_list.append(power)

        algorithm_fdr_list[i].append(np.mean(fdr_list))
        algorithm_power_list[i].append(np.mean(power_list))


colors = [
    '#1f77b4',  # Blue
    '#d62728',
    '#4DB99D',  # Bluish green
    '#FF6200',
    "#6495ED"

]

markers = ['o', '^', 'D', 's', 'v', 'p', '*', 'X']

large_font_size = 42
small_font_size = 36
plt.figure(figsize=(10, 8))


for i in range(len(algorithm_list)):
    plt.plot(
        alpha_list, algorithm_power_list[i],
        marker=markers[i % len(markers)],
        label=algorithm_list[i],
        markersize=18, linestyle='-', linewidth=6, color=colors[i], alpha=0.65
    )
#ax2.set_title("Power comparison", fontsize=large_font_size, weight='bold')
plt.xlabel("Target FDR Level (α)", fontsize=large_font_size)
plt.ylabel("Power", fontsize=large_font_size)
plt.tick_params(axis='both', which='major', labelsize=small_font_size)

plt.legend(fontsize=small_font_size, framealpha=1, shadow=True)

plt.tight_layout()
plt.savefig("score_power_comparison.pdf")
plt.show()