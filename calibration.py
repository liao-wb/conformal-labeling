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

parser.add_argument("--datasets", type=str, default="calibration_imagenet",)
parser.add_argument("--calib_ratio", type=float, default=0.2, help="Calibration ratio")
parser.add_argument("--random", default="True", choices=["True", "False"])
parser.add_argument("--num_trials", type=int, default=100, help="Number of trials")
parser.add_argument("--alpha", default=0.1, type=float, help="FDR threshold q")

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
    ds_list = [dataset]


j = 0


ds = args.datasets
data = pd.read_csv("./datasets/" + ds + '.csv', sep=",")
Y = data["Y"].to_numpy()
Yhat = data["Y_hat_resnet152"].to_numpy()

confidence = data["confidence"].to_numpy()

calibrated_confidence = data["calibrated_confidence"].to_numpy()



algorithm_fdr_list = [[] for _ in range(2)]
algorithm_power_list = [[] for _ in range(2)]

n = len(Y)
for j, alpha in enumerate(alpha_list):
    num_trials = args.num_trials
    fdr_list = []
    power_list = []

    calibrated_fdr_list = []
    calibrated_power_list = []

    for z in range(num_trials):
        n_samples = len(Y)
        n_calib = int(n_samples * args.calib_ratio)
        n_test = n_samples - n_calib
        cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)
        fdp, power, selection_size, _ = selection(Y, Yhat, confidence, cal_indices, alpha, args, calib_ratio=args.calib_ratio, random=True)
        fdr_list.append(fdp)
        power_list.append(power)

        calib_fdp, calib_power, calib_selection_size, _ = selection(Y, Yhat, calibrated_confidence, cal_indices, alpha, args,
                                                  calib_ratio=args.calib_ratio, random=True)
        calibrated_fdr_list.append(calib_fdp)
        calibrated_power_list.append(calib_power)

    algorithm_fdr_list[0].append(np.mean(fdr_list))
    algorithm_power_list[0].append(np.mean(power_list))

    algorithm_fdr_list[1].append(np.mean(calibrated_fdr_list))
    algorithm_power_list[1].append(np.mean(calibrated_power_list))



colors = [
    '#4682B4',  # Blue
    '#E69F00',
    '#4DB99D',  # Bluish green
    '#FF6200',
    "#6495ED"
]


large_font_size = 20
small_font_size = 16
plt.rcParams['axes.edgecolor'] = '#CCCCCC'  # Lighter axis spine color
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# --- Subplot 1: FDR Comparison ---
ax1.plot(
    alpha_list, alpha_list,
    label="Target FDR",
    linestyle="--", color=colors[-1], linewidth=1.5, alpha=0.7
)

markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'X']
label_list = ["Before calibrated", "After calibrated"]

for i in range(2):
    ax1.plot(
        alpha_list, algorithm_fdr_list[i],
        marker=markers[i % len(markers)],
        label=label_list[i],
        markersize=8, linestyle='-', linewidth=3, color=colors[i], alpha=1.0
    )

#ax1.set_title("FDR Comparison", fontsize=large_font_size, weight='bold')
ax1.set_xlabel("Target FDR Level (α)", fontsize=large_font_size)
ax1.set_ylabel("FDR", fontsize=large_font_size)

ax1.legend(loc='upper left', fontsize=small_font_size, framealpha=1, shadow=True)

for i in range(2):
    ax2.plot(
        alpha_list, algorithm_power_list[i],
        marker=markers[i % len(markers)],
        label=label_list[i],
        markersize=8, linestyle='-', linewidth=3, color=colors[i], alpha=1
    )
#ax2.set_title("Power comparison", fontsize=large_font_size, weight='bold')
ax2.set_xlabel("Target FDR Level (α)", fontsize=large_font_size)
ax2.set_ylabel("Power", fontsize=large_font_size)
plt.legend(fontsize=small_font_size, framealpha=1, shadow=True)

plt.tight_layout()
plt.savefig("calibration_comparison.pdf")
plt.show()