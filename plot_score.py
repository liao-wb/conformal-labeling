import matplotlib.pyplot as plt
import numpy as np
from algorithm.select_alg import selection, get_p_values
from algorithm.preprocess import get_data, get_ood_data
import argparse
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="resnet34_imagenet_oodscore")
parser.add_argument("--calib_ratio", type=float, default=0.1, help="Calibration ratio")
parser.add_argument("--random", default="True", choices=["True", "False"])
parser.add_argument("--num_trials", type=int, default=1, help="Number of trials")
parser.add_argument("--alpha", default=0.1, type=float, help="FDR threshold q")
parser.add_argument("--algorithm", default="cbh", choices=["bh", "sbh", "cbh", "quantbh", "integrative"])
parser.add_argument("--temperature", type=float, default=1, help="Temperature")
parser.add_argument("--score_function", default="energy")
args = parser.parse_args()

dataset = args.datasets
if dataset == "vision":
    ds_list = ["imagenet", "imagenetv2"]
elif dataset == "text":
    ds_list = ['bias', 'stance', 'misinfo']
elif dataset == "all":
    ds_list = ["imagenet", "imagenetv2", 'stance', 'misinfo', 'bias']
else:
    ds_list = [dataset]

fdr_array = np.zeros(shape=(len(ds_list), args.num_trials))
power_array = np.zeros(shape=(len(ds_list), args.num_trials))
selection_size_array = np.zeros(shape=(len(ds_list), args.num_trials))

large_font_size = 30
small_font_size = 24

for i, ds in enumerate(ds_list):
    Y, Yhat, msp_confidence, react_confidence, energy_confidence = get_ood_data(ds)
    n = len(Y)

    alpha = args.alpha
    num_trials = args.num_trials

    n_samples = len(Y)
    n_calib = int(n_samples * args.calib_ratio)
    n_test = n_samples - n_calib
    cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)

    if args.score_function == "msp":
        p_values, y_test, y_test_hat, t = get_p_values(Y, Yhat, msp_confidence, cal_indices, alpha, args, calib_ratio=args.calib_ratio, random=True)
    elif args.score_function == "react":
        p_values, y_test, y_test_hat, t = get_p_values(Y, Yhat, react_confidence, cal_indices, alpha, args,
                                                       calib_ratio=args.calib_ratio, random=True)
    elif args.score_function == "energy":
        p_values, y_test, y_test_hat, t = get_p_values(Y, Yhat, energy_confidence, cal_indices, alpha, args,
                                                       calib_ratio=args.calib_ratio, random=True)
    else:
        raise NotImplementedError
    p_0 = p_values[y_test != y_test_hat]
    p_1 = p_values[y_test == y_test_hat]

    plt.figure(figsize=(8, 6))

    # 绘制填充的KDE
    sns.kdeplot(p_0, color='#1f77b4', fill=True, alpha=0.5, linewidth=0, label='p_values under H0')  # 蓝色
    sns.kdeplot(p_1, color='#d62728', fill=True, alpha=0.5, linewidth=0, label='p_values under H1')  # 红色

    # 绘制轮廓线
    sns.kdeplot(p_0, color='black', fill=False, linewidth=1, alpha=0.9)
    sns.kdeplot(p_1, color='black', fill=False, linewidth=1, alpha=0.9)

    plt.tick_params(axis='both', which='major', labelsize=small_font_size)
    #plt.xlim(-0.1, 1)
    plt.xlabel("p-values", fontsize=large_font_size)
    plt.ylabel("Density", fontsize=large_font_size)
    plt.tight_layout()
    plt.legend(fontsize=small_font_size, framealpha=1, shadow=True)
    plt.savefig(f"{args.score_function}.pdf", dpi=300)
    plt.show()
