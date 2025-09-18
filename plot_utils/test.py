import matplotlib.pyplot as plt
import numpy as np
from algorithm.select_alg import selection, get_p_values
from algorithm.preprocess import get_data
import argparse
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="imagenet")
parser.add_argument("--calib_ratio", type=float, default=0.1, help="Calibration ratio")
parser.add_argument("--random", default="True", choices=["True", "False"])
parser.add_argument("--num_trials", type=int, default=10, help="Number of trials")
parser.add_argument("--alpha", default=0.1, type=float, help="FDR threshold q")
parser.add_argument("--algorithm", default="cbh", choices=["bh", "sbh", "cbh", "quantbh", "integrative"])
parser.add_argument("--temperature", type=float, default=1, help="Temperature")
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

for i, ds in enumerate(ds_list):
    Y, Yhat, confidence = get_data(ds)
    n = len(Y)

    alpha = args.alpha
    num_trials = args.num_trials

    n_samples = len(Y)
    n_calib = int(n_samples * args.calib_ratio)
    n_test = n_samples - n_calib
    cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)

    p_values, y_test, y_test_hat, t = get_p_values(Y, Yhat, confidence, cal_indices, alpha, args, calib_ratio=args.calib_ratio, random=True)
    p_0 = p_values[y_test != y_test_hat]
    p_1 = p_values[y_test == y_test_hat]

    sns.set_style("white")
    plt.figure(figsize=(4, 3.5))

    # 绘制填充的KDE
    sns.kdeplot(p_0, color='#7F7F7F', fill=True, alpha=0.2, linewidth=0)
    sns.kdeplot(p_1, color='#2E8B57', fill=True, alpha=0.2, linewidth=0)

    # 绘制轮廓线
    sns.kdeplot(p_0, color='#7F7F7F', fill=False, linewidth=1.5, label='p_0')
    sns.kdeplot(p_1, color='#2E8B57', fill=False, linewidth=1.5, label='p_1')

    # 获取当前图形的线条
    lines = plt.gca().get_lines()

    # 检查是否有足够的线条
    if len(lines) >= 2:
        # 获取KDE数据
        x0, y0 = lines[0].get_data()  # p_0的轮廓线
        x1, y1 = lines[1].get_data()  # p_1的轮廓线

        # 只为绿色的p_1在t左侧添加阴影
        left_mask_1 = x1 <= t

        # 为p_1添加黑色阴影线条
        plt.vlines(x1[left_mask_1][::2], 0, y1[left_mask_1][::2],
                   color='black', alpha=0.6, linewidth=0.8)

    # 添加垂直线
    plt.axvline(t, color='#D62728', linestyle='--', linewidth=2.0, alpha=0.9)

    # 移除坐标轴和装饰
    plt.gca().set_frame_on(False)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("")
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    plt.savefig("right_fig.pdf", dpi=300)
    plt.show()
