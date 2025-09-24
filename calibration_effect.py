import numpy as np
from algorithm.select_alg import selection
from algorithm.preprocess import get_data
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="Qwen2.5-32B-Instruct_misinformation_calibration")
parser.add_argument("--calib_ratio", type=float, default=0.1, help="Calibration ratio")
parser.add_argument("--random", default="True", choices=["True", "False"])
parser.add_argument("--num_trials", type=int, default=100, help="Number of trials")
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
    data = pd.read_csv("./datasets/" + dataset + '.csv')
    Y = data["Y"].to_numpy()
    before_confidence = data["before_confidence"].to_numpy()
    Yhat = data["Yhat"].to_numpy()
    after_confidence = data["after_confidence"].to_numpy()

    alpha = args.alpha
    num_trials = args.num_trials
    fdr_list = []
    power_list = []
    selection_size_list = []
    error_list = []

    after_fdr_list = []
    after_power_list = []
    after_selection_size_list = []
    after_error_list = []
    for j in range(num_trials):
        n_samples = len(Y)
        n_calib = int(n_samples * args.calib_ratio)
        n_test = n_samples - n_calib
        cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)

        fdp, power, selection_size, selection_indices = selection(Y, Yhat, before_confidence, cal_indices, alpha, calib_ratio=args.calib_ratio, random=(args.random == "True"), args=args)
        after_fdp, after_power, after_selection_size, after_selection_indices = selection(Y, Yhat, after_confidence, cal_indices, alpha, calib_ratio=args.calib_ratio, random=(args.random == "True"), args=args)
        fdr_list.append(fdp)
        power_list.append(power)
        selection_size_list.append(selection_size)
        error_list.append(fdp * selection_size / len(Y))

        after_fdr_list.append(after_fdp)
        after_power_list.append(after_power)
        after_selection_size_list.append(after_selection_size)
        after_error_list.append(fdp * selection_size / len(Y))

    fdr_array[i] = np.array(fdr_list)
    power_array[i] = np.array(power_list)
    selection_size_array[i] = np.array(selection_size_list)

    #print(f"Mean Overall Error: {np.mean(np.array(error_list)) * 100}")
    print(f"Mean FDR: {np.mean(fdr_list)}")
    print(f"Mean Power: {np.mean(power_list) * 100}")


    print(f"Error:{100 - np.sum(Yhat==Y)/len(Y) * 100}")

    print(f"After Mean FDR: {np.mean(after_fdr_list)}")
    print(f"After Mean Power: {np.mean(after_power_list) * 100}")


    print(f"After Error:{100 - np.sum(Yhat == Y) / len(Y) * 100}")


def calculate_ece_multiclass(y_true, y_pred, confidence_scores, n_bins=10):
    """
    计算多分类问题的预期校准误差 (Expected Calibration Error, ECE)

    参数:
    y_true: 真实标签
    y_pred: 预测标签
    confidence_scores: 最大softmax概率（置信度）
    n_bins: 分箱数量

    返回:
    ece: 预期校准误差
    """
    # 确保输入是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confidence_scores = np.array(confidence_scores)

    # 检查预测是否正确
    correct_predictions = (y_pred == y_true).astype(float)

    # 按置信度分箱
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = len(y_true)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到在当前bin中的样本
        in_bin_mask = np.logical_and(confidence_scores >= bin_lower,
                                     confidence_scores < bin_upper)
        bin_count = np.sum(in_bin_mask)

        if bin_count == 0:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
            continue

        # 计算bin内的准确率
        bin_accuracy = np.mean(correct_predictions[in_bin_mask])
        # 计算bin内的平均置信度
        bin_confidence = np.mean(confidence_scores[in_bin_mask])

        bin_accuracies.append(bin_accuracy)
        bin_confidences.append(bin_confidence)
        bin_counts.append(bin_count)

        # 计算该bin的校准误差
        bin_ece = np.abs(bin_accuracy - bin_confidence) * bin_count
        ece += bin_ece

    # 归一化
    ece = ece / total_samples

    return ece

print(f"Before ECE: {calculate_ece_multiclass(Y, Yhat, before_confidence)}")
print(f"After ECE: {calculate_ece_multiclass(Y, Yhat, after_confidence)}")