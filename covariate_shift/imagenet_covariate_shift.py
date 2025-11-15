import numpy as np
from algorithm.select_alg import selection
from algorithm.preprocess import get_data
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="deepseek-math-7b-instruct_math500")
parser.add_argument("--calib_ratio", type=float, default=0.1, help="Calibration ratio")
parser.add_argument("--random", default="True", choices=["True", "False"])
parser.add_argument("--num_trials", type=int, default=5, help="Number of trials")
parser.add_argument("--alpha", default=0.05, type=float, help="FDR threshold q")
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

    cal_Y, cal_Yhat, cal_confidence = get_data("resnet34_imagenet")
    test_Y, test_Yhat, test_confidence = get_data("resnet34_imagenetv2")
    print(f"cal acc: {np.sum(cal_Y == cal_Yhat) / len(cal_Y)}")
    print(f"test acc: {np.sum(test_Y == test_Yhat) / len(test_Y)}")
    Y = np.concatenate((cal_Y, test_Y), axis=0)
    Yhat = np.concatenate((cal_Yhat, test_Yhat), axis=0)
    confidence = np.concatenate((cal_confidence, test_confidence), axis=0)


    alpha = args.alpha
    num_trials = args.num_trials
    fdr_list = []
    power_list = []
    selection_size_list = []
    error_list = []
    for j in range(num_trials):
        n_samples = len(Y)
        n_calib = int(n_samples * args.calib_ratio)
        n_test = n_samples - n_calib
        cal_mask = np.zeros(len(Y), dtype=bool)
        cal_mask[:n_calib] = True

        fdp, power, selection_size, selection_indices = selection(Y, Yhat, confidence, cal_mask, alpha, calib_ratio=args.calib_ratio, random=(args.random == "True"), args=args)
        print(f"{np.sum(selection_indices)}, {selection_size}")


        fdr_list.append(fdp)
        power_list.append(power)
        selection_size_list.append(selection_size)
        error_list.append(fdp * selection_size / len(Y))

    fdr_array[i] = np.array(fdr_list)
    power_array[i] = np.array(power_list)
    selection_size_array[i] = np.array(selection_size_list)

    #print(f"Mean Overall Error: {np.mean(np.array(error_list)) * 100}")
    print(f"Mean FDR: {np.mean(fdr_list)}")
    #print(f"90% Quantile Risk: {np.quantile(fdr_list, 0.9)}")
    print(f"Mean Power: {np.mean(power_list) * 100}")
    print(f"Budget save:{np.mean(selection_size_array) / len(Y) * 100}")

    print(f"Error:{100 - np.sum(Yhat==Y)/len(Y) * 100}")
