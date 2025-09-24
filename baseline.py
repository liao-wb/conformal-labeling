import numpy as np
from algorithm.select_alg import selection
from algorithm.preprocess import get_data
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="Llama-3.1-8B-Instruct_mmlu")
parser.add_argument("--calib_ratio", type=float, default=0.1, help="Calibration ratio")
parser.add_argument("--random", default="True", choices=["True", "False"])
parser.add_argument("--num_trials", type=int, default=1000, help="Number of trials")
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
        cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)
        all_indices = np.arange(len(Y))
        test_indices = np.setdiff1d(all_indices, cal_indices)

        #fdp, power, selection_size, selection_indices = selection(Y, Yhat, confidence, cal_indices, alpha, calib_ratio=args.calib_ratio, random=(args.random == "True"), args=args)
        y_test, y_hat_test, confidence_test = Y[test_indices], Yhat[test_indices], confidence[test_indices]
        selection_indices = confidence_test >= 0.9
        y_reject, y_hat_reject = y_test[selection_indices], y_hat_test[selection_indices]
        # y_accept, y_hat_accept = y_test[selection_indices == 0], y_hat_test[selection_indices == 0]

        selection_size = y_reject.shape[0]

        fdp = np.sum(y_reject != y_hat_reject) / selection_size if selection_size > 0 else 0
        power = np.sum(y_reject == y_hat_reject) / np.sum(y_test == y_hat_test)

        fdr_list.append(fdp)
        power_list.append(power)
        selection_size_list.append(selection_size)
        error_list.append(fdp * selection_size / len(Y))

    fdr_array[i] = np.array(fdr_list)
    power_array[i] = np.array(power_list)
    selection_size_array[i] = np.array(selection_size_list)

    #print(f"Mean Overall Error: {np.mean(np.array(error_list)) * 100}")
    print(f"Mean FDR: {np.mean(fdr_list) * 100}")
    print(f"Mean Power: {np.mean(power_list) * 100}")
    print(f"Budget save:{np.mean(selection_size_array) / len(Y) * 100}")

    print(f"Error:{100 - np.sum(Yhat==Y)/len(Y) * 100}")


