import pandas as pd
import numpy as np
from algorithm.select_alg import new_selection
from algorithm.preprocess import get_preference_data
import argparse

def selection(data, n_calib, alpha, args):
    Y = data["Y"].to_numpy()
    confidence = data["confidence"].to_numpy()
    Yhat = data["Yhat"].to_numpy()
    y_calib, y_test = Y[:n_calib], Y[n_calib:]
    y_hat_calib, y_hat_test = Yhat[:n_calib], Yhat[n_calib:]
    conf_calib, conf_test = confidence[:n_calib], confidence[n_calib:]

    fdr, power, selection_size, selection_indices = new_selection(y_calib, y_hat_calib, conf_calib, y_test, y_hat_test, conf_test, alpha, random=True, args=args)
    return fdr, power, selection_size, selection_indices

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="gpt-4-turbo" ,choices=["vision", "text", "all", 'stance', 'misinfo', 'bias', 'imagenet', 'imagenetv2'])
parser.add_argument("--calib_ratio", type=float, default=0.2, help="Calibration ratio")
parser.add_argument("--random", default="True", choices=["True", "False"])
parser.add_argument("--num_trials", type=int, default=100, help="Number of trials")
parser.add_argument("--alpha", default=0.05, type=float, help="FDR threshold q")
parser.add_argument("--algorithm", default="cbh", choices=["bh", "sbh", "cbh", "quantbh", "integrative"])
parser.add_argument("--temperature", type=float, default=1, help="Temperature")
args = parser.parse_args()

size_list1 = []
fdr_list1 = []
power_list1 = []

size_list2 = []
fdr_list2 = []
power_list2 = []

size_list3 = []
fdr_list3 = []
power_list3 = []

total_fdr_list = []


for _ in range(1000):
    #data1 = pd.read_csv("./datasets/" + "Qwen3-8B_medmcqa.csv")
    #data2 = pd.read_csv("./datasets/" + "Qwen3-32B_medmcqa.csv")
    #data3 = pd.read_csv("./datasets/" + "Llama-3.1-70B-Instruct_medmcqa.csv")

    _,_,_, data1 = get_preference_data("mistral-7b-instruct")
    _,_,_, data2 = get_preference_data("gpt-3.5-turbo")
    _, _, _, data3 = get_preference_data("gpt-4-turbo")

    shuffle_indices = np.random.permutation(len(data1))
    data1 = data1.iloc[shuffle_indices]
    data2 = data2.iloc[shuffle_indices]
    data3 = data3.iloc[shuffle_indices]

    alpha = 0.1
    n_calib = int(len(data1) * args.calib_ratio / 3)

    fdr1, power1, selection_size1, selection_indices1 = selection(data1,  n_calib=n_calib, alpha=alpha, args=args)
    data2 = data2[n_calib:][selection_indices1 == 0]
    data3 = data3[n_calib:][selection_indices1 == 0]
    fdr2, power2, selection_size2, selection_indices2 = selection(data2,  n_calib=n_calib, alpha=alpha, args=args)
    data3 = data3[n_calib:][selection_indices2 == 0]
    fdr3, power3, selection_size3, selection_indices3 = selection(data3,  n_calib=n_calib, alpha=alpha, args=args)

    fdr_list1.append(fdr1)
    fdr_list2.append(fdr2)
    fdr_list3.append(fdr3)
    power_list1.append(power1)
    power_list2.append(power2)
    power_list3.append(power3)

    size_list1.append(selection_size1)
    size_list2.append(selection_size2)
    size_list3.append(selection_size3)

    fdp = (selection_size1 * fdr1 + selection_size2 * fdr2 + selection_size3 * fdr3) / (
                selection_size1 + selection_size2 + selection_size3)
    total_fdr_list.append(fdp)


print("1")
print(np.mean(fdr_list1), np.mean(power_list1), np.mean(size_list1))

print("2")
print(np.mean(fdr_list2), np.mean(power_list2), np.mean(size_list2))

print("3")
print(np.mean(fdr_list3), np.mean(power_list3), np.mean(size_list3))

print("Total Selection Size")
size_list1 = np.array(size_list1)
size_list2 = np.array(size_list2)
size_list3 = np.array(size_list3)
print(np.mean(size_list1 + size_list2 + size_list3))

#print('Total budget save')
#print((np.mean(size_list1) + np.mean(size_list2) + np.mean(size_list3)) / len(Y))
print("total fdr")
print(np.mean(total_fdr_list))
print("Total fdr var")
print(np.var(total_fdr_list))

fdr_list1 = np.array(fdr_list1)
fdr_list2 = np.array(fdr_list2)
size_list1 = np.array(size_list1)
size_list2 = np.array(size_list2)

def covariance_pandas(x_list, y_list):
    """
    Calculates the sample covariance between two lists using pandas.

    Args:
        x_list (list): List of numerical values for the first variable.
        y_list (list): List of numerical values for the second variable.

    Returns:
        float: The sample covariance between the two lists.

    Raises:
        ValueError: If the lists are of different lengths.
    """
    # Create Series objects
    x_series = pd.Series(x_list)
    y_series = pd.Series(y_list)

    # Use the .cov() method
    return x_series.cov(y_series)

print(covariance_pandas(size_list1 / (size_list1 + size_list2), fdr_list1))
print(covariance_pandas(size_list2 / (size_list1 + size_list2), fdr_list2))