import pandas as pd
import numpy as np
from algorithm.select_alg import new_selection
from algorithm.preprocess import get_preference_data
import argparse

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
budget_save_list1 = []

size_list2 = []
fdr_list2 = []
power_list2 = []
budget_save_list2 = []

size_list3 = []
fdr_list3 = []
power_list3 = []
budget_save_list3 = []

total_fdr_list = []
alpha_list = [0.2]

for alpha in alpha_list:
    for _ in range(args.num_trials):

        Y, Yhat_1, confidence_1, _ = get_preference_data(dataset="mistral-7b-instruct")
        total_size = len(Y)

        shuffle_indices = np.random.permutation(len(Y))
        Y = Y[shuffle_indices]
        confidence_1 = confidence_1[shuffle_indices]
        Yhat_1 = Yhat_1[shuffle_indices]

        _, Yhat_2, confidence_2, _ = get_preference_data(dataset="gpt-3.5-turbo")

        confidence_2 = confidence_2[shuffle_indices]
        Yhat_2 = Yhat_2[shuffle_indices]

        _, Yhat_3, confidence_3, _ = get_preference_data(dataset="gpt-4-turbo")

        confidence_3 = confidence_3[shuffle_indices]
        Yhat_3 = Yhat_3[shuffle_indices]


        n_calib = int(0.2 * len(Y))
        y_calib = Y[:n_calib]
        y_hat_calib1 = Yhat_1[:n_calib]
        conf_calib1 = confidence_1[:n_calib]
        y_hat_calib2 = Yhat_2[:n_calib]
        conf_calib2 = confidence_2[:n_calib]
        y_hat_calib3 = Yhat_3[:n_calib]
        conf_calib3 = confidence_3[:n_calib]

        y_test = Y[n_calib:]
        y_hat1_test = Yhat_1[n_calib:]
        conf1_test = confidence_1[n_calib:]
        y_hat2_test = Yhat_2[n_calib:]
        conf2_test = confidence_2[n_calib:]
        y_hat3_test = Yhat_3[n_calib:]
        conf3_test = confidence_3[n_calib:]

        fdr1, power1, selection_size1, selection_indices1 = new_selection(y_calib, y_hat_calib1, conf_calib1, y_test, y_hat1_test, conf1_test, alpha, random=True, args=args)
        budget_save_list1.append(selection_size1 / total_size)

        fdr_list1.append(fdr1)
        power_list1.append(power1)
        size_list1.append(selection_size1)

        idx = (selection_indices1 == 0)
        y_test = y_test[idx]
        y_hat2_test = y_hat2_test[idx]
        conf2_test = conf2_test[idx]
        y_hat3_test = y_hat3_test[idx]
        conf3_test = conf3_test[idx]

        if len(y_test) == 0:
            fdr2, power2, selection_size2 = 0, 0, 0
            fdr3, power3, selection_size3 = 0, 0, 0
            fdp = (selection_size1 * fdr1 + selection_size2 * fdr2 + selection_size3 * fdr3) / (
                        selection_size1 + selection_size2 + selection_size3)
            total_fdr_list.append(fdp)
            budget_save_list2.append(0.0)
            budget_save_list3.append(0.0)
            continue
        else:
            fdr2, power2, selection_size2, selection_indices2 = new_selection(y_calib, y_hat_calib2, conf_calib2, y_test, y_hat2_test, conf2_test, alpha, random=True, args=args)
            budget_save_list2.append(selection_size2 / total_size)
        fdr_list2.append(fdr2)
        power_list2.append(power2)
        size_list2.append(selection_size2)


        idx = (selection_indices2 == 0)
        y_test = y_test[idx]
        y_hat3_test = y_hat3_test[idx]
        conf3_test = conf3_test[idx]

        if len(y_test) == 0:
            fdr3, power3, selection_size3 = 0, 0, 0
            fdp = (selection_size1 * fdr1 + selection_size2 * fdr2 + selection_size3 * fdr3) / (
                    selection_size1 + selection_size2 + selection_size3)
            total_fdr_list.append(fdp)
            budget_save_list3.append(0.0)
            continue

        fdr3, power3, selection_size3, selection_indices3 = new_selection(y_calib, y_hat_calib3, conf_calib3, y_test, y_hat3_test, conf3_test, alpha, random=True, args=args)

        fdr_list3.append(fdr3)
        power_list3.append(power3)
        size_list3.append(selection_size3)
        budget_save_list3.append(selection_size3 / total_size)

        fdp = (selection_size1 * fdr1 + selection_size2 * fdr2 + selection_size3 * fdr3) / (selection_size1 + selection_size2 + selection_size3)
        total_fdr_list.append(fdp)
    print("total fdr")
    print(np.mean(total_fdr_list))
    print(np.mean(budget_save_list1), np.mean(budget_save_list2), np.mean(budget_save_list3))
    print(np.mean(budget_save_list1) + np.mean(budget_save_list2) + np.mean(budget_save_list3))
