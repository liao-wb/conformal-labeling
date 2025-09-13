import pandas as pd
import numpy as np
from algorithm.select_alg import selection
from algorithm.pac_alg import pac_labeling
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from plot_utils.plot import plot_results


parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="misinfo" ,choices=["vision", "text", "all", 'stance', 'misinfo', 'bias', 'imagenet', 'imagenetv2'])
parser.add_argument("--calib_ratio", type=float, default=0.2, help="Calibration ratio")
parser.add_argument("--random", default="True", choices=["True", "False"])
parser.add_argument("--num_trials", type=int, default=1000, help="Number of trials")
parser.add_argument("--prob", default=0.05, type=float, help="FDR threshold q")
parser.add_argument("--error", default=0.05, type=float, help="pac guarantee")
parser.add_argument("--algorithm", default="cbh", choices=["bh", "sbh", "cbh", "quantbh", "integrative"])
parser.add_argument("--temperature", type=float, default=1, help="Temperature")
args = parser.parse_args()

dataset = args.datasets


fdr_array = np.zeros(shape=(args.num_trials,))
power_array = np.zeros(shape=(args.num_trials,))
selection_size_array = np.zeros(shape=(args.num_trials,))


data = pd.read_csv("./datasets/" + dataset + '.csv')
Y = data["Y"].to_numpy()
Yhat = None
if dataset in ['stance', 'misinfo', 'bias', 'sentiment']:
    Yhat = data["Yhat (GPT4o)"].to_numpy()
elif dataset in ['imagenet', 'imagenetv2', 'alphafold']:
    Yhat = data["Yhat"].to_numpy()

confidence = data["confidence"].to_numpy()
n = len(Y)
pi = np.ones(n)

def zero_one_loss(Y,Yhat):
    return np.mean(Y != Yhat, axis=-1)

alpha = args.error
num_trials = args.num_trials
fdr_list = []
power_list = []
selection_size_list = []
error_list = []


for j in tqdm(range(num_trials)):
    uncertainty = 1 - confidence + 1e-5 * np.random.normal(size=n)
    fdp, power, selection_size = pac_labeling(Y, Yhat, loss=zero_one_loss, error_hyperparameter=args.error, prob_hyperparameter=args.prob, uncertainty=uncertainty, pi=pi, num_draws=int(0.2 * n), asymptotic=True)
    fdr_list.append(fdp)
    power_list.append(power)
    selection_size_list.append(selection_size)
    error_list.append(selection_size * fdp / len(Y))
fdr_array = np.array(fdr_list)
power_array = np.array(power_list)
selection_size_array = np.array(selection_size_list)
error_array = np.array(error_list)
print(f"Results of {dataset} dataset. q = {args.error}")
print(f"Mean FDR: {np.mean(fdr_list)}")
print(f"Mean Power: {np.mean(power_list)}")
print(f"Mean Selection Size: {np.mean(selection_size_list)}")
print(f"Budget save:{np.mean(selection_size_array) / len(Y) * 100}")
print(f"Overall Error: {np.mean(error_list) * 100}")
print(f"Quantile Error: {np.quantile(error_list, 1 - args.prob) * 100}")
# plt.scatter(error_array * 100, 100 * selection_size_array / len(Y), marker='x',
#         s=60,
#         color='#2274A5',
#         alpha=0.9,
#         label='PAC labeling',
#         linewidths=2)
#
# plt.xlabel('error', fontsize=16)
# plt.ylabel('budget save (%)', fontsize=16)
#
# # Add grid with subtle style
# plt.grid(True, linestyle=':', color='gray', alpha=0.4)
#
# # Add legend with smart placement
# plt.legend(frameon=False, fontsize=16, loc='lower right')
#
# # Remove top and right borders
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
#
# # Clean layout
# plt.tight_layout()
# plt.show()