from rlhf_utils import get_dataset
from datasets import load_dataset
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="tl;dr")
args = parser.parse_args()

get_dataset(args)
ds = load_dataset(path="json",
                  data_files="/mnt/e/Users/27859/PycharmProjects/select_reliable_predictions/data/comparisons/batch19.json")

i = 100
print(ds)
print(ds["train"]["info"][i]["post"])
print()
print()
print(1)
print(ds["train"]["summaries"][i][0]["text"])
print(2)
print(ds["train"]["summaries"][i][1]["text"])
print()
print(ds["train"]["choice"][i])