from datasets import Dataset
import pandas as pd
import numpy as np

file_path = "/mnt/e/Users/27859/PycharmProjects/select_reliable_predictions/data/misinformation/test.tsv"
df = pd.read_csv(file_path, sep='\t')

print(len(df))
print(df.columns)
i = 1000
print(df.iloc[i]["gold_label"])
print(df.iloc[i]["headline"])
print(df.iloc[i]["writer_intent"])