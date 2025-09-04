import os
from datasets import load_dataset, concatenate_datasets

_SUBJECTS = ["all"]
_SUBJECTS = ["college_biology"]
local_path = r"/mnt/e/Users/27859/PycharmProjects/select_reliable_predictions/llm\dataset\mmlu"


# Load a specific subject (e.g., global_facts)
dataset = None
for subject in _SUBJECTS:
    dataset_i = load_dataset("parquet", data_files={
        'test': f"{local_path}/{subject}/test*.parquet",
        'validation': f"{local_path}/{subject}/validation*.parquet"
    })
    dataset_i = concatenate_datasets([dataset_i["validation"], dataset_i["test"]])
    if dataset is None:
        dataset = dataset_i
    else:
        dataset = concatenate_datasets([dataset, dataset_i])
print(dataset)
print(min(dataset["answer"]))