from datasets import load_dataset
from datasets import concatenate_datasets
import os
import pandas as pd
def load_data(dataset_name):
    if dataset_name == 'mmlu':
        label_list = ['A', 'B', 'C', 'D']
        # test_dataset = load_from_disk("/mnt/sharedata/ssd_large/common/datasets/mmlu/all/test-00000-of-00001.parquet")
        # val_dataset = load_from_disk("/mnt/sharedata/ssd_large/common/datasets/mmlu/all/validation-00000-of-00001.parquet")
        # dataset = concatenate_datasets([val_dataset, test_dataset])

        full_dataset = load_dataset(
            "parquet",
            data_files={
                "test": "/mnt/sharedata/ssd_large/common/datasets/mmlu/all/test-00000-of-00001.parquet",
                "validation": "/mnt/sharedata/ssd_large/common/datasets/mmlu/all/validation-00000-of-00001.parquet"}
        )
        dataset = concatenate_datasets([full_dataset["validation"], full_dataset["test"]])

        reformat = lambda x: {
            'question': x['question'],
            'choices': x['choices'],
            'answer': label_list[x['answer']],
            'label': label_list[:len(x["choices"])]
        }

        dataset = [reformat(data) for data in dataset]
        return dataset[:100]

def save_result(args, results):
    output_dir = './result/'
    output_file = f"./result/{args.dataset}_{args.model}_results.pkl"
    os.makedirs(output_dir, exist_ok=True)


    df = pd.DataFrame(results)
    df.to_csv(f"./result/{args.model}_{args.dataset}.csv", sep=",", index=True)