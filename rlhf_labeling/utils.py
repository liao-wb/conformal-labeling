from datasets import load_dataset, concatenate_datasets, load_from_disk, Sequence, Value, Features
import re
import pickle
import pandas as pd
import os

def fix_note(example):
    for s in example["summaries"]:
        if s["note"] is None:
            s["note"] = " "
    return example

def get_dataset(args):
    if args.dataset == "tldr":
        label_list = ['A', 'B']

        """batch_files = [
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch3.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch4.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch5.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch6.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch7.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch8.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch9.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch10.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch11.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch12.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch13.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch14.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch15.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch16.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch17.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch18.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch19.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch20.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch22.json"
        ]
"""
        batch_files = [
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch3.json",
            "/mnt/sharedata/hdd/users/huanghp/comparisons/batch22.json"
        ]

        # Load each dataset individually
        datasets = []
        for file_path in batch_files:
            try:
                ds = load_dataset("json", data_files=file_path, split="train")
                ds = ds.map(fix_note)
                datasets.append(ds)
                print(f"Loaded {file_path} with {len(ds)} samples")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # Concatenate all datasets
        dataset = concatenate_datasets(datasets)

        reformat = lambda x: {
            'question': x['info']["post"],
            'choices': [x["summaries"][0]["text"], x["summaries"][1]["text"]],
            'answer': label_list[x['choice']].upper(),  # Convert 'a' -> 'A'
            'label': label_list
        }

    elif args.dataset == "mmlu":
        label_list = ['A', 'B', 'C', 'D']
        #test_dataset = load_from_disk("/mnt/sharedata/ssd_large/common/datasets/mmlu/all/test-00000-of-00001.parquet")
        #val_dataset = load_from_disk("/mnt/sharedata/ssd_large/common/datasets/mmlu/all/validation-00000-of-00001.parquet")
        #dataset = concatenate_datasets([val_dataset, test_dataset])

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
    else:
        raise NotImplementedError

    dataset = [reformat(data) for data in dataset]
    return dataset, label_list

def save_result(args, results):
    output_dir = './result/'
    output_file = f"./result/{args.dataset}_{args.model}_results.pkl"
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    df = pd.DataFrame(results)
    df.to_csv(f"./result/{args.model}_{args.dataset}.csv", sep=",", index=True)

def parse_options(options_str):
    options = re.findall(r'[a-z]\)\s*([^a-z]*)', options_str.lower())
    return [opt.strip() for opt in options]

def format_example(example):

    prompt = 'You are given a post and several candidate summaries. Chooce the summary that a human would prefer. Response with ONLY the letter of the correct options (A,B,C, ...) Do not include any explanation or extra text. \n'

    question = example['question']
    label = example['label']
    answer = example['answer']
    text = example['choices']

    prompt += ('Question: ' + question + '\n')

    for i in range(len(text)):
        prompt += label[i] + ': ' + text[i] + '\n'
    prompt += 'Answer: '

    return prompt, answer