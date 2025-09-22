from datasets import load_dataset, concatenate_datasets, load_from_disk
import re
import pickle
import pandas as pd
import os


def get_dataset(args):
    dataset_name = args.dataset

    if args.cal_dataset == "mathqa":
        label_list = ['A', 'B', 'C', 'D', 'E']
        full_dataset = load_dataset('json', data_files={
            'test': f'/mnt/sharedata/ssd_large/users/huanghp/{dataset_name}/test.json',
            'validation': f'/mnt/sharedata/ssd_large/users/huanghp/{dataset_name}/dev.json'  # dev.json is typically validation
        })
        dataset = concatenate_datasets([full_dataset["test"], full_dataset["validation"]])
        reformat = lambda x: {
            'question': x['Problem'],
            'choices': parse_options(x['options']),
            'answer': x['correct'].upper(),  # Convert 'a' -> 'A'
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
    elif args.dataset == "mmlu_pro":
        label_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        full_dataset = load_dataset(
            "parquet",
            data_files={
                "test": "/mnt/sharedata/ssd_large/common/datasets/MMLU-Pro/data/test-00000-of-00001.parquet",
                "validation": "/mnt/sharedata/ssd_large/common/datasets/MMLU-Pro/data/validation-00000-of-00001.parquet"}
        )
        dataset = concatenate_datasets([full_dataset["validation"], full_dataset["test"]])
        reformat = lambda x: {
            'question': x['question'],
            'choices': x['options'],
            'answer': x['answer'].upper(),  # Convert 'a' -> 'A'
            'label': label_list[:len(x["options"])]
        }
    elif args.dataset == "medmcqa":
        label_list = ['A', 'B', 'C', 'D']
        full_dataset = load_dataset('json', data_files={
            'dev': f'/mnt/sharedata/ssd_large/users/huanghp/medmcqa/dev.json',
        })
        dataset = full_dataset["dev"]
        reformat = lambda x: {
            'question': x['question'],
            'choices': [x['opa'], x['opb'], x['opc'], x['opd']],
            'answer': label_list[x['cop'] - 1],
            'label': label_list,
        }
    elif args.dataset == "commonsenseqa":
        label_list = ['A', 'B', 'C', 'D']
        reformat = lambda x: {
            'question': x['Problem'],
            'choices': parse_options(x['options']),
            'answer': x['correct'].upper(),  # Convert 'a' -> 'A'
            'label': label_list
        }
    elif args.dataset == "arc_easy":
        full_dataset = load_dataset('json', data_files={
            'test': f'/mnt/sharedata/ssd_large/users/huanghp/arc_easy/test.jsonl',
            'validation': f'/mnt/sharedata/ssd_large/users/huanghp/arc_easy/validation.jsonl'
        })
        dataset = concatenate_datasets([full_dataset["test"], full_dataset["validation"]])
        label_list = ['A', 'B', 'C', 'D', 'E']
        reformat = lambda x: {
            'question': x['question'],
            'choices': x["choices"]["text"],
            'answer': x['label'].upper(),  # Convert 'a' -> 'A'
            'label': label_list[:len(x["choices"]["text"])]
        }
    else:
        raise NotImplementedError

    dataset = [reformat(data) for data in dataset]
    return dataset, label_list

def save_result(args, results):
    output_dir = './result/'
    output_file = f"./result/{args.cal_dataset}_{args.model}_results.pkl"
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    df = pd.DataFrame(results)
    df.to_csv(f"./result/{args.model}_{args.cal_dataset}.csv", sep=",", index=True)

def parse_options(options_str):
    options = re.findall(r'[a-z]\)\s*([^a-z]*)', options_str.lower())
    return [opt.strip() for opt in options]

def format_example(example):

    prompt = 'The following are multi choice questions. Give ONLY the correct option, no other words or explanation:\n'

    question = example['question']
    label = example['label']
    answer = example['answer']
    text = example['choices']

    prompt += ('Question: ' + question + '\n')

    for i in range(len(text)):
        prompt += label[i] + ': ' + text[i] + '\n'
    prompt += 'Answer: '

    return prompt, answer