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
    print(args.dataset)
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
    elif args.dataset == "hh-rlhf":
        label_list = ["A", "B"]
        full_dataset = {
             'train':  '/mnt/sharedata/hdd/users/huanghp/hh-rlhf/helpful-base/train.jsonl',
            'test': "/mnt/sharedata/hdd/users/huanghp/hh-rlhf/helpful-base/test.jsonl",
        }
        dataset = concatenate_datasets([full_dataset["validation"], full_dataset["test"]])

        reformat = lambda x: reformat_hh_rlhf(x)
    else:
        raise NotImplementedError

    dataset = [reformat(data) for data in dataset]
    return dataset, label_list


def reformat_hh_rlhf(x):
    # The dataset has a 'chosen' and a 'rejected' field, each containing the full dialogue.
    # We need to extract the final assistant response from each.
    label_list = ["A", "B"]
    # The chosen (preferred) response
    chosen_text = x['chosen'].split('\n\nAssistant: ')[-1]
    # The rejected (not preferred) response
    rejected_text = x['rejected'].split('\n\nAssistant: ')[-1]

    # The 'context' or 'question' is the human's last prompt.
    # It's everything before the final "Assistant:" in the chosen string.
    human_prompt = x['chosen'].rsplit('\n\nHuman: ', 1)[-1].split('\n\nAssistant: ')[0]

    return {
        'question': human_prompt,  # The human's last question/instruction
        'choices': [chosen_text, rejected_text],  # [preferred_response, dispreferred_response]
        'answer': 'A',  # The first choice (index 0) is always the preferred one
        'label': label_list
    }

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

def format_example(example, dataset):

    if dataset == "tldr":
        prompt = 'You are given a post and several candidate summaries. Chooce the summary that a human would prefer. Response with ONLY the letter of the correct options (A,B,C, ...) Do not include any explanation or extra text. \n'
    elif dataset == "hh-rlhf":
        prompt = 'Evaluate two assistant responses to a human query. Choose the one that is more helpful, harmless, and aligned with human preferences. Respond with ONLY "A" or "B".\n\n'
    question = example['question']
    label = example['label']
    answer = example['answer']
    text = example['choices']

    prompt += ('Question: ' + question + '\n')

    for i in range(len(text)):
        prompt += label[i] + ': ' + text[i] + '\n'
    prompt += 'Answer: '

    return prompt, answer