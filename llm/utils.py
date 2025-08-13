from datasets import load_dataset, concatenate_datasets
import re
import pickle
import pandas as pd
import os

def get_dataset(args):
    dataset_name = args.dataset

    if args.dataset == "mathqa":
        label_list = ['A', 'B', 'C', 'D', 'E']
        full_dataset = load_dataset('json', data_files={
            'test': f'dataset/{dataset_name}/test.json',
            'validation': f'dataset/{dataset_name}/dev.json'  # dev.json is typically validation
        })
        dataset = concatenate_datasets([full_dataset["test"], full_dataset["validation"]])
        reformat = lambda x: {
            'question': x['Problem'],
            'choices': parse_options(x['options']),
            'answer': x['correct'].upper(),  # Convert 'a' -> 'A'
            'label': label_list
        }
    elif args.dataset == "medmcqa":
        label_list = ['A', 'B', 'C', 'D']
        full_dataset = load_dataset('json', data_files={
            'dev': f'dataset/medmcqa/dev.json',
        })
        dataset = full_dataset["dev"]
        reformat = lambda x: {
            'question': x['question'],
            'choices': [x['opa'], x['opb'], x['opc'], x['opd']],
            'answer': label_list[x['cop']],
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