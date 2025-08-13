from datasets import load_dataset, Value, Features
import re
import pickle
import pandas as pd
import os

def get_dataset(args):
    dataset_name = args.dataset

    if args.dataset == "mathqa":
        full_dataset = load_dataset('json', data_files={
            'test': f'dataset/{dataset_name}/test.json',
            'validation': f'dataset/{dataset_name}/dev.json'  # dev.json is typically validation
        })

        label_list = ['A', 'B', 'C', 'D', 'E']
        reformat = lambda x: {
            'question': x['Problem'],
            'choices': parse_options(x['options']),
            'answer': x['correct'].upper(),  # Convert 'a' -> 'A'
            'label': label_list
        }
    elif args.dataset == "medmcqa":
        features = Features({
            "question": Value("string"),
            "opa": Value("string"),
            "opb": Value("string"),
            "opc": Value("string"),
            "opd": Value("string"),
            "subject_name": Value("string"),
            "topic_name": Value("null"),  # Explicitly allow nulls
            "id": Value("string"),
            "choice_type": Value("string"),
            "cop": Value("int64"),
            "exp": Value("string")
        })

        full_dataset = load_dataset('json', data_files={
            'test': f'dataset/{dataset_name}/test.json',
            'validation': f'dataset/{dataset_name}/dev.json'  # dev.json is typically validation
        }, features=features)

        label_list = ['A', 'B', 'C', 'D']
        reformat = lambda x: {
            'question': x.get("question", ""),
            'choices': [  # Combine opa, opb, opc, opd into a list like MathQA
                x.get("opa", ""),
                x.get("opb", ""),
                x.get("opc", ""),
                x.get("opd", "")
            ],
            'answer': label_list[int(x.get("cop", 1)) - 1],  # Convert cop (1-4) to 'A'-'D'
            'label': label_list
        }
    else:
        raise NotImplementedError

    cal_dataset = full_dataset['validation']
    test_dataset = full_dataset['test']

    cal_dataset = [reformat(data) for data in cal_dataset]
    test_dataset = [reformat(data) for data in test_dataset]
    return cal_dataset, test_dataset, label_list

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