from datasets import load_dataset
import re
import pickle
import pandas as pd
import os

def get_dataset(args):
    dataset_name = args.dataset
    full_dataset = load_dataset('json', data_files={
        'test': f'dataset/{dataset_name}/test.json',
        'validation': f'dataset/{dataset_name}/dev.json'  # dev.json is typically validation
    })
    cal_dataset = full_dataset['validation']
    test_dataset = full_dataset['test']

    if args.dataset == "mathqa":
        label_list = ['A', 'B', 'C', 'D', 'E']
        reformat = lambda x: {
            'question': x['Problem'],
            'choices': parse_options(x['options']),
            'answer': x['correct'].upper(),  # Convert 'a' -> 'A'
            'label': label_list
        }
    elif args.dataset == "medmcqa":
        label_list = ['A', 'B', 'C', 'D']

        # Ensure all required fields exist, handle missing/null values
        reformat = lambda x: {
            "question": x.get("question", ""),  # Fallback to empty string if missing
            "choices": [
                x.get("opa", ""),
                x.get("opb", ""),
                x.get("opc", ""),
                x.get("opd", "")
            ],
            "answer": str(x.get("cop", "A")),  # Ensure 'cop' is treated as string (e.g., '1' -> 'A')
            "label": label_list
        }
    else:
        raise NotImplementedError

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