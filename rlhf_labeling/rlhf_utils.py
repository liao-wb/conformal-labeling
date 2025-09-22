from datasets import load_dataset, concatenate_datasets, load_from_disk, Sequence, Value, Features, Dataset, DatasetDict
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

        batch_files = [
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
    elif args.dataset == "ag_news":
        label_list = ['A', 'B', 'C', 'D']

        full_dataset = load_dataset(
            "parquet",
            data_files={
                "test": "/mnt/sharedata/hdd/users/huanghp/ag_news/data/test-00000-of-00001.parquet",}
        )
        dataset = full_dataset["test"]
        label_map_list = ["World", "Sports", "Business", "Science/Technology"]
        reformat = lambda x: {
            'question': x['text'],
            'choices': label_map_list,
            'answer': label_list[x['label']],
            'label': label_list
        }
    elif args.dataset == "dbpedia":
        label_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

        # DBpedia 的类别映射（完整的14个类别）
        label_map_list = [
            "Company", "Educational Institution", "Artist", "Athlete",
            "Office Holder", "Mean of Transportation", "Building",
            "Natural Place", "Village", "Animal", "Plant",
            "Album", "Film", "Written Work"
        ]


        full_dataset = load_dataset(
            "parquet",
            data_files={
                "test": "/mnt/sharedata/hdd/users/huanghp/dbpedia/dbpedia_14/test-00000-of-00001.parquet"
            }
        )

        dataset = full_dataset["test"]

        reformat = lambda x: {
                'question': x['content'],
                'choices': label_map_list,
                'answer': label_list[x['label']],
                'label': label_list
            }
    elif args.dataset == "misinformation":
        label_list = ['A', 'B']

        # DBpedia 的类别映射（完整的14个类别）
        label_map_list = [
            "real", "misinfo"
        ]

        train_df = pd.read_csv("/mnt/sharedata/hdd/users/huanghp/misinformation/val.tsv", sep='\t')
        test_df = pd.read_csv("/mnt/sharedata/hdd/users/huanghp/misinformation/test.tsv", sep='\t')

        # 转换为 Hugging Face Dataset
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        # 创建 DatasetDict
        full_dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })

        dataset = concatenate_datasets([full_dataset["train"], full_dataset["test"]])

        reformat = lambda x: {
                'question': x['headline'],
                'choices': label_map_list,
                'answer': "A" if x["gold_label"] == "real" else "B",
                'label': label_list
            }

    elif args.dataset == "stance":
        label_list = ["A", "B", "C"]
        file_path = "/mnt/sharedata/hdd/users/huanghp/stance/GWSD.tsv"
        df = pd.read_csv(file_path, sep='\t')

        # 初始化计数列
        df["agrees"] = 0
        df["neutral"] = 0
        df["disagrees"] = 0

        # 统计每个worker的标注
        for i in range(8):
            df["agrees"] += (df[f"worker_{i}"] == 'agrees').astype(int)
            df["neutral"] += (df[f"worker_{i}"] == 'neutral').astype(int)
            df["disagrees"] += (df[f"worker_{i}"] == 'disagrees').astype(int)

        # 确定最终标签（多数投票）
        def get_final_label(row):
            counts = {
                'agrees': row['agrees'],
                'neutral': row['neutral'],
                'disagrees': row['disagrees']
            }
            # 返回数量最多的标签
            return max(counts.items(), key=lambda x: x[1])[0]

        # 应用函数
        df["label"] = df.apply(get_final_label, axis=1)
        map_answer = {"agrees": "A", "neutral": "B", "disagrees": "C"}

        # 转换为Hugging Face Dataset格式
        dataset = Dataset.from_pandas(df)
        reformat = lambda x: {
            'question': x['sentence'],
            'choices': ["agrees", "neutral", "disagrees"],
            'answer': map_answer[x['label']],
            'label': label_list
        }

    elif args.dataset == "hh-rlhf-helpful":
        label_list = ["A", "B"]
        full_dataset = load_dataset(
            "json",
            data_files={
                'train':  '/mnt/sharedata/hdd/users/huanghp/hh-rlhf/helpful-base/train.jsonl',
            'test': "/mnt/sharedata/hdd/users/huanghp/hh-rlhf/helpful-base/1",}
        )
        dataset = concatenate_datasets([full_dataset["train"], full_dataset["test"]])
        #dataset = full_dataset["test"]
        reformat = lambda x: reformat_hh_rlhf(x)
    elif args.dataset == "hh-rlhf-harmless":
        label_list = ["A", "B"]
        full_dataset = load_dataset(
            "json",
            data_files={
                'train':  '/mnt/sharedata/hdd/users/huanghp/hh-rlhf/harmless-base/train.jsonl',
            'test': "/mnt/sharedata/hdd/users/huanghp/hh-rlhf/harmless-base/1",}
        )
        dataset = concatenate_datasets([full_dataset["train"], full_dataset["test"]])
        #dataset = full_dataset["test"]
        reformat = lambda x: reformat_hh_rlhf(x)
    elif args.dataset == "shp":
        label_list = ["A", "B"]
        data_dir = "/mnt/sharedata/hdd/users/huanghp/SHP"

        data_files = []
        subreddits = [d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
        print(subreddits)
        for subreddit in subreddits:
            subreddit_path = os.path.join(data_dir, subreddit)
            for split in ['test.json', 'validation.json']:
                file_path = os.path.join(subreddit_path, split)
                if os.path.exists(file_path):
                    data_files.append(file_path)

        if not data_files:
            raise ValueError("No SHP data files found!")

        # Load all files at once
        dataset = load_dataset('json', data_files=data_files, split="train")
        reformat = lambda x: {
            'question': x['history'],
            'choices': [x['human_ref_A'], x['human_ref_B']],
            'answer': "A" if x["labels"] == 0 else "B",
            'label': label_list,
        }

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
    elif dataset == "hh-rlhf-helpful":
        prompt = 'Evaluate two assistant responses to a human query. Choose the one that is more helpful, and aligned with human preferences. Respond with ONLY "A" or "B".\n\n'
    elif dataset == "hh-rlhf-harmless":
        prompt = 'Evaluate two assistant responses to a human query. Choose the one that is more harmless, and aligned with human preferences. Respond with ONLY "A" or "B".\n\n'
    elif dataset == "shp":
        prompt = 'You are given a question from an online forum and two candidate answers. Choose the answer that is more helpful, informative, and valuable according to human preferences. Respond with ONLY "A" or "B".\n\n'
    elif dataset == "ag_news":
        prompt = 'You are given a news article. Classify it into the correct category. Respond with ONLY the letter (A, B, C, or D) of the correct option.\n\n'
    elif dataset == "dbpedia":
        prompt = '''You are given a text excerpt from Wikipedia. Classify it into the correct category based on its content.
        Respond with ONLY the letter (A-N) of the correct category. Do not include any explanation.\n\n'''
    elif dataset == "stance":
        prompt = "You are given a statement about climate change. Determine whether the headline agrees that global warming is a serious concern. Respond with ONLY the letter (A, B, or C) of the correct option. Do not include any explanation.\n\n"
    elif dataset == "misinformation":
        prompt = '''You are a fact-checking assistant. You are given a news headline as the question. Your task is to classify whether the headline contains factual information or misinformation.
Important guidelines:
- "real": The headline presents accurate, verifiable factual information
- "misinfo": The headline contains false, misleading, or unverified claims

Consider the source, plausibility, and potential for misinformation.

Choose from: 
A) real
B) misinfo

Respond with ONLY the letter (A or B) of the correct option. Do not include any explanation or additional text.
'''

    question = example['question']
    label = example['label']
    answer = example['answer']
    text = example['choices']

    prompt += ('Question: ' + question + '\n')

    for i in range(len(text)):
        prompt += label[i] + ': ' + text[i] + '\n'
    prompt += 'Answer: '

    return prompt, answer