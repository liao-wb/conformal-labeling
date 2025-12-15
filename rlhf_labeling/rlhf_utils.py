"""
Utility functions for RLHF and Text Classification experiments.
Handles dataset loading, formatting, and preprocessing.
"""

import os
import pickle
import re
from typing import List, Dict, Tuple, Any

import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict

def get_dataset(args) -> Tuple[Any, List[str]]:
    """
    Loads and formats the specified dataset.
    
    Args:
        args: Argument parser object containing 'dataset' and 'data_root'.
        
    Returns:
        dataset: HuggingFace Dataset or list of dicts.
        label_list: List of valid labels (e.g., ['A', 'B']).
    """
    dataset_name = args.dataset
    data_root = args.data_root
    
    # Define relative paths for datasets
    # Users should structure their data_root to match these or modify as needed.
    paths = {
        "tldr": "comparisons", # Expects batch*.json inside
        "mmlu": "mmlu/all",
        "ag_news": "ag_news/data",
        "dbpedia": "dbpedia/dbpedia_14",
        "misinformation": "misinformation",
        "stance": "stance",
        "hh-rlhf-helpful": "hh-rlhf/helpful-base",
        "hh-rlhf-harmless": "hh-rlhf/harmless-base",
        "shp": "SHP"
    }
    
    base_path = os.path.join(data_root, paths.get(dataset_name, dataset_name))

    if dataset_name == "tldr":
        label_list = ['A', 'B']
        # Load all batch json files in the directory
        data_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.startswith("batch") and f.endswith(".json")]
        
        datasets = []
        for file_path in sorted(data_files):
            try:
                ds = load_dataset("json", data_files=file_path, split="train")
                ds = ds.map(_fix_tldr_note)
                datasets.append(ds)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
        
        raw_dataset = concatenate_datasets(datasets)
        
        reformat = lambda x: {
            'question': x['info']["post"],
            'choices': [x["summaries"][0]["text"], x["summaries"][1]["text"]],
            'answer': label_list[x['choice']].upper(),
            'label': label_list
        }

    elif dataset_name == "mmlu":
        label_list = ['A', 'B', 'C', 'D']
        full_dataset = load_dataset("parquet", data_files={
            "test": os.path.join(base_path, "test-00000-of-00001.parquet"),
            "validation": os.path.join(base_path, "validation-00000-of-00001.parquet")
        })
        raw_dataset = concatenate_datasets([full_dataset["validation"], full_dataset["test"]])

        reformat = lambda x: {
            'question': x['question'],
            'choices': x['choices'],
            'answer': label_list[x['answer']],
            'label': label_list[:len(x["choices"])]
        }

    elif dataset_name == "ag_news":
        label_list = ['A', 'B', 'C', 'D']
        full_dataset = load_dataset("parquet", data_files={
            "test": os.path.join(base_path, "test-00000-of-00001.parquet")
        })
        raw_dataset = full_dataset["test"]
        label_map_list = ["World", "Sports", "Business", "Science/Technology"]
        
        reformat = lambda x: {
            'question': x['text'],
            'choices': label_map_list,
            'answer': label_list[x['label']],
            'label': label_list
        }

    elif dataset_name == "dbpedia":
        label_list = list("ABCDEFGHIJKLMN")
        label_map_list = [
            "Company", "Educational Institution", "Artist", "Athlete",
            "Office Holder", "Mean of Transportation", "Building",
            "Natural Place", "Village", "Animal", "Plant",
            "Album", "Film", "Written Work"
        ]
        full_dataset = load_dataset("parquet", data_files={
            "test": os.path.join(base_path, "test-00000-of-00001.parquet")
        })
        raw_dataset = full_dataset["test"]

        reformat = lambda x: {
            'question': x['content'],
            'choices': label_map_list,
            'answer': label_list[x['label']],
            'label': label_list
        }

    elif dataset_name == "misinformation":
        label_list = ['A', 'B']
        train_df = pd.read_csv(os.path.join(base_path, "val.tsv"), sep='\t')
        test_df = pd.read_csv(os.path.join(base_path, "test.tsv"), sep='\t')
        
        raw_dataset = concatenate_datasets([
            Dataset.from_pandas(train_df),
            Dataset.from_pandas(test_df)
        ])

        reformat = lambda x: {
            'question': x['headline'],
            'choices': ["real", "misinfo"],
            'answer': "A" if x["gold_label"] == "real" else "B",
            'label': label_list
        }

    elif dataset_name == "stance":
        label_list = ["A", "B", "C"]
        file_path = os.path.join(base_path, "GWSD.tsv")
        df = pd.read_csv(file_path, sep='\t')
        
        # Determine label by majority vote
        # Assuming columns worker_0 to worker_7 exist
        worker_cols = [c for c in df.columns if c.startswith("worker_")]
        
        def get_majority_label(row):
            votes = row[worker_cols].value_counts()
            if votes.empty: return "neutral" # Fallback
            return votes.idxmax()

        df["label"] = df.apply(get_majority_label, axis=1)
        map_answer = {"agrees": "A", "neutral": "B", "disagrees": "C"}
        
        raw_dataset = Dataset.from_pandas(df)
        reformat = lambda x: {
            'question': x['sentence'],
            'choices': ["agrees", "neutral", "disagrees"],
            'answer': map_answer.get(x['label'], "B"),
            'label': label_list
        }

    elif dataset_name.startswith("hh-rlhf"):
        label_list = ["A", "B"]
        full_dataset = load_dataset("json", data_files={
            'train': os.path.join(base_path, 'train.jsonl'),
            'test': os.path.join(base_path, 'test.jsonl') # Assuming standard name, adjust if needed
        })
        raw_dataset = concatenate_datasets([full_dataset["train"], full_dataset["test"]])
        reformat = _reformat_hh_rlhf

    elif dataset_name == "shp":
        label_list = ["A", "B"]
        data_files = []
        # Walk through subdirectories
        for root, _, files in os.walk(base_path):
            for file in files:
                if file in ['test.json', 'validation.json']:
                    data_files.append(os.path.join(root, file))
        
        if not data_files:
            raise ValueError(f"No SHP data files found in {base_path}")

        raw_dataset = load_dataset('json', data_files=data_files, split="train")
        reformat = lambda x: {
            'question': x['history'],
            'choices': [x['human_ref_A'], x['human_ref_B']],
            'answer': "A" if x["labels"] == 0 else "B",
            'label': label_list,
        }

    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")

    formatted_dataset = [reformat(data) for data in raw_dataset]
    return formatted_dataset, label_list


def _fix_tldr_note(example):
    """Helper to fix None notes in TLDR dataset."""
    for s in example["summaries"]:
        if s["note"] is None:
            s["note"] = " "
    return example

def _reformat_hh_rlhf(x):
    """Parses Anthropic HH-RLHF format."""
    label_list = ["A", "B"]
    chosen_text = x['chosen'].split('\n\nAssistant: ')[-1]
    rejected_text = x['rejected'].split('\n\nAssistant: ')[-1]
    
    # Extract prompt
    try:
        human_prompt = x['chosen'].rsplit('\n\nHuman: ', 1)[-1].split('\n\nAssistant: ')[0]
    except IndexError:
        human_prompt = x['chosen'][-500:] # Fallback

    return {
        'question': human_prompt,
        'choices': [chosen_text, rejected_text],
        'answer': 'A', # Chosen is always A in raw data, we will shuffle or keep as is
        'label': label_list
    }

def format_example(example: Dict, dataset_name: str) -> Tuple[str, str]:
    """Generates the prompt for the specific dataset task."""
    
    prompts = {
        "tldr": 'You are given a post and several candidate summaries. Choose the summary that a human would prefer. Respond with ONLY the letter of the correct option (A, B). Do not include any explanation.\n',
        "hh-rlhf-helpful": 'Evaluate two assistant responses to a human query. Choose the one that is more helpful and aligned with human preferences. Respond with ONLY "A" or "B".\n\n',
        "hh-rlhf-harmless": 'Evaluate two assistant responses to a human query. Choose the one that is more harmless and aligned with human preferences. Respond with ONLY "A" or "B".\n\n',
        "shp": 'You are given a question from an online forum and two candidate answers. Choose the answer that is more helpful, informative, and valuable according to human preferences. Respond with ONLY "A" or "B".\n\n',
        "ag_news": 'You are given a news article. Classify it into the correct category. Respond with ONLY the letter (A, B, C, or D) of the correct option.\n\n',
        "dbpedia": 'You are given a text excerpt from Wikipedia. Classify it into the correct category based on its content. Respond with ONLY the letter (A-N) of the correct category. Do not include any explanation.\n\n',
        "stance": 'You are given a statement about climate change. Determine whether the headline agrees that global warming is a serious concern. Respond with ONLY the letter (A, B, or C). Do not include any explanation.\n\n',
        "misinformation": 'You are a fact-checking assistant. Classify whether the headline contains factual information (real) or misinformation (misinfo). Respond with ONLY the letter (A or B).\nA) real\nB) misinfo\n',
        "mmlu": 'The following are multiple choice questions (with answers). Respond with ONLY the letter of the correct answer.\n'
    }

    base_prompt = prompts.get(dataset_name, "")
    
    question = example['question']
    labels = example['label']
    answer = example['answer']
    choices = example['choices']

    full_prompt = base_prompt + f"Question: {question}\n"
    for i, choice in enumerate(choices):
        full_prompt += f"{labels[i]}: {choice}\n"
    full_prompt += "Answer: "

    return full_prompt, answer

def save_result(args, results):
    """Saves results to pickle and CSV."""
    output_dir = getattr(args, 'output_dir', './results')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{args.dataset}_{os.path.basename(args.model)}"
    
    with open(os.path.join(output_dir, f"{filename}_results.pkl"), 'wb') as f:
        pickle.dump(results, f)

    pd.DataFrame(results).to_csv(os.path.join(output_dir, f"{filename}.csv"), index=False)
