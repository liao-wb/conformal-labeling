"""
Utility functions for LLM experiments.
Handles dataset loading, prompt formatting, and vLLM output parsing.
"""

import os
import pickle
import re
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets

# Map dataset names to their relative paths (from data_root)
# Adjust these relative paths according to your actual zip file structure
DATASET_PATHS = {
    "mathqa": "mathqa",  # Expecting test.json/dev.json inside
    "mmlu": "mmlu/all",
    "mmlu_pro": "MMLU-Pro/data",
    "medmcqa": "medmcqa",
    "arc_easy": "arc_easy",
    "commonsenseqa": "commonsenseqa"
}

def parse_options(options_str: str) -> List[str]:
    """Parses option strings like '(a) option1 (b) option2'."""
    options = re.findall(r'[a-z]\)\s*([^a-z]*)', options_str.lower())
    return [opt.strip() for opt in options]

def get_dataset(args) -> Tuple[List[Dict], List[str]]:
    """
    Loads and formats the specified dataset.
    
    Returns:
        dataset: A list of formatted examples.
        label_list: List of valid labels (e.g., ['A', 'B', 'C', 'D']).
    """
    dataset_name = args.dataset
    data_root = args.data_root
    
    # Construct base path
    base_path = os.path.join(data_root, DATASET_PATHS.get(dataset_name, dataset_name))
    
    if dataset_name == "mathqa":
        label_list = ['A', 'B', 'C', 'D', 'E']
        full_dataset = load_dataset('json', data_files={
            'test': os.path.join(base_path, 'test.json'),
            'validation': os.path.join(base_path, 'dev.json')
        })
        raw_dataset = concatenate_datasets([full_dataset["test"], full_dataset["validation"]])
        
        def reformat(x):
            return {
                'question': x['Problem'],
                'choices': parse_options(x['options']),
                'answer': x['correct'].upper(),
                'label': label_list
            }

    elif dataset_name == "mmlu":
        label_list = ['A', 'B', 'C', 'D']
        full_dataset = load_dataset("parquet", data_files={
            "test": os.path.join(base_path, "test-00000-of-00001.parquet"),
            "validation": os.path.join(base_path, "validation-00000-of-00001.parquet")
        })
        raw_dataset = concatenate_datasets([full_dataset["validation"], full_dataset["test"]])

        def reformat(x):
            return {
                'question': x['question'],
                'choices': x['choices'],
                'answer': label_list[x['answer']],
                'label': label_list[:len(x["choices"])]
            }

    elif dataset_name == "mmlu_pro":
        label_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        full_dataset = load_dataset("parquet", data_files={
            "test": os.path.join(base_path, "test-00000-of-00001.parquet"),
            "validation": os.path.join(base_path, "validation-00000-of-00001.parquet")
        })
        raw_dataset = concatenate_datasets([full_dataset["validation"], full_dataset["test"]])

        def reformat(x):
            return {
                'question': x['question'],
                'choices': x['options'],
                'answer': x['answer'].upper(),
                'label': label_list[:len(x["options"])]
            }

    elif dataset_name == "medmcqa":
        label_list = ['A', 'B', 'C', 'D']
        # Note: MedMCQA usually only provides dev/train in some distributions
        full_dataset = load_dataset('json', data_files={
            'dev': os.path.join(base_path, 'dev.json'),
        })
        raw_dataset = full_dataset["dev"]

        def reformat(x):
            return {
                'question': x['question'],
                'choices': [x['opa'], x['opb'], x['opc'], x['opd']],
                'answer': label_list[x['cop'] - 1],
                'label': label_list,
            }

    elif dataset_name == "arc_easy":
        label_list = ['A', 'B', 'C', 'D', 'E']
        full_dataset = load_dataset('json', data_files={
            'test': os.path.join(base_path, 'test.jsonl'),
            'validation': os.path.join(base_path, 'validation.jsonl')
        })
        raw_dataset = concatenate_datasets([full_dataset["test"], full_dataset["validation"]])

        def reformat(x):
            return {
                'question': x['question'],
                'choices': x["choices"]["text"],
                'answer': x['label'].upper(),
                'label': label_list[:len(x["choices"]["text"])]
            }

    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")

    formatted_dataset = [reformat(data) for data in raw_dataset]
    return formatted_dataset, label_list

def format_example(example: Dict, prompt_type: int = 1) -> Tuple[str, str]:
    """Formats a single example into a prompt."""
    question = example['question']
    labels = example['label']
    answer = example['answer']
    choices = example['choices']

    prompt = ""
    
    if prompt_type == 1:
        prompt = "The following are multi choice questions. Give ONLY the correct option, no other words or explanation:\n"
        prompt += f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{labels[i]}: {choice}\n"
        prompt += "Answer: "
        
    elif prompt_type == 2:
        prompt = "You will be given multiple-choice questions. Respond with ONLY the letter of the correct choice. No explanations.\n"
        prompt += f"\nQuestion: {question}\n\n"
        for i, choice in enumerate(choices):
            prompt += f"{labels[i]}: {choice}\n"
        prompt += "\nAnswer: "

    elif prompt_type == 3:
        prompt = "Answer the following multiple-choice question. Output ONLY the correct option (A, B, C, etc.). No other text.\n"
        prompt += f"\nQuestion: {question}\n\nOptions:\n"
        for i, choice in enumerate(choices):
            prompt += f"{labels[i]}: {choice}\n"
        prompt += "\nCorrect option: "
    else:
        raise ValueError("Invalid prompt_type. Choose 1, 2, or 3.")

    return prompt, answer

def extract_logprobs(
    vllm_output_obj: Any, 
    label_list: List[str]
) -> Tuple[str, List[float]]:
    """
    Extracts the logits for the specific target labels (A, B, C, D...) from vLLM output.
    
    Args:
        vllm_output_obj: The RequestOutput object from vLLM.
        label_list: List of valid tokens to look for.
        
    Returns:
        pred_label: The label with the highest probability.
        logits: List of log probabilities corresponding to label_list.
    """
    # Get the top logprobs from the first generated token
    # output.outputs[0].logprobs is a list (one per token), we want the first one
    top_logprobs = vllm_output_obj.outputs[0].logprobs[0]
    
    logits = []
    # vLLM returns a dict mapping ID -> Logprob. We match by decoded token string.
    # Note: This assumes the model's tokenizer maps "A" to a single token.
    
    # Create a mapping from token string to logprob object for O(1) lookup if possible,
    # but since we need to match decoded text, we iterate.
    
    for target_label in label_list:
        found = False
        for token_id, logprob_obj in top_logprobs.items():
            # Strip whitespace in case tokenizer adds spaces
            if logprob_obj.decoded_token.strip() == target_label:
                logits.append(logprob_obj.logprob)
                found = True
                break
        
        if not found:
            # Handle case where a valid label wasn't in the top-K returned logprobs
            # Assign a very low logprob (effectively 0 probability)
            logits.append(-1e9) 

    pred_idx = np.argmax(logits)
    pred_label = label_list[pred_idx]
    
    return pred_label, logits

def save_result(args, results: Dict, suffix: str = ""):
    """Saves results to pickle and CSV."""
    output_dir = getattr(args, 'output_dir', './results')
    os.makedirs(output_dir, exist_ok=True)
    
    filename_base = f"{args.dataset}_{os.path.basename(args.model)}{suffix}"
    
    pkl_path = os.path.join(output_dir, f"{filename_base}_results.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)

    csv_path = os.path.join(output_dir, f"{filename_base}.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, sep=",", index=False)
    
    print(f"Results saved to {csv_path}")
