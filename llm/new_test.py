from vllm import LLM, SamplingParams
import numpy as np
import argparse
from tqdm import tqdm
from vllm.sampling_params import GuidedDecodingParams
import torch
from utils import get_dataset, format_example, save_result

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen3-8B")
parser.add_argument("--dataset", type=str, default="arc_easy")
parser.add_argument("--tensor_parallel_size", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=32)
#parser.add_argument("--subject", type=str, default="college_biology")  # MMLU has different subjects


args = parser.parse_args()

batch_size = args.batch_size
OUTPUT_FILE = f"output/{args.model}_{args.dataset}_results.json"
model_path =f"/mnt/sharedata/ssd_large/common/LLMs/{args.model}"

dataset, label_list = get_dataset(args)


guided_decoding_params = GuidedDecodingParams(choice=label_list)
sampling_params = SamplingParams(guided_decoding=guided_decoding_params, logprobs=5, max_tokens=1)

results = {
    "Yhat": [],
    "Y": [],
    "is_correct": [],
    "confidence": [],
}


indices = len(dataset)
input_texts, labels = [], []
for example in dataset:
    input_text, label = format_example(example)
    input_texts.append(input_text)
    labels.append(label)


