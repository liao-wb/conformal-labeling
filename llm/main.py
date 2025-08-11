from vllm import LLM, SamplingParams
from datasets import load_dataset
import os
import numpy as np
import argparse
import re
from tqdm import tqdm
from vllm.sampling_params import GuidedDecodingParams
import pickle
import torch
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="TinyLlama-1.1B")
parser.add_argument("--dataset", type=str, default="mathqa")
parser.add_argument("--tensor_parallel_size", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=32)
#parser.add_argument("--subject", type=str, default="college_biology")  # MMLU has different subjects


args = parser.parse_args()

batch_size = args.batch_size
OUTPUT_FILE = f"output/{args.model}_{args.dataset}_results.json"

model_path =f"/mnt/sharedata/ssd_large/common/LLMs/{args.model}"
dataset_path = f"./dataset/{args.dataset}"

#if args.dataset == "mmlu":
 #   dataset_path =f"/mnt/sharedata/ssd_large/common/datasets/{args.dataset}/{args.subject}"
full_dataset = load_dataset('json', data_files={
    'train': './dataset/mathqa/train.json',
    'test': './dataset/mathqa/test.json',
    'validation': './dataset/mathqa/dev.json'  # dev.json is typically validation
})

# vLLM Setup
model = LLM(
    model=model_path,
    gpu_memory_utilization=0.8, max_model_len=2048, guided_decoding_backend="xgrammar",
    tensor_parallel_size=args.tensor_parallel_size,

)
tokenizer = model.get_tokenizer()


# Extract splits (first 10 examples each)
test_dataset = full_dataset["test"]
cal_dataset = full_dataset["validation"]
label_list = ['A', 'B', 'C', 'D', 'E']

def parse_options(options_str):
    options = re.findall(r'[a-z]\)\s*([^a-z]*)', options_str.lower())
    return [opt.strip() for opt in options]

# Reformat function for MathQA (supports A-E)
reformat = lambda x: {
    'question': x['Problem'],
    'choices': parse_options(x['options']),
    'answer': x['correct'].upper(),  # Convert 'a' -> 'A'
    'label': [label_list][:len(parse_options(x['options']))]  # Now supports up to E
}

# Apply reformatting
cal_dataset = [reformat(data) for data in cal_dataset]
test_dataset = [reformat(data) for data in test_dataset]

# Initialize results (with proper population)
results = {
    "question": [],
    "Yhat": [],
    "Y": [],
    "logits": [],
    "is_correct": [],
    "confidence": [],
}

guided_decoding_params = GuidedDecodingParams(choice=label_list)
sampling_params = SamplingParams(guided_decoding=guided_decoding_params, logprobs=5, max_tokens=1)


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

tokens_in_order = ['A', 'B', 'C', 'D', "E"]

indices = len(cal_dataset)
for i in tqdm(range(0, indices, batch_size)):
    batch = cal_dataset[i:i + batch_size]
    input_texts, labels = [], []

    for example in batch:
        input_text, label = format_example(example)
        input_texts.append(input_text)
        labels.append(label)

    outputs = model.generate(
        prompts=input_texts,
        sampling_params=sampling_params,
        use_tqdm=False
    )

    for j, output in enumerate(outputs):
        outputs_dict = output.outputs[0].logprobs[0]
        token_ids = {token: next(key for key, value in outputs_dict.items() if value.decoded_token == token) for token in tokens_in_order}
        logits = [output.outputs[0].logprobs[0][id].logprob for id in token_ids.values()]
        preds = label_list[np.argmax(logits)]

        results['question'].append(input_texts[j])
        results['is_correct'].append(np.array(preds == labels[j]))
        results['logits'].append(logits)
        results['Yhat'].append(preds)
        results['Y'].append(labels[j])
        results["confidence"].append(torch.max(torch.softmax(torch.tensor(logits, device="cuda"), dim=-1)).item())

output_dir = './result/'
output_file = f"./result/{args.dataset}_{args.model}_results.pkl"
# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)  # `exist_ok=True` prevents errors if dir already exists

# Save the pickle file
with open(output_file, 'wb') as f:
    pickle.dump(results, f)

#with open(f'./result/{args.dataset}_{args.model}_results.pkl', "wb") as f:
#    pickle.dump(results, f)

df = pd.DataFrame(results)
# Save as plain text (tab-separated for readability)
df.to_csv(f"./result/{args.model}_{args.dataset}.csv", sep=",", index=True)

#torch.cuda.empty_cache()
#del model
#del tokenizer