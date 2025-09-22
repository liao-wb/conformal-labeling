from datasets import load_dataset
from vllm import LLM, SamplingParams
import numpy as np
import argparse
from tqdm import tqdm
from vllm.sampling_params import GuidedDecodingParams
import torch
from utils import  format_example, save_result
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen3-8B")
parser.add_argument("--dataset", type=str, default="mmlu")
parser.add_argument("--tensor_parallel_size", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

label_list = ['A', 'B', 'C', 'D']
label_map = {"A":0, "B":1, "C":2, "D":3}

full_dataset = load_dataset(
    "parquet",
    data_files={
        "test": "/mnt/sharedata/ssd_large/common/datasets/mmlu/all/test-00000-of-00001.parquet",
        "validation": "/mnt/sharedata/ssd_large/common/datasets/mmlu/all/validation-00000-of-00001.parquet"}
)
cal_dataset = full_dataset["validation"]
test_dataset = full_dataset["test"]

reformat = lambda x: {
    'question': x['question'],
    'choices': x['choices'],
    'answer': label_list[x['answer']],
    'label': label_list[:len(x["choices"])]
}

cal_dataset = [reformat(data) for data in cal_dataset]
test_dataset = [reformat(data) for data in test_dataset]

guided_decoding_params = GuidedDecodingParams(choice=label_list)
sampling_params = SamplingParams(guided_decoding=guided_decoding_params, logprobs=20, max_tokens=1)

model_path =f"/mnt/sharedata/ssd_large/common/LLMs/{args.model}"
model = LLM(
    model=model_path,
    gpu_memory_utilization=0.5, max_model_len=2048,
    tensor_parallel_size=args.tensor_parallel_size,
)



indices = len(cal_dataset)
cal_input_texts, labels = [], []

for example in cal_dataset:
    input_text, label = format_example(example)
    cal_input_texts.append(input_text)
    num_label = label_map[label]
    labels.append(num_label)



cal_outputs = model.generate(
        prompts=cal_input_texts,
        sampling_params=sampling_params,
        use_tqdm=False
    )

cal_logits = torch.tensor([], dtype=torch.float, device="cuda")
cal_labels = torch.tensor(labels, device="cuda")

for i, output in tqdm(enumerate(cal_outputs)):
    outputs_dict = output.outputs[0].logprobs[0]
    token_ids = {token: next(key for key, value in outputs_dict.items() if value.decoded_token == token) for token in label_list}
    logits = [output.outputs[0].logprobs[0][id].logprob for id in token_ids.values()]
    preds = label_list[np.argmax(logits)]

    logit_tensor = torch.tensor(logits, device="cuda").unsqueeze(0)
    cal_logits = torch.cat((cal_logits, logit_tensor), dim=0)


tensor_dataset = TensorDataset(cal_logits, cal_labels)

dataloader = DataLoader(tensor_dataset, batch_size=256)

t = torch.tensor(1.0, requires_grad=True, dtype=torch.float32)
optimizer = torch.optim.Adam([t], lr=0.1)

for epoch in tqdm(range(400)):
    # Batch processing for all epochs except the last
    for batch_logits, batch_label in dataloader:
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(batch_logits / t, batch_label)
        loss.backward()
        optimizer.step()

t = t.item()


results = {
    "Yhat": [],
    "Y": [],
    "is_correct": [],
    "before_confidence": [],
    "after_confidence": [],
}


indices = len(test_dataset)
input_texts, labels = [], []
for example in test_dataset:
    input_text, label = format_example(example)
    input_texts.append(input_text)
    labels.append(label)

test_outputs = model.generate(
        prompts=input_texts,
        sampling_params=sampling_params,
        use_tqdm=False
    )

for i, output in tqdm(enumerate(test_outputs)):
    outputs_dict = output.outputs[0].logprobs[0]
    token_ids = {token: next(key for key, value in outputs_dict.items() if value.decoded_token == token) for token in label_list}
    logits = [output.outputs[0].logprobs[0][id].logprob for id in token_ids.values()]
    preds = label_list[np.argmax(logits)]

    #results['question'].append(input_texts[j])
    #results['logits'].append(logits)
    results['is_correct'].append(np.array(preds == labels[i]))
    results['Yhat'].append(preds)
    results['Y'].append(labels[i])
    results["before_confidence"].append(torch.max(torch.softmax(torch.tensor(logits, device="cuda"), dim=-1)).item())
    results["after_confidence"].append(torch.max(torch.softmax(torch.tensor(logits, device="cuda") / t, dim=-1)).item())

save_result(args, results)