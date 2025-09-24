from vllm import LLM, SamplingParams
import numpy as np
import argparse
from tqdm import tqdm
from vllm.sampling_params import GuidedDecodingParams
import torch
from utils import get_dataset, format_example, save_result
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen3-8B")
parser.add_argument("--dataset", type=str, default="mathqa")
parser.add_argument("--tensor_parallel_size", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_model_len", type=int, default=1024)
#parser.add_argument("--subject", type=str, default="college_biology")  # MMLU has different subjects


args = parser.parse_args()

batch_size = args.batch_size
OUTPUT_FILE = f"output/{args.model}_{args.dataset}_results.json"
model_path =f"/mnt/sharedata/ssd_large/common/LLMs/{args.model}"

dataset, label_list = get_dataset(args)

print()
# vLLM Setup
model = LLM(
    model=model_path,
    gpu_memory_utilization=0.5, max_model_len=args.max_model_len,
    tensor_parallel_size=args.tensor_parallel_size,
)

guided_decoding_params = GuidedDecodingParams(choice=label_list)
sampling_params = SamplingParams(guided_decoding=guided_decoding_params, logprobs=20, max_tokens=1)

results = {
    "Yhat": [],
    "Y": [],
    "is_correct": [],
    "msp_confidence": [],
    "entropy_confidence":[],
    "alpha_confidence":[]
}


indices = len(dataset)
input_texts, labels = [], []
for example in dataset:
    input_text, label = format_example(example)
    input_texts.append(input_text)
    labels.append(label)

outputs = model.generate(
        prompts=input_texts,
        sampling_params=sampling_params,
        use_tqdm=False
    )



results = {
    "Yhat": [],
    "Y": [],
    "is_correct": [],
    "msp_confidence": [],
    "entropy_confidence":[],
    "alpha_confidence": []
}

for i, output in tqdm(enumerate(outputs)):
    outputs_dict = output.outputs[0].logprobs[0]
    token_ids = {token: next(key for key, value in outputs_dict.items() if value.decoded_token == token) for token in label_list}
    logits = [output.outputs[0].logprobs[0][id].logprob for id in token_ids.values()]

    logits = torch.tensor(logits, device="cuda")

    prob = torch.softmax(logits, dim=-1)
    y_hat_msp = torch.argmax(prob, dim=-1)
    msp_conf = prob[y_hat_msp]


    logits = [output.outputs[0].logprobs[0][id].logprob for id in token_ids.values()]
    preds = label_list[np.argmax(logits)]

    # results['question'].append(input_texts[j])
    # results['logits'].append(logits)
    logits_tensor = torch.tensor(logits, device="cuda")
    results['is_correct'].append(np.array(preds == labels[i]))
    results['Yhat'].append(preds)
    results['Y'].append(labels[i])
    results["msp_confidence"].append(torch.max(torch.softmax(logits_tensor, dim=-1)).item())
    results["entropy_confidence"].append(torch.sum(prob * torch.log(prob), dim=-1).item())
    results["alpha_confidence"].append(torch.sum(torch.softmax(logits_tensor, dim=-1) ** 2, dim=-1).item())



save_result(args, results)
