from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--dataset", type=str, default="mmlu")
parser.add_argument("--tensor_parallel_size", type=int, default=4)

parser.add_argument("--subject", type=str, default="college_biology")  # MMLU has different subjects

args = parser.parse_args()

OUTPUT_FILE = "mmlu_vllm_results.json"

# vLLM Setup
llm = LLM(
    model=args.model,
    dtype="bfloat16",
    tensor_parallel_size=4,
    trust_remote_code=True
)

# Sampling params for original answer generation
generation_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

# Sampling params for P(True) confidence estimation
confidence_params = SamplingParams(
    temperature=0.01,
    top_p=0.9,
    max_tokens=2,  # Just need "Yes" or "No"
    logprobs=1
)

# Load MMLU dataset
dataset = load_dataset("lukaemon/mmlu", args.subject)['test'].select(range(10))  # First 10 examples


def get_confidence(prompt, generated_answer):
    """Calculate P(True) confidence by asking if the answer is correct"""
    confidence_prompt = f"""Consider the following question and answer:

Question: {prompt}
Answer: {generated_answer}

Is this answer correct? Respond only with 'Yes' or 'No'."""

    output = llm.generate(confidence_prompt, confidence_params)
    response = output.outputs[0].text.strip().lower()

    # Get the probability of "yes"
    if output.outputs[0].logprobs:
        # Check if "yes" is in the first token's options
        first_token_probs = output.outputs[0].logprobs[0].logprob_dict
        yes_prob = np.exp(first_token_probs.get("yes", first_token_probs.get(" Yes", -float('inf'))))
    else:
        yes_prob = 1.0 if response.startswith('yes') else 0.0

    return yes_prob


# Prompt template for MMLU (multiple choice)
def format_prompt(example):
    choices = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(example['choices'])])
    return f"""{example['question']}

Options:
{choices}

Answer with the letter only:"""


results = []
for example in dataset:
    prompt = format_prompt(example)

    # Generate initial answer
    output = llm.generate(prompt, generation_params)
    answer = output.outputs[0].text.strip()

    # Get confidence using P(True) approach
    confidence = get_confidence(prompt, answer)

    results.append({
        "question": example['question'],
        "choices": example['choices'],
        "generated_answer": answer,
        "confidence": float(confidence),
        "ground_truth": chr(65 + example['answer'])  # Convert to A/B/C/D
    })

    print(f"Q: {example['question'][:60]}...")
    print(f"Choices: {example['choices']}")
    print(f"A: {answer} (Confidence: {confidence:.2%})")
    print(f"GT: {chr(65 + example['answer'])}")
    print("---")

# Save results
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {OUTPUT_FILE}")