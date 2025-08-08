
def get_confidence(llm, confidence_params, prompt, generated_answer):
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