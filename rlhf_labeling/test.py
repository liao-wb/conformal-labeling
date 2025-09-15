import json
from pathlib import Path

# Define your local path
data_dir = "/mnt/e/Users/27859/PycharmProjects/select_reliable_predictions/Anthropic/hh-rlhf/helpful-base"

# Load each split manually
def load_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Load all splits
dataset = {
    'train': load_jsonl_file(Path(data_dir) / 'train.jsonl'),
    'test': load_jsonl_file(Path(data_dir) / 'test.jsonl'),
}

print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")



print()
for key, value in dataset["test"][1].items():
    print(key)
    print("Afer key")
    print(value)
    print("-----")