import json
from pathlib import Path

# Define your local path
data_dir = "/mnt/e/Users/27859/PycharmProjects/select_reliable_predictions/data/stanfordnlp/SHP/askhr"

# Load each split manually
def load_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Load all splits
dataset = {
    'test': load_jsonl_file(Path(data_dir) / 'test.json'),
}

print(f"Test samples: {len(dataset['test'])}")


print()
for key, value in dataset["test"][1].items():
    print(key)
    print(value)
    print("-----")