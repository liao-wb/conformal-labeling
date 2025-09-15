import os
from datasets import load_dataset
label_list = ["A", "B"]
data_dir = "/mnt/e/Users/27859/PycharmProjects/select_reliable_predictions/data/stanfordnlp/SHP"

data_files = []
subreddits = [d for d in os.listdir(data_dir)
              if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
print(subreddits)
for subreddit in subreddits:
    subreddit_path = os.path.join(data_dir, subreddit)
    for split in ['test.json', 'validation.json']:
        file_path = os.path.join(subreddit_path, split)
        if os.path.exists(file_path):
            data_files.append(file_path)

if not data_files:
    raise ValueError("No SHP data files found!")

# Load all files at once
dataset = load_dataset('json', data_files=data_files)
reformat = lambda x: {
    'question': x['history'],
    'choices': [x['human_ref_A'], x['human_ref_B']],
    'answer': "A" if x["labels"] == 0 else "B",
    'label': label_list,
}