from datasets import load_dataset

ds = load_dataset(path="json",
    data_files={
        'test': '/mnt/e/Users/27859/PycharmProjects/select_reliable_predictions/llm/dataset/arc_easy/test.jsonl'
    }
)
"""ds = load_dataset("parquet",
    data_files={
        'test': 'dataset/ScienceQA/data/test-00000-of-00001-f0e719df791966ff.parquet',
    }
)"""
print(ds["test"][0]["choices"]['text'])