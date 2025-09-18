from datasets import Dataset, DatasetDict
import pandas as pd

# 先用pandas读取，然后转换为Hugging Face的Dataset格式
train_df = pd.read_parquet('/mnt/e/Users/27859/PycharmProjects/select_reliable_predictions/data/ag_news/data/train-00000-of-00001.parquet')
test_df = pd.read_parquet('/mnt/e/Users/27859/PycharmProjects/select_reliable_predictions/data/ag_news/data/test-00000-of-00001.parquet')

# 转换为Dataset格式
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 创建DatasetDict（与Hugging Face transformers兼容的格式）
dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# 使用数据集
print(dataset['train'][0])  # 查看第一条训练数据
print(dataset["train"])