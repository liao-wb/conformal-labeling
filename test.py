from datasets import Dataset
import pandas as pd
import numpy as np

file_path = "/mnt/e/Users/27859/PycharmProjects/select_reliable_predictions/data/stance/GWSD.tsv"
df = pd.read_csv(file_path, sep='\t')

# 初始化计数列
df["agrees"] = 0
df["neutral"] = 0
df["disagrees"] = 0

# 统计每个worker的标注
for i in range(8):
    df["agrees"] += (df[f"worker_{i}"] == 'agrees').astype(int)
    df["neutral"] += (df[f"worker_{i}"] == 'neutral').astype(int)
    df["disagrees"] += (df[f"worker_{i}"] == 'disagrees').astype(int)

# 确定最终标签（多数投票）
def get_final_label(row):
    counts = {
        'agrees': row['agrees'],
        'neutral': row['neutral'],
        'disagrees': row['disagrees']
    }
    # 返回数量最多的标签
    return max(counts.items(), key=lambda x: x[1])[0]

# 应用函数
df["label"] = df.apply(get_final_label, axis=1)

# 转换为Hugging Face Dataset格式
dataset = Dataset.from_pandas(df)

print(f"数据集大小: {len(dataset)}")
print(f"特征: {dataset.features}")
print("\n第一个样本:")
print(dataset[0]["sentence"])
print(dataset[0]["label"])