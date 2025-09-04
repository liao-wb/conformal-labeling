from datasets import load_dataset

ds = load_dataset(
    "parquet",
    data_files={"test":"/mnt/e/Users/27859/PycharmProjects/select_reliable_predictions/llm/dataset/mmlu_pro/test-00000-of-00001.parquet",
                "validation": "/mnt/e/Users/27859/PycharmProjects/select_reliable_predictions/llm/dataset/mmlu_pro/validation-00000-of-00001.parquet"}
)
print(ds)
print()
print(ds["test"]["answer"][0])
print(ds["test"]["options"][0])


# ds = load_dataset(
#     "json",
#     data_files={"test":"/mnt/e/Users/27859/PycharmProjects/select_reliable_predictions/llm/dataset/medmcqa/dev.json",}
# )
# print(ds)
# print()
# print(ds["test"]["opa"][0])
# print(ds["test"]["cop"][0])