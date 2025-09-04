import os
from datasets import load_dataset, concatenate_datasets
_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]
_SUBJECTS = ["world_religions"]

local_path = "/mnt/e/Users/27859/PycharmProjects/select_reliable_predictions/llm/dataset/mmlu"

# Load a specific subject (e.g., global_facts)
dataset = None
for subject in _SUBJECTS:
    dataset_i = load_dataset("parquet", data_files={
        'test': f"{local_path}/{subject}/test*.parquet",
        'validation': f"{local_path}/{subject}/validation*.parquet"
    })
    dataset_i = concatenate_datasets([dataset_i["validation"], dataset_i["test"]])
    if dataset is None:
        dataset = dataset_i
    else:
        dataset = concatenate_datasets([dataset, dataset_i])
print(dataset)
print(dataset["choices"][0])