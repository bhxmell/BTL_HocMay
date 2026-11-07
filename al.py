import os

# BẮT BUỘC: Đặt HF_HOME trước khi import bất kỳ thư viện Hugging Face nào
os.environ["HF_HOME"] = "D:/huggingface_cache"

from datasets import load_dataset

dataset = load_dataset("UniverseTBD/arxiv-abstracts-large")

print(dataset)
print(dataset["train"][0])