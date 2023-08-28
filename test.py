from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize


dataset = load_dataset(
    "ChilleD/StrategyQA",
    download_mode="force_redownload",
)

# get dataset name
print(dataset["train"][0]["answer"])

print(f'The answer is {dataset["train"][0]["answer"]}.')
