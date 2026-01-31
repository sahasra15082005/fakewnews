from transformers import AutoTokenizer
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

df = pd.read_csv("data/raw/fake_news.csv")

tokens = tokenizer(
    df["text"].tolist(),
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

print(tokens.keys())
