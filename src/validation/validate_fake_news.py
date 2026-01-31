import pandas as pd

df = pd.read_csv("data/raw/fake_news.csv")

assert df.isnull().sum().sum() == 0
assert set(df["label"].unique()) == {0,1}

print("Validation passed")
