from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

# emotions
emotions = load_dataset("dair-ai/emotion")
emotions.set_format(type="pandas")
df = emotions["train"][:]

df["label_name"] = df["label"].apply(label_int2str)
print(df.head())

df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency")
plt.show()