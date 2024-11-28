import pandas as pd
import torch
import torch.nn.functional as F

text = "Tokenizing  text is a core task of NLP."

tokenized_text = list(text)
print(tokenized_text)

token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(token2idx)

input_ids = [token2idx[token] for token in tokenized_text]
print(input_ids)

categorical_df = pd.DataFrame({"Name": ["Roger", "Barack", "Zetta"], "Label ID": [2,1,0]})
print(categorical_df)

print(pd.get_dummies(categorical_df["Name"]))

input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
print(one_hot_encodings.shape)

print(f"Token {tokenized_text[0]}")
print(f"Tensor index: {input_ids[0]}")
print(f"One hot: {one_hot_encodings[0]}")
