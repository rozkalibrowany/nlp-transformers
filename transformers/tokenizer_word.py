import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

text = "Tokenizing text is a core task of NLP."

tokenized_text = text.split()
print(tokenized_text)

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
encoded_text = tokenizer(text)s
print(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)

print(tokenizer.convert_tokens_to_string(tokens))
print(f"Vocab size {tokenizer.vocab_size}, max context: {tokenizer.model_max_length}")