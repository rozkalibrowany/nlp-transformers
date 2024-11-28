from transformers import AutoTokenizer
import datasets

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch["text"], padding = True, truncation = True)

emotions = datasets.load_dataset("dair-ai/emotion")
train = emotions["train"]
print(train["text"][:2])

# Get the first two sentences
last_two_sentences = train["text"][:2]

# Count words in each sentence
for i, sentence in enumerate(last_two_sentences, start=1):
    word_count = len(sentence.split())  # Split sentence by spaces and count
    print(f"Sentence {i}: '{sentence}'")
    print(f"Word count: {word_count}")


print(tokenize(train[:2]))

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print(emotions_encoded["train"].column_names)