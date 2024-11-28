from Input import TEXT_REVIEW
from transformers import pipeline
import pandas as pd

classifier = pipeline("text-classification", device=0)

outputs = classifier(TEXT_REVIEW)
df = pd.DataFrame(outputs)

print(df)