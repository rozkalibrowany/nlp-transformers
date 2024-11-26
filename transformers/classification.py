from Input import TEXT_REVIEW
from transformers import pipeline
import pandas as pd

classifier = pipeline("text-classification")

outputs = classifier(TEXT_REVIEW)
pd.DataFrame(outputs)
