from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from transformers import DataCollatorWithPadding
import pandas as pd
import numpy as np
import evaluate

from transformers import pipeline

model_path = "./Test_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

#text to test on 
TrueData= pd.read_csv("True.csv")
FakeData= pd.read_csv("Fake.csv")
text= TrueData["text"].iloc[0]
text2= FakeData["text"].iloc[0]

inputs = tokenizer(text2, return_tensors="pt",truncation=True, max_length=128)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]
print(predicted_class_id)
print(model.config.id2label[predicted_class_id])