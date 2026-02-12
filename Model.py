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

accuracy= evaluate.load("accuracy")
def computer_metrics(eval_pred):
    predictions,labels=eval_pred
    predictions= np.argmax(predictions,axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

test_data=pd.read_csv("test.csv")

Trained_model= AutoModelForSequenceClassification("./Test_Model")
New_tokenizer = AutoTokenizer("./Test_Model")
#New function for the new tokenizer
def New_preprocess_function(examples):
    return New_tokenizer(examples["text"], examples["title"], truncation=True, max_length=512)
#Turn the test_data into dataset
test_data= Dataset.from_pandas(test_data)
#Use our new version of the tokenizer to tokenize the test_data
test_token= test_data.map(New_preprocess_function, batched=True)
#Our training arguments
training_args= TrainingArguments(
    output_dir="Model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=62,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
trainer=Trainer(
    model=Trained_model,
    args=training_args,
    train_dataset=test_token,
    eval_dataset=test_token,
    processing_class=New_tokenizer,
    compute_metrics=computer_metrics,
)
training_metrics= trainer.train()
metrics= trainer.evaluate()
print("The trainig metrics", training_metrics.metrics)
print("The metrics\n",metrics)
Trained_model.save_pretrained("./Model")
New_tokenizer.save_pretrained("./Model")

"""
model_path = "./Test_Model"
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
"""