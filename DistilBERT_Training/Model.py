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

accuracy_metrics= evaluate.load("accuracy")
precision_metrics= evaluate.load("precision")
recall_metrics= evaluate.load("recall")
f1_metrics= evaluate.load("f1")

def compute_metrics(eval_pred):
    logits,labels=eval_pred
    preds= np.argmax(logits,axis=1)
    
    accuracy= accuracy_metrics.compute(predictions=preds, references=labels)["accuracy"]
    precision= precision_metrics.compute(predictions=preds, references=labels, average="binary")["precision"]
    recall= recall_metrics.compute(predictions=preds, references=labels, average="binary")["recall"]
    f1= f1_metrics.compute(predictions=preds,references=labels, average="binary")["f1"]
    
    return {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1":f1
    }
test_data=pd.read_csv("test.csv")

Trained_model= AutoModelForSequenceClassification.from_pretrained("./Model_1")
New_tokenizer = AutoTokenizer.from_pretrained("./Model_1")
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
    compute_metrics=compute_metrics,
)
training_metrics= trainer.train()
metrics= trainer.evaluate()
print("The trainig metrics", training_metrics.metrics)
print("The metrics\n",metrics)
Trained_model.save_pretrained("./Model_2")
New_tokenizer.save_pretrained("./Model_2")

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
