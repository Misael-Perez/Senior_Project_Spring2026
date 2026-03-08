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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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
    
    
    
Trained_model= AutoModelForSequenceClassification.from_pretrained("./Model_1")
New_tokenizer = AutoTokenizer.from_pretrained("./Model_1")
def preprocess_function(examples):
    return New_tokenizer(examples["title"], examples["text"], truncation=True, max_length=512)

#Let's load our data
WEL_1_data= pd.read_csv("WEL1/Wel_PT1.csv")
WEL_2_data=pd.read_csv("WEL2/Wel_PT2.csv")
WEL_3_data=pd.read_csv("WEL3/Wel_PT3.csv")

#There is a problem, the model and the tokens can only be used as a hugging face dataset
#let's convert it.
WEL_1_data= Dataset.from_pandas(WEL_1_data)
WEL_2_data= Dataset.from_pandas(WEL_2_data)
WEL_3_data=Dataset.from_pandas(WEL_3_data)

#tokenize every text for both datasets
WEL_1_data=WEL_1_data.map(preprocess_function, batched=True)
WEL_2_data= WEL_2_data.map(preprocess_function, batched=True)
WEL_3_data= WEL_3_data.map(preprocess_function, batched=True)

training_args= TrainingArguments(
    output_dir="Model_1_results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4, #Changed because it doesn't require a large number for a small portion
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
    tokenizer=New_tokenizer,
    compute_metrics=compute_metrics,
)

print("Evaluation on the first WEL data")
results1= trainer.evaluate(WEL_1_data)
print(results1)
print("The Matrix")
predictions= trainer.predict(WEL_1_data)
pred= np.argmax(predictions.predictions, axis=1)
labels= predictions.label_ids
final_matrix= confusion_matrix(labels,pred)
print(final_matrix)

print("Evaluation on the second WEL data")
results2= trainer.evaluate(WEL_2_data)
print(results2)
print("The Matrix")
predictions= trainer.predict(WEL_2_data)
pred= np.argmax(predictions.predictions, axis=1)
labels= predictions.label_ids
final_matrix= confusion_matrix(labels,pred)
print(final_matrix)

print("Evaluation on the third WEL data")
results3= trainer.evaluate(WEL_3_data)
print(results3)
print("The Matrix")
predictions= trainer.predict(WEL_3_data)
pred= np.argmax(predictions.predictions, axis=1)
labels= predictions.label_ids
final_matrix= confusion_matrix(labels,pred)
print(final_matrix)