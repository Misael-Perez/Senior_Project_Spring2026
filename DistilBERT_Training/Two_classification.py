from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModel,AutoModelForSequenceClassification,
    Trainer, TrainingArguments)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
import pandas as pd
import numpy as np
import evaluate
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


""" 

We will create a model that will perform two tasks. That is why we will create a model with 
two head classification. In order to do this, we will have to create our custom model for the task
First we will create our custom model before training.

Reminder, theoretically this model can be used for student model from BERT. However, because it is a two-head classification model,
you need to use a GPU based model like BERT
"""
class two_TaskModel(nn.Module):
    #Let's build our constructor
    #let's put a place holder for the model_name
    def __init__(self,model_name="roberta-base"):
        super(two_TaskModel,self).__init__()
        #load the headless Model
        self.encoder = AutoModel.from_pretrained(model_name)
        #Load the number of hidden size of distilbert or RoBERTa
        hidden_size = self.encoder.config.hidden_size
        #We will create our outputs for the fake/real news, it is similar to the num_label
        self.real_or_fake= nn.Linear(hidden_size,2)
        #Same goes for the fever dataset
        self.evidence_based= nn.Linear(hidden_size,3)
        #To prevent overfitting we will drop some neurons during training.
        self.dropout= nn.Dropout(0.3)
    
    def forward(self,input_ids,attention_mask,task,labels=None):
        outputs=self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_outputs= outputs.last_hidden_state[:,0]
        cls_output= self.dropout(cls_outputs)
        
        #The following lines of code will decide which head to use
        if task=="News":
            logits= self.real_or_fake(cls_output)
            loss_fn= nn.CrossEntropyLoss()
        elif task == "evidence":
            logits = self.evidence_based(cls_output)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError("Please select the right task")
        loss=None
        return {"loss": loss, "logits": logits}
    def data_tokenizer(self,dataset,task,)
#This class is to tokenizer the kaggle dataset used for distilBERT   
class News_Datasets(Dataset):
    def __init__(self,texts,labels,tokenizer):
        self.texts=texts
        self.labels=labels
        self.tokenizer=tokenizer
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding=self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx]),
            "task":"News"
        }
#class is to tokenizer the fever gold dataset
class Evidence_Datasets(Dataset):
    def __init__(self,dataset,labels,tokenizer):
        self.texts=dataset
        self.labels=labels
        self.tokenizer=tokenizer
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        evidences=[]
        for e in idx["evidence"]:
            if len(e)>0:
                text=" ".join([ev[2] for ev in e])
            else:
                text=""
            evidences.append(text)
        encoding= self.tokenizer(
            self.dataset["claim"],
            self.evidences,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
            
        )
        return{
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx]),
            "task":"evidence"
        }
# A function that will set the training arguments base on the task
def get_Arguments(task):
    if (task=="News"):
        return TrainingArguments(
        output_dir="news",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        )
    if (task=="evidence"):
        return TrainingArguments(
        "evidence",
        eval_strategy="steps",
        eval_steps=1000,
        learning_rate=2e-5,                 
        per_device_train_batch_size=32,
        per_device_eval_batch_size=256,     
        num_train_epochs=3,                 
        weight_decay=0.01,
        logging_steps=500,   
        save_strategy="steps",
        save_steps=1000,                     
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True
)
#The following classes are for the compute metrics
accuracy_metric= evaluate.load("accuracy")
precision_metrics= evaluate.load("precision")
recall_metrics= evaluate.load("recall")
f1_metric= evaluate.load("f1")
#These can be used individually for each task
def new_compute_metrics(eval_pred):
    logits,labels=eval_pred
    preds= np.argmax(logits,axis=1)
    
    accuracy= accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    precision= precision_metrics.compute(predictions=preds, references=labels, average="binary")["precision"]
    recall= recall_metrics.compute(predictions=preds, references=labels, average="binary")["recall"]
    f1= f1_metric.compute(predictions=preds,references=labels, average="binary")["f1"]
    
    return {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1":f1
    }
#These can be used individually for each task
def evidence_compute_metrics(eval_preds):
    logits, labels=eval_preds
    predicts=np.argmax(logits, axis=-1)
    acc=accuracy_metric.compute(predictions=predicts, references=labels)["accuracy"]
    macro_f1=f1_metric.compute(predictions=predicts, references=labels, average="macro")["f1"]    #macro since there are way more "SUPPORTS" then the other 2 labels
    weighted_f1=f1_metric.compute(predictions=predicts, references=labels, average="weighted")["f1"]
    per_class=f1_metric.compute(predictions=predicts, references=labels, average=None )["f1"]
    return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "f1_supports": per_class[0],
            "f1_refutes": per_class[1],
            "f1_nei": per_class[2],
            }



"""
To organize our datasets and the way we are going to modify them, we are going to group operations together
"""
#News articles data
train_data= pd.read_csv("train.csv")
eval_data=pd.read_csv("eval.csv")
test_data=pd.read_csv("test.csv")

id2label= {0: "Fake", 1:"Real"}

"""The following part of the code is to prepare for the tokenization of the fever gold evidence data"""
#We load the fever dataset along side witht the labels and tokenizers
feverDataset=load_dataset("copenlu/fever_gold_evidence")
label_map={
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2
}

fever_train_dataset=feverDataset["train"]
fever_validation_dataset=feverDataset["validation"] # will use to evaluate model
fever_test_dataset=feverDataset["test"] #will use to test the model later

supports_count=fever_train_dataset["label"].count(0)
refutes_count=fever_train_dataset["label"].count(1)
nei_count=fever_train_dataset["label"].count(2)
evidence_tokenizer=AutoTokenizer.from_pretrained("roberta-base")
class_weights=torch.tensor([1.0/supports_count, 1.0/refutes_count, 1.0/nei_count])  




