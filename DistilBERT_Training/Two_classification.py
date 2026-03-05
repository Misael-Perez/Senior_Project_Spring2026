from datasets import load_dataset
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModel,AutoModelForSequenceClassification,
    Trainer, TrainingArguments)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

"""
class two_TaskModel(nn.Module):
    #Let's build our constructor
    def __init__(self,model_name="distilbert-base-uncased"):
        super(two_TaskModel,self).__init__()
        #load the headless Model
        self.encoder = AutoModel.from_pretrained(model_name)
        #Load the number of hidden size of distilbert 
        hidden_size = self.encoder.config.hidden_size
        #We will create our outputs for the fake/real news, it is similar to the num_label
        self.real_or_fake= nn.Linear(hidden_size,2)
        #Same goes for the fever dataset
        self.evidence_based= nn.Linear(hidden_size,3)
        #To prevent overfitting we will drop some neurons during training.
        self.dropout= nn.Dropout(0.3)
    
    def forward(self,input_ids,attention_mask,task,labels=None):
        outputs=self.encoder(
            input_ids=input_ids
            attention_mask=attention_mask
        )
        
        cls_outputs= outputs.last_hidden_state[:,0,:]
        cls_output= self.dropout(cls_outputs)
        
        #The following lines of code will decide which head to use
        if task=="News":
            logits= self.real_or_fake(cls_output)
        elif task == "evidence":
            logits = self.entertainment_head(cls_output)
        else:
            raise ValueError("Please select the right task")
        
        loss=None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}
    
class ourDatasets(Dataset):
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
            "labels": torch.tensor(self.labels[idx])
        }




#Let's load our new datasets for the training
#These csv files are NEWS datasets
train_data= pd.read_csv("train.csv")
eval_data=pd.read_csv("eval.csv")
test_data=pd.read_csv("test.csv")

#The following datasets are from the fever dataset
feverDataset=load_dataset("copenlu/fever_gold_evidence")


tokenizer= AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
id2label= {0: "Fake", 1:"Real"}






