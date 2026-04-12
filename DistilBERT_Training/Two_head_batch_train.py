from datasets import load_dataset, concatenate_datasets, interleave_datasets
from transformers import (AutoTokenizer, AutoModel,AutoModelForSequenceClassification,
    Trainer, TrainingArguments)
import torch
device = torch.device("cpu")
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import evaluate
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from datasets import Dataset
import json


""" 

We will create a model that will perform two tasks. That is why we will create a model with 
two head classification. In order to do this, we will have to create our custom model for the task
First we will create our custom model before training.

Reminder, theoretically this model can be used for student model from BERT. However, because it is a two-head classification model,
you need to use a GPU based model like BERT
"""
def map_labels(labels):
    return {"label": [label_map[m] for m in labels["label"]]}

# list your tokenizer here:
checkpoint="distilbert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

def news_tokenizer(dataset):
    return tokenizer(
        dataset["title"], dataset["text"], truncation=True,padding="max_length", max_length=512, return_token_type_ids=False
    )
    
def evidence_tokenization(dataset):
    evidences=[]
    for e in dataset["evidence"]:
        if len(e)>0:
            text=" ".join([ev[2] for ev in e])
        else:
            text=""
        evidences.append(text)

    return tokenizer(
        dataset["claim"],
        evidences,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_token_type_ids=False
    )
#The following classes are for the compute metrics
accuracy_metric= evaluate.load("accuracy")
precision_metrics= evaluate.load("precision")
recall_metrics= evaluate.load("recall")
f1_metric= evaluate.load("f1")
#These can be used individually for each task
def news_compute_metrics(eval_pred):
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
Reminder, add a column named task for each of the datasets
"""

#News articles data
train_data= pd.read_csv("train.csv")
eval_data=pd.read_csv("eval.csv")


id2label= {0: "Fake", 1:"Real"}
#Add a column named task
#Let's turn it into a dataset for the model
train_data= Dataset.from_pandas(train_data)
eval_data= Dataset.from_pandas(eval_data)
def add_task_news(data_set):
    data_set["task"]=0#News will be 0
    return data_set
#map the tokenizer
train_token=train_data.map(news_tokenizer, batched=True)
eval_token= eval_data.map(news_tokenizer, batched=True)
#add the task for training
train_token=train_token.map(add_task_news)
eval_token=eval_token.map(add_task_news)
#remove them
train_token = train_token.remove_columns(["title","text"])
eval_token = eval_token.remove_columns(["title","text"])
#The following code is used to make it work in DistilBERT



#set them to torch format
train_token.set_format("torch")
eval_token.set_format("torch")
print(train_token.column_names)


"""The following part of the code is to prepare for the tokenization of the fever gold evidence data"""
#We load the fever dataset along side witht the labels and tokenizers
feverDataset=load_dataset("copenlu/fever_gold_evidence")
label_map={
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2
}
#using the functions, we map them
feverDataset=feverDataset.map(evidence_tokenization, batched=True)
feverDataset=feverDataset.map(map_labels, batched=True)
def add_task_evidence(data_set):
    data_set["task"]=1#evidence will be 1
    return data_set

fever_train_dataset=feverDataset["train"]
fever_train_dataset=fever_train_dataset.rename_column("label","labels")
fever_train_dataset=fever_train_dataset.map(add_task_evidence)
fever_train_dataset = fever_train_dataset.select_columns(
    ["input_ids", "attention_mask", "labels", "task"]
)
print(fever_train_dataset.column_names)
fever_validation_dataset=feverDataset["validation"] # will use to evaluate model
fever_validation_dataset=fever_validation_dataset.rename_column("label","labels")
fever_validation_dataset=fever_validation_dataset.map(add_task_evidence)
fever_validation_dataset = fever_validation_dataset.select_columns(
    ["input_ids", "attention_mask", "labels", "task"]
)

fever_test_dataset=feverDataset["test"]#will use to test the model later
fever_test_dataset=fever_test_dataset.rename_column("label","labels")
fever_test_dataset=fever_test_dataset.map(add_task_evidence)
fever_test_dataset = fever_test_dataset.select_columns(
    ["input_ids", "attention_mask", "labels", "task"]
)

supports_count=fever_train_dataset["labels"].count(0)
refutes_count=fever_train_dataset["labels"].count(1)
nei_count=fever_train_dataset["labels"].count(2)


class_weights=torch.tensor([1.0/supports_count, 1.0/refutes_count, 1.0/nei_count])  
class_weights=class_weights.to(device)

#combine the datasets

class two_TaskModel(nn.Module):
    #Let's build our constructor
    #let's put a place holder for the model_name
    def __init__(self,model_name):
        super(two_TaskModel,self).__init__()
        #load the headless Model
        self.encoder = AutoModel.from_pretrained(model_name)
        #Load the number of hidden size of distilbert or RoBERTa
        hidden_size = self.encoder.config.hidden_size
        #We will create our outputs for the fake/real news, it is similar to the num_label
        self.real_or_fake= nn.Linear(hidden_size,2)
        #Same goes for the fever dataset
        self.evidence_based= nn.Linear(hidden_size,3)
        #self.current_task=None
        #To prevent overfitting we will drop some neurons during training.
        self.dropout= nn.Dropout(0.3)
    
    def forward(self,input_ids,attention_mask,labels=None,task=None):
        outputs=self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_outputs= outputs.last_hidden_state[:,0,:]
        cls_output= self.dropout(cls_outputs)
        
        #The following lines of code will decide which head to use
        if task==0:
            logits= self.real_or_fake(cls_output)
        elif task == 1:
            logits = self.evidence_based(cls_output)
        else:
            raise ValueError("Please select the right task")
        
        return {"logits": logits}
    

#We will now combine the dataset so that it would only be one single training session.
training_data=interleave_datasets([train_token,fever_train_dataset],probabilities=[0.5,0.5],seed=42)
#We will now load our verison of the model
model= two_TaskModel("distilbert-base-uncased")
model.to(device)

"""The following section would be about the training arguments and their trainer.
We would be training the model sequentially. So, first train on task 1 and then on task 2."""
#One main training arguments
main_args= TrainingArguments(
    "Single_trained_multiModel",
    learning_rate=2e-5,
    eval_strategy="steps",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=256,
    num_train_epochs=4,
    logging_steps=500,   
    save_strategy="steps",
    save_steps=1000,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False
    
)


class Two_Task_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        task=inputs.pop("task")
        print(task)
        labels=inputs.pop("labels")
        input_ids=inputs["input_ids"]
        attention_mask=inputs["attention_mask"]
        loss=0
        total_samples=0
        outputs_all={}
        
        mask_news=(task==0)
        mask_evidence=(task==1)
        if mask_news.any():
            outputs_news=model(input_ids=input_ids[mask_news],attention_mask=attention_mask[mask_news],task=0)
            logits_news=outputs_news["logits"]
            label_news=labels[mask_news]
            
            loss_news=nn.CrossEntropyLoss()(logits_news,label_news)
            loss+=loss_news*label_news.size(0)
            total_samples+=label_news.size(0)
            outputs_all["news_logits"]= logits_news
        if mask_evidence.any():
            outputs_evidence=model(input_ids=input_ids[mask_evidence],attention_mask=attention_mask[mask_evidence],task=1)
            logits_evidence= outputs_evidence["logits"]
            label_evidence=labels[mask_evidence]
            loss_evidence=nn.CrossEntropyLoss(weight=class_weights)(logits_evidence,label_evidence)
            loss+= loss_evidence*label_evidence.size(0)
            total_samples+=label_evidence.size(0)
            outputs_all["evidence_logits"]=logits_evidence
        loss=loss/total_samples
        
        return (loss,outputs_all) if return_outputs else loss

Trainer = Two_Task_Trainer(
    model,
    main_args,
    train_dataset=training_data,
    eval_dataset=fever_validation_dataset,
    processing_class=tokenizer,
)

Trainer.train()
#Now, we will see the individual metrics of the predicts
news_pred= Trainer.predict(eval_token)
news_results= news_compute_metrics((news_pred.predictions,news_pred.label_ids))
evidence_preds= Trainer.predict(fever_validation_dataset)
evidence_result=evidence_compute_metrics((evidence_preds.predictions,evidence_preds.label_ids))








testPredictions=Trainer.predict(fever_test_dataset)
print(dir(testPredictions))
print(testPredictions.label_ids)
print("---------------------")
print(testPredictions.predictions)
print("---------------------")
print(testPredictions.metrics)

print("METRICS: ", testPredictions.metrics)
#print(df.head())

with open('metrics_single_session.txt', 'w') as f:
    f.write(json.dumps(testPredictions.metrics, indent=4))

"""Now that we have trained the model, we will begin to save. We can't save it the regular way, so
we have to use the pytorch checkpoint save
REMINDER: In order to use the model for a test or anywhere else, you have to replicate the architecture of the model
Which is class above."""


tokenizer.save_pretrained("combined_tokenizer/")


"""Now that we have trained the model, we will begin to save. We can't save it the regular way, so
we have to use the pytorch checkpoint save
REMINDER: In order to use the model for a test or anywhere else, you have to replicate the architecture of the model
Which is class above."""
torch.save({
    "Two_task_single_session": model.state_dict(),
    "class_weights": class_weights,
    "label_map":label_map,
    "id2label":id2label
}, "TwoTask_single_session.pt")



