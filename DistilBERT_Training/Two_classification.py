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
from datasets import Dataset


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
checkpoint="roberta-base"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

def news_tokenizer(dataset):
    return tokenizer(
        dataset["title"], dataset["text"], truncation=True,padding="max_length", max_length=512
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
        max_length=512
    )
def map_labels(labels):
    return {"label": [label_map[m] for m in labels["label"]]}

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
test_data=pd.read_csv("test.csv")
id2label= {0: "Fake", 1:"Real"}
#Add a column named task
train_data["task"]="News"
eval_data["task"]="News"
test_data["task"]="News"
#Let's turn it into a dataset for the model
train_data= Dataset.from_pandas(train_data)
eval_data= Dataset.from_pandas(eval_data)
test_data=Dataset.from_pandas(test_data)
#map the tokenizer
train_token=train_data.map(news_tokenizer, batched=True)
eval_token= eval_data.map(news_tokenizer, batched=True)
test_token= test_data.map(news_tokenizer, batched=True)
#remove them
train_token = train_token.remove_columns(["title","text"])
eval_token = eval_token.remove_columns(["title","text"])
test_token = test_token.remove_columns(["title","text"])
#set them to torch format
train_token.set_format("torch")
eval_token.set_format("torch")
test_token.set_format("torch")

"""The following part of the code is to prepare for the tokenization of the fever gold evidence data"""
#We load the fever dataset along side witht the labels and tokenizers
feverDataset=load_dataset("copenlu/fever_gold_evidence")
label_map={
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2
}
#using the functions, we map them
def addcolumn(data):
    data["task"]="evidence"
    return data
feverDataset= feverDataset.map(addcolumn)
feverDataset=feverDataset.map(evidence_tokenization, batched=True)
feverDataset=feverDataset.map(map_labels, batched=True)

fever_train_dataset=feverDataset["train"].select(range(2000))
fever_validation_dataset=feverDataset["validation"].select(range(2000)) # will use to evaluate model
fever_test_dataset=feverDataset["test"].select(range(2000)) #will use to test the model later

supports_count=fever_train_dataset["label"].count(0)
refutes_count=fever_train_dataset["label"].count(1)
nei_count=fever_train_dataset["label"].count(2)
fever_train_dataset=fever_train_dataset.select_columns(["input_ids", "attention_mask", "label"])
class_weights=torch.tensor([1.0/supports_count, 1.0/refutes_count, 1.0/nei_count])  

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
        #To prevent overfitting we will drop some neurons during training.
        self.dropout= nn.Dropout(0.3)
    
    def forward(self,input_ids,attention_mask,task,labels=None):
        outputs=self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_outputs= outputs.last_hidden_state[:,0,:]
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
        if labels is not None:
            loss= loss_fn(logits,labels)
        return {"loss": loss, "logits": logits}
    


#We will now load our verison of the model
model= two_TaskModel("roberta-base")

"""The following section would be about the training arguments and their trainer.
We would be training the model sequentially. So, first train on task 1 and then on task 2."""
#task1 Arguments
news_training_args= TrainingArguments(
    output_dir="Two_Task_Model_1_news",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4, #Changed because it doesn't require a large number for a small portion
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
)
#task2 Arguments
evidence_training_args=TrainingArguments(
    "Two_Task_Model_1_evidence",
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
#news trainer
news_trainer=Trainer(
    model=model,
    args=news_training_args,
    train_dataset=train_token,
    eval_dataset=eval_token,
    processing_class=news_tokenizer,
    compute_metrics=news_compute_metrics,
)
news_trainer.train()

class myTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels=inputs.get("labels")
        outputs=model(**inputs)        
        logits=outputs.logits
        if labels is not None:
            cross_func = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = cross_func(logits, labels)
            if loss.dim() != 0:
                loss = loss.mean()
        else:
            loss = torch.tensor(0.0, device=logits.device)
        return (loss, outputs) if return_outputs else loss
    
evidence_trainer = myTrainer(
    model,
    evidence_training_args,
    train_dataset=fever_train_dataset,
    eval_dataset=fever_validation_dataset,
    tokenizer=evidence_tokenization,
    compute_metrics=evidence_compute_metrics,
)

evidence_trainer()
"""Now that we have trained the model, we will begin to save. We can't save it the regular way, so
we have to use the pytorch checkpoint save
REMINDER: In order to use the model for a test or anywhere else, you have to replicate the architecture of the model
Which is class above."""
torch.save(model.state_dict(),"TwoTask_Model_1.pt")
news_tokenizer.save_pretrained("news_tokenizer/")
evidence_tokenization.save_pretrained("evidence/")
news_training_args.save_pretrained("news_training_args.bin")
evidence_training_args.save_pretrained("evidence_training_args.bin")




