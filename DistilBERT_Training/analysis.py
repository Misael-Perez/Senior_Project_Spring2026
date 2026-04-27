from transformers import (AutoTokenizer, AutoModel,AutoModelForSequenceClassification,
    Trainer, TrainingArguments)
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
import torch
#device = torch.device("cpu") #you can change to GPU if you want
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#make sure cpu only
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
from collections import Counter

import pandas as pd
from datasets import Dataset
import evaluate
from sklearn.metrics import confusion_matrix
from datasets import load_dataset
import numpy as np
import re
import torch.nn.functional as F

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation
    return text
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("TwoTask_single_session_FT2_CPU.pt", map_location=device)

class_weights = checkpoint["class_weights"].to(device)
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
        #this is different from the actual training architecture
        if (task==0).all():
            logits= self.real_or_fake(cls_output)
            loss_fn= nn.CrossEntropyLoss()
        elif (task==1).all():
            logits = self.evidence_based(cls_output)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError("Please select the right task")
        loss=None
        if labels is not None:
            loss= loss_fn(logits,labels)
        
        return {"loss": loss, "logits": logits}



model = two_TaskModel("distilbert-base-uncased")
model.load_state_dict(checkpoint["Two_task_single_session"])

model.to(device)
model.eval()
for param in model.parameters():
    param.data = param.data.to(device)
    model.encoder = model.encoder.to(device)
model.real_or_fake = model.real_or_fake.to(device)
model.evidence_based = model.evidence_based.to(device)
model.dropout = model.dropout.to(device)
tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")
def news_tokenizer(dataset):
    return tokenizer(
        dataset["title"], dataset["text"], truncation=True,padding="max_length", max_length=512, return_token_type_ids=False
    )
accuracy_metric= evaluate.load("accuracy")
precision_metrics= evaluate.load("precision")
recall_metrics= evaluate.load("recall")
f1_metric= evaluate.load("f1")
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

def add_task_news(data_set):
    data_set["task"]=0#News will be 0
    return data_set
training_args= TrainingArguments(
    output_dir="./Two_head_results", #Changed because it doesn't require a large number for a small portion
    per_device_eval_batch_size=16,

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
trainer=Trainer(
    model=model,
    args=training_args,
    processing_class=news_tokenizer,
    compute_metrics=news_compute_metrics,
)

def get_matrix(trainer, dataset):
    predictions = trainer.predict(dataset)

    logits = predictions.predictions
    pred = np.argmax(logits, axis=1)
    labels = predictions.label_ids

    matrix = confusion_matrix(labels, pred)

    print("\nConfusion Matrix:")
    print(matrix)

    return logits, pred, labels

def get_prediction_indices(pred, labels):
    correct = np.where(pred == labels)[0]
    wrong = np.where(pred != labels)[0]

    fake_preds = np.where(pred == 0)[0]
    real_preds = np.where(pred == 1)[0]

    return correct, wrong, fake_preds, real_preds

def print_articles(dataset, indices, label_name, n=10):
    print(f"\nSample {label_name} articles:")
    for i in indices[:n]:
        text = tokenizer.decode(dataset[i]["input_ids"], skip_special_tokens=True)
        print(text[:500])
        print("---")
        
def show_uncertain_cases(logits, pred, labels, dataset, n=20):
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    confidence = np.max(probs, axis=1)

    uncertain_idx = np.argsort(confidence)[:n]

    print("\nMost uncertain predictions:")
    for i in uncertain_idx:
        text = tokenizer.decode(dataset[i]["input_ids"], skip_special_tokens=True)
        print(f"Pred: {pred[i]} Label: {labels[i]} Conf: {confidence[i]:.4f}")
        print(text[:500])
        print("------")
        

def word_importance(text, model, tokenizer, device):
    words = text.split()

    base_inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_token_type_ids=False
    )

    base_inputs = {
        "input_ids": base_inputs["input_ids"].to(device),
        "attention_mask": base_inputs["attention_mask"].to(device)
    }

    task = torch.zeros(base_inputs["input_ids"].size(0), dtype=torch.long).to(device)

    with torch.no_grad():
        base_logits = model(**base_inputs, task=task)

    base_pred = torch.argmax(base_logits["logits"], dim=1)
    base_prob = torch.softmax(base_logits["logits"], dim=1)[0, base_pred]

    importance = []

    for i in range(len(words)):
        new_text = " ".join(words[:i] + words[i+1:])

        inputs = tokenizer(
            new_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_token_type_ids=False
        )

        inputs = {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device)
        }

        task = torch.zeros(inputs["input_ids"].size(0), dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(**inputs, task=task)

        prob = torch.softmax(logits["logits"], dim=1)[0, base_pred]
        drop = base_prob - prob

        importance.append((words[i], drop.item()))

    return sorted(importance, key=lambda x: x[1], reverse=True)

def analyze_wrong_predictions(dataset, wrong_idx, pred, labels, model, tokenizer, device, n=5):
    print("\nWord Importance Analysis:")

    for i in wrong_idx[:n]:
        text = tokenizer.decode(dataset[i]["input_ids"], skip_special_tokens=True)

        print(f"\nPrediction: {pred[i]} Label: {labels[i]}")

        important_words = word_importance(text, model, tokenizer, device)
        print("Top important words:", important_words[:10])

        print(text[:500])
        print("------")
        
        
def global_analysis(dataset, pred):
    texts = [
        tokenizer.decode(dataset[i]["input_ids"], skip_special_tokens=True)
        for i in range(len(pred))
    ]

    lengths = [len(t) for t in texts]

    fake_lengths = [lengths[i] for i in range(len(pred)) if pred[i] == 0]
    real_lengths = [lengths[i] for i in range(len(pred)) if pred[i] == 1]

    print("\nAvg fake length:", np.mean(fake_lengths))
    print("Avg real length:", np.mean(real_lengths))

    fake_texts = [clean_text(texts[i]) for i in range(len(pred)) if pred[i] == 0]
    real_texts = [clean_text(texts[i]) for i in range(len(pred)) if pred[i] == 1]

    fake_words = Counter(" ".join(fake_texts).split()).most_common(20)
    real_words = Counter(" ".join(real_texts).split()).most_common(20)

    print("\nTop FAKE words:", fake_words)
    print("Top REAL words:", real_words)


WEL_3_data=pd.read_csv("WEL3/Wel_PT3.csv")
WEL_3_data["labels"] = 1 - WEL_3_data["labels"]
WEL_3_data=Dataset.from_pandas(WEL_3_data)
WEL_3_data= WEL_3_data.map(news_tokenizer, batched=True)
WEL_3_data=WEL_3_data.map(add_task_news)
WEL_3_data = WEL_3_data.remove_columns(["title","text"])
WEL_3_data.set_format("torch")
"""
test_data=pd.read_csv("test.csv")
test_data=Dataset.from_pandas(test_data)
test_token= test_data.map(news_tokenizer, batched=True)
test_token = test_token.remove_columns(["title","text"])
test_token = test_token.map(add_task_news)
test_token.set_format("torch")

print("The Matrix test.csv")
predictions= trainer.predict(test_token)
pred= np.argmax(predictions.predictions, axis=1)
labels= predictions.label_ids
final_matrix= confusion_matrix(labels,pred)
print(final_matrix)
"""
#now to analyze the data.

logits, pred, labels=get_matrix(trainer,WEL_3_data)

#From this point on, article analysis

correct, wrong, fake_preds, real_preds = get_prediction_indices(pred, labels)
print_articles(WEL_3_data,fake_preds,"FAKE")
print_articles(WEL_3_data,real_preds,"REAL")

show_uncertain_cases(logits, pred, labels, WEL_3_data)

analyze_wrong_predictions(WEL_3_data, wrong, pred, labels, model, tokenizer, device)

global_analysis(WEL_3_data,pred)
    
















