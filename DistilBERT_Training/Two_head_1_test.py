from transformers import (AutoTokenizer, AutoModel,AutoModelForSequenceClassification,
    Trainer, TrainingArguments)
import torch.nn as nn
import torch
import pandas as pd
from datasets import Dataset
import evaluate
from sklearn.metrics import confusion_matrix
from datasets import load_dataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


checkpoint="roberta-base"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
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
        padding=True,
        max_length=512
    )
def map_labels(labels):
    return {"label": [label_map[m] for m in labels["label"]]}
feverDataset=load_dataset("copenlu/fever_gold_evidence")
label_map={
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2
}
feverDataset=feverDataset.map(evidence_tokenization, batched=True)
feverDataset=feverDataset.map(map_labels, batched=True)
fever_train_dataset=feverDataset["train"].select(range(2000))
fever_validation_dataset=feverDataset["validation"].select(range(2000)) # will use to evaluate model
fever_test_dataset=feverDataset["test"].select(range(2000)) #will use to test the model later
supports_count=fever_train_dataset["label"].count(0)
refutes_count=fever_train_dataset["label"].count(1)
nei_count=fever_train_dataset["label"].count(2)
class_weights=torch.tensor([1.0/supports_count, 1.0/refutes_count, 1.0/nei_count])  
class_weights=class_weights.to(device)
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
        self.current_task=None
        #To prevent overfitting we will drop some neurons during training.
        self.dropout= nn.Dropout(0.3)
    
    def forward(self,input_ids,attention_mask,labels=None):
        outputs=self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_outputs= outputs.last_hidden_state[:,0,:]
        cls_output= self.dropout(cls_outputs)
        
        #The following lines of code will decide which head to use
        if self.current_task=="News":
            logits= self.real_or_fake(cls_output)
            loss_fn= nn.CrossEntropyLoss()
        elif self.current_task == "evidence":
            logits = self.evidence_based(cls_output)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError("Please select the right task")
        loss=None
        if labels is not None:
            loss= loss_fn(logits,labels)
        return {"loss": loss, "logits": logits}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = two_TaskModel("roberta-base")
model.load_state_dict(torch.load("TwoTask_Model_1.pt", map_location=device))
model.to(device)
model.eval()

from transformers import AutoTokenizer

news_tokenizer = AutoTokenizer.from_pretrained("news_tokenizer/")
evidence_tokenizer= AutoTokenizer.from_pretrained("evidence/")

    
test_data=pd.read_csv("test.csv")
test_data=Dataset.from_pandas(test_data)
def preprocess_function(examples):
    return news_tokenizer(examples["title"], examples["text"], truncation=True, max_length=512)

test_token= test_data.map(preprocess_function, batched=True)
test_token = test_token.remove_columns(["title","text"])
test_token.set_format("torch")


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
model.current_task = "News"
training_args= TrainingArguments(
    output_dir="./Two_Task_Model_1_results", #Changed because it doesn't require a large number for a small portion
    per_device_eval_batch_size=16,

)
trainer=Trainer(
    model=model,
    args=training_args,
    processing_class=news_tokenizer,
    compute_metrics=news_compute_metrics,
)

print("Evaluation on the first WEL data")
results1= trainer.evaluate(test_token)
print(results1)
print("The Matrix")
predictions= trainer.predict(test_token)
pred= np.argmax(predictions.predictions, axis=1)
labels= predictions.label_ids
final_matrix= confusion_matrix(labels,pred)
print(final_matrix)