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
checkpoint = torch.load("TwoTask_Model_1_full_SEQ_freeze.pt", map_location=device)

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



model = two_TaskModel("roberta-base")
model.load_state_dict(checkpoint["Two_task_Model_1"])

model.to(device)
model.eval()

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


print("The Matrix test.csv")
predictions= trainer.predict(test_token)
pred= np.argmax(predictions.predictions, axis=1)
labels= predictions.label_ids
final_matrix= confusion_matrix(labels,pred)
print(final_matrix)

WEL_3_data=pd.read_csv("WEL3/Wel_PT3.csv")
WEL_3_data["labels"] = 1 - WEL_3_data["labels"]
WEL_3_data=Dataset.from_pandas(WEL_3_data)
WEL_3_data= WEL_3_data.map(preprocess_function, batched=True)
WEL_3_data = WEL_3_data.remove_columns(["title","text"])
WEL_3_data.set_format("torch")

print("The Matrix Wel_PT3.csv")
predictions= trainer.predict(WEL_3_data)
pred= np.argmax(predictions.predictions, axis=1)
labels= predictions.label_ids
final_matrix= confusion_matrix(labels,pred)
print(final_matrix)