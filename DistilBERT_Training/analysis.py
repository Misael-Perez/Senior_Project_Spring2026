from transformers import (AutoTokenizer, AutoModel,AutoModelForSequenceClassification,
    Trainer, TrainingArguments)
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
import torch
device = torch.device("cpu") #you can change to GPU if you want
#make sure cpu only
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from collections import Counter

import pandas as pd
from datasets import Dataset
import evaluate
from sklearn.metrics import confusion_matrix
from datasets import load_dataset
import numpy as np
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
predictions= trainer.predict(WEL_3_data)
pred= np.argmax(predictions.predictions, axis=1)
labels= predictions.label_ids
final_matrix= confusion_matrix(labels,pred)
print("\nThe Matrix on the WEL3 data")
print(final_matrix)
correct_id= np.where(pred==labels)[0]
wrong_answer=np.where(pred!=labels)[0]
"""
for i in wrong_answer[:10]:
    print("Prediction:", pred[i], "Label:", labels[i])
    decoded_text = tokenizer.decode(WEL_3_data[i]["input_ids"], skip_special_tokens=True)
    print(decoded_text)
    print("------")
"""



# For FAKE class (assuming label 0 = Fake)
precision_fake = precision_score(labels, pred, pos_label=0)
recall_fake = recall_score(labels, pred, pos_label=0)

# For REAL class (label 1)
precision_real = precision_score(labels, pred, pos_label=1)
recall_real = recall_score(labels, pred, pos_label=1)
print(precision_fake)
print(recall_fake)
print(precision_real)
print(recall_real)

print(np.bincount(pred))

TP_fake = np.sum((pred == 0) & (labels == 0))
FN_fake = np.sum((pred == 1) & (labels == 0))

TP_real = np.sum((pred == 1) & (labels == 1))
FN_real = np.sum((pred == 0) & (labels == 1))
print(TP_fake)
print(FN_fake)
print(TP_real)
print(FN_real)

#From this point on, article analysis
fake_preds = np.where(pred == 0)[0]
real_preds = np.where(pred == 1)[0]
for i in fake_preds[:10]:
    text = tokenizer.decode(WEL_3_data[i]["input_ids"], skip_special_tokens=True)
    print("Predicted FAKE:\n", text, "\n---")

for i in real_preds[:10]:
    text = tokenizer.decode(WEL_3_data[i]["input_ids"], skip_special_tokens=True)
    print("Predicted REAL:\n", text, "\n---")
    import torch.nn.functional as F

logits = predictions.predictions
probs = F.softmax(torch.tensor(logits), dim=1).numpy()

confidence = np.max(probs, axis=1)

uncertain_idx = np.argsort(confidence)[:20]
for i in uncertain_idx:
    print("Pred:", pred[i], "Label:", labels[i], "Conf:", confidence[i])
    text = tokenizer.decode(WEL_3_data[i]["input_ids"], skip_special_tokens=True)
    print(text)
    print("------")
    
def word_importance(text):
    words = text.split()
    base_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    base_logits = model(**base_inputs, task=torch.tensor([0]))
    base_pred = torch.argmax(base_logits["logits"], dim=1)

    importance = []

    for i in range(len(words)):
        new_text = " ".join(words[:i] + words[i+1:])
        inputs = tokenizer(new_text, return_tensors="pt", truncation=True, max_length=512,return_token_type_ids=False)
        logits = model(**inputs, task=torch.tensor([0]))
        pred = torch.argmax(logits["logits"], dim=1)

        if pred != base_pred:
            importance.append(words[i])

    return importance
print("Analyze words")
for i in wrong_answer[:5]:
    text = tokenizer.decode(WEL_3_data[i]["input_ids"], skip_special_tokens=True,return_token_type_ids=False)
    
    print("Prediction:", pred[i], "Label:", labels[i])
    
    important_words = word_importance(text)
    
    print("Important words:", important_words[:10])
    print(text[:500])  # don’t print everything
    print("------")

lengths = [len(tokenizer.decode(x["input_ids"])) for x in WEL_3_data]
fake_lengths = [lengths[i] for i in range(len(pred)) if pred[i]==0]
real_lengths = [lengths[i] for i in range(len(pred)) if pred[i]==1]

print("Avg fake length:", np.mean(fake_lengths))
print("Avg real length:", np.mean(real_lengths))


fake_texts = [tokenizer.decode(WEL_3_data[i]["input_ids"]) for i in range(len(pred)) if pred[i]==0]
real_texts = [tokenizer.decode(WEL_3_data[i]["input_ids"]) for i in range(len(pred)) if pred[i]==1]

fake_words = Counter(" ".join(fake_texts).split()).most_common(20)
real_words = Counter(" ".join(real_texts).split()).most_common(20)
print("Top FAKE words:", fake_words)
print("Top REAL words:", real_words)

fake_lengths = [lengths[i] for i in range(len(pred)) if pred[i]==0]
real_lengths = [lengths[i] for i in range(len(pred)) if pred[i]==1]

print("Avg fake length:", np.mean(fake_lengths))
print("Avg real length:", np.mean(real_lengths))
