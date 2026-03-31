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
checkpoint = torch.load("TwoTask_single_session.pt", map_location=device)

class_weights = checkpoint["class_weights"].to(device)
checkpoint="roberta-base"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
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
        if (task==0).all():
            logits= self.real_or_fake(cls_output)
            loss_fn= nn.CrossEntropyLoss()
        elif (task==1).all():
            logits = self.evidence_based(cls_output)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError("Please select the right task")
        if labels is not None:
            loss= loss_fn(logits,labels)
        
        return {"loss": loss, "logits": logits}
    
model = two_TaskModel("roberta-base")
model.load_state_dict(checkpoint["Two_task_single_session"])

model.to(device)
model.eval()

news_tokenizer = AutoTokenizer.from_pretrained("news_tokenizer/")
evidence_tokenizer= AutoTokenizer.from_pretrained("evidence/")
def news_function(examples):
    return news_tokenizer(examples["title"], examples["text"], truncation=True, max_length=512)

def evidence_tokenization(dataset):
    evidences=[]
    for e in dataset["evidence"]:
        if len(e)>0:
            text=" ".join([ev[2] for ev in e])
        else:
            text=""
        evidences.append(text)

    return evidence_tokenizer(
        dataset["claim"],
        evidences,
        truncation=True,
        padding="max_length",
        max_length=512
    )

label_map={
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2
}
id2label= {0: "Fake", 1:"Real"}
def map_labels(labels):
    return {"label": [label_map[m] for m in labels["label"]]}

#from this point, we are going to test on a different dataset
#Let's load our data
WEL_1_data= pd.read_csv("WEL1/Wel_PT1.csv")
WEL_2_data=pd.read_csv("WEL2/Wel_PT2.csv")
WEL_3_data=pd.read_csv("WEL3/Wel_PT3.csv")
#Invert the labels
WEL_1_data["labels"] = 1 - WEL_1_data["labels"]
WEL_2_data["labels"] = 1 - WEL_2_data["labels"]
WEL_3_data["labels"] = 1 - WEL_3_data["labels"]
Large_Data = pd.concat([WEL_1_data, WEL_2_data], ignore_index=True)

Large_Data=Dataset.from_pandas(Large_Data)
WEL_3_data=Dataset.from_pandas(WEL_3_data)


#tokenize every text for both datasets

WEL_3_data= WEL_3_data.map(news_function, batched=True)
Large_Data= Large_Data.map(news_function, batched=True)
def add_task_news(data_set):
    data_set["task"]=0#News will be 0
    return data_set
#we are going to add a task column for the dataset
Large_Data-Large_Data.map(add_task_news)
WEL_3_data=WEL_3_data.map(add_task_news)
#Remove all colmuns that will not be used
Large_Data=Large_Data.remove_columns(["title","text"])
WEL_3_data = WEL_3_data.remove_columns(["title","text"])
#set to torch
Large_Data.set_format("torch")
WEL_3_data.set_format("torch")

#loading the test dataset for the evidence head 
feverDataset=load_dataset("copenlu/fever_gold_evidence")
feverDataset=feverDataset.map(evidence_tokenization, batched=True)
feverDataset=feverDataset.map(map_labels, batched=True)
def add_task_evidence(data_set):
    data_set["task"]=1#evidence will be 1
    return data_set

fever_test_dataset=feverDataset["test"]#will use to test the model later
fever_test_dataset=fever_test_dataset.rename_column("label","labels")
fever_test_dataset=fever_test_dataset.map(add_task_evidence)
fever_test_dataset = fever_test_dataset.select_columns(
    ["input_ids", "attention_mask", "labels", "task"]
)

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
    train_dataset=Large_Data,
    eval_dataset=WEL_3_data,
    processing_class=tokenizer,
)
Trainer.train()
predictions= Trainer.predict(WEL_3_data)
pred= np.argmax(predictions.predictions, axis=1)
labels= predictions.label_ids

final_matrix= confusion_matrix(labels,pred)
print("\nThe Matrix below is the confusion matrix on the Test data")
print(final_matrix)
test_metrics= Trainer.evaluate(WEL_3_data)
print("\nThe Test metrics\n", test_metrics)

testPredictions=Trainer.predict(fever_test_dataset)
print(dir(testPredictions))
print(testPredictions.label_ids)
print("---------------------")
print(testPredictions.predictions)
print("---------------------")
print(testPredictions.metrics)

print("METRICS: ", testPredictions.metrics)

torch.save({
    "Two_task_single_session": model.state_dict(),
    "class_weights": class_weights,
    "label_map":label_map,
    "id2label":id2label
}, "TwoTask_single_session_FT2.pt")