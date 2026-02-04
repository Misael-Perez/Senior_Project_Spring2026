from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from transformers import DataCollatorWithPadding
import pandas as pd
import numpy as np
import evaluate
#we want to clean the data and get rid of all null for both datasets

TrueData= pd.read_csv("True.csv")
FakeData=pd.read_csv("Fake.csv")
print(TrueData.head())
TrueData= TrueData.dropna()
FakeData= FakeData.dropna()
#Add a new columna for each dataset, so we can associate it with being true or false

TrueData["Verify"]=True
FakeData["Verify"]=False
#We want to test the model with an existing article that is True in TrueDATA
text= TrueData["text"].iloc[0]
print(text)
print(TrueData.head())
#let's make a new table with both of them merged
officialTable= pd.concat([TrueData,FakeData])
officialTable=officialTable.reset_index(drop=True)
#all the trues and false are grouped together. We want them to be separated so the model and have a chunk of each
shuffled= officialTable.sample(frac=1).reset_index(drop=True)
print(shuffled.head(20))
print(shuffled.info())
#Now, we want a data for training and the other for testing. So, we split the table into two
two_tables= np.array_split(shuffled,2)
train_data= two_tables[0]
test_data=two_tables[1]
train_data= train_data.reset_index(drop=True)
test_data=test_data.reset_index(drop=True)
print(train_data.head())
print(test_data.head())
#We now have our complete datasets for training and for testing.
#For now, we will just test the text column and the verify column. So, we will drop the rest.

#"title" "date" "subject"

train_data.drop(columns=["subject"],inplace=True)
test_data.drop(columns=["subject"],inplace=True)
#Let's make the true and false numerical for the model
train_data["Verify"]= train_data["Verify"].astype(int)
test_data["Verify"]= test_data["Verify"].astype(int)
#WARNING hugging face's distilbert ONLY looks for a column named "label"
train_data = train_data.rename(columns={"Verify": "label"})
test_data = test_data.rename(columns={"Verify": "label"})

print(train_data.head())
print(test_data.head())
#This is just for testing the model. Not the official method we will choose.
# we will now load the model, make tokens, and only trains for 2000 rows each.
#for making tokens of the text
tokenizer= AutoTokenizer.from_pretrained("distilbert-base-uncased")

def token(data):
    return tokenizer(data["text"],data["title"], truncation=True,padding="max_length", max_length=128)

#Let's limit the number of rows
train_data= train_data[:1000]
test_data = test_data[:1000]

#There is a problem, the model and the tokens can only be used as a hugging face dataset
#let's convert it.
train_data= Dataset.from_pandas(train_data)
test_data= Dataset.from_pandas(test_data)

train_ds=train_data.map(token, batched=True)
test_ds= test_data.map(token, batched=True)
train_ds.set_format("torch", columns=["input_ids","attention_mask","label"])
test_ds.set_format("torch", columns=["input_ids","attention_mask","label"])
#bring the model
#we have two classes= [True, False]. 
model=AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
accuracy= evaluate.load("accuracy")

#function to determine accuracy
def compute_metrics(eval_pred):
    predictions,labels=eval_pred
    predictions =np.argmax(predictions,axis=1)
    return accuracy.compute(predictions=predictions,references=labels)

#training arguments= meaning the information on the way we want our model to be

args=TrainingArguments(
    output_dir="./results",
    #evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    logging_steps=50,
    save_strategy="no",
    report_to="none"
)

trainer= Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=token,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

device=torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
model.to(device)
model.eval()

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
inputs= {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)

print(outputs.logits)
print(outputs.logits.argmax(dim=1))
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()

label_map = {
    0: "Fake",
    1: "Real"
}

print("Prediction:", label_map[predicted_class])
#get the accuracy and evaluation of the model.