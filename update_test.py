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
"""
This file will be an updated version of the news_test.py. Only the important parts of the code


"""
#Let's load our new datasets for the training
train_data= pd.read_csv("train.csv")
eval_data=pd.read_csv("eval.csv")


#We want to create tokens for our text
tokenizer= AutoTokenizer.from_pretrained("distilbert-base-uncased")

#Using the code from huggingface
#we want to get the title and the text
def preprocess_function(examples):
    return tokenizer(examples["text"], examples["title"], truncation=True, max_length=512)

#Let's limit the number of rows. Might increase
#train_data= train_data[:3000]
#eval_data = eval_data[:3000]

#There is a problem, the model and the tokens can only be used as a hugging face dataset
#let's convert it.
train_data= Dataset.from_pandas(train_data)
eval_data= Dataset.from_pandas(eval_data)

#tokenize every text for both datasets
train_token=train_data.map(preprocess_function, batched=True)
eval_token= eval_data.map(preprocess_function, batched=True)

#this is like a auto detect that will detect the largest length needed and apply it.
# This is not needed no more data_collator= DataCollatorWithPadding(tokenizer=tokenizer)
#The code below will allow us to view the metrics of the model
accuracy= evaluate.load("accuracy")
def computer_metrics(eval_pred):
    predictions,labels=eval_pred
    predictions= np.argmax(predictions,axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
#This will allow us to classify what is good and bad
id2label= {0: "NEGATIVE", 1:"POSITIVE"}
label2id={"NEGATIVE":0, "POSITIVE":1}
#Our model
model=AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",num_labels=2,id2label=id2label, label2id=label2id
)
#Traing Arguments: What to set
#The recommended epoch for sentient analysis is 3
training_args= TrainingArguments(
    output_dir="Test_Model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=62,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
#What we will use for the training.
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_token,
    eval_dataset=eval_token,
    processing_class=tokenizer,
    compute_metrics=computer_metrics,
)
training_metrics= trainer.train()
metrics= trainer.evaluate()
print("The trainig metrics", training_metrics.metrics)
print("The metrics\n",metrics)
model.save_pretrained("./Test_Model")
tokenizer.save_pretrained("./Test_Model")

#We are officially done with the original training
""" 
Things to note about finetuning the model
At 1000 rows it take about 40 second of training
At 2000 rows it was about 2 minutes

"""
# Split the dataset into 3 dataframes training, test, and evaluation.
# Goal of the project is to be above 21% accuracy (which is decent for a model. We want at least 80%)
#pytorch to be able to split the dataset for 90% to 10% for evaluation
#Goal for now: split the dataset into 3. Increase the nunber

"""
Notes:
the professor stated that the training dataset can be used for the evaluation for the.
But I can split the dataset into training, evaluation and test.
The test is what can be used to train
"""
