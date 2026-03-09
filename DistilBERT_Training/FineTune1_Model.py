import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

"""
This file will be an updated version of the news_test.py. Only the important parts of the code


"""
#Let's load our new datasets for the training
train_data= pd.read_csv("train.csv")
eval_data=pd.read_csv("eval.csv")
test_data=pd.read_csv("test.csv")
#see the balance of fake and real articles
print(train_data["labels"].value_counts())


#We want to create tokens for our text
tokenizer= AutoTokenizer.from_pretrained("distilbert-base-uncased")

#Using the code from huggingface
#we want to get the title and the text
def preprocess_function(examples):
    return tokenizer(examples["title"], examples["text"], truncation=True, max_length=512)

#Let's limit the number of rows. Might increase
#train_data= train_data[:3000]
#eval_data = eval_data[:3000]

#There is a problem, the model and the tokens can only be used as a hugging face dataset
#let's convert it.
train_data= Dataset.from_pandas(train_data)
eval_data= Dataset.from_pandas(eval_data)
test_data=Dataset.from_pandas(test_data)

#tokenize every text for both datasets
train_token=train_data.map(preprocess_function, batched=True)
eval_token= eval_data.map(preprocess_function, batched=True)
test_token= test_data.map(preprocess_function, batched=True)
#remove them
train_token = train_token.remove_columns(["title","text"])
eval_token = eval_token.remove_columns(["title","text"])
test_token = test_token.remove_columns(["title","text"])
#set them to torch format
train_token.set_format("torch")
eval_token.set_format("torch")
test_token.set_format("torch")

#this is like a auto detect that will detect the largest length needed and apply it.
# This is not needed no more data_collator= DataCollatorWithPadding(tokenizer=tokenizer)
#The code below will allow us to view the metrics of the model
#We will update the compute_metrics() function to make a better statistics analysis
#we want to investigate how often does it spot fake articles. How correct is it?
accuracy_metrics= evaluate.load("accuracy")
precision_metrics= evaluate.load("precision")
recall_metrics= evaluate.load("recall")
f1_metrics= evaluate.load("f1")

def compute_metrics(eval_pred):
    logits,labels=eval_pred
    preds= np.argmax(logits,axis=1)
    
    accuracy= accuracy_metrics.compute(predictions=preds, references=labels)["accuracy"]
    precision= precision_metrics.compute(predictions=preds, references=labels, average="binary")["precision"]
    recall= recall_metrics.compute(predictions=preds, references=labels, average="binary")["recall"]
    f1= f1_metrics.compute(predictions=preds,references=labels, average="binary")["f1"]
    
    return {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1":f1
    }
#This will allow us to classify what is good and bad
id2label= {0: "Fake", 1:"Real"}
label2id={"Fake":0, "Real":1}
#Our model
model=AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",num_labels=2,id2label=id2label, label2id=label2id
)
#Traing Arguments: What to set
#The recommended epoch for sentient analysis is 3
training_args= TrainingArguments(
    output_dir="Model_1",
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
#What we will use for the training.
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_token,
    eval_dataset=eval_token,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)
training_metrics=trainer.train()
#We will use the test.csv here for predictions, Although we will also
#be used for further finetuning on Model.py
#We will be adding a confusion matrix to see the true positive, true negative,
#fake positive, and fake negative. Plus for some visual aid.
predictions= trainer.predict(test_token)
pred= np.argmax(predictions.predictions, axis=1)
labels= predictions.label_ids

final_matrix= confusion_matrix(labels,pred)
print("\nThe Matrix below is the confusion matrix on the Test data")
print(final_matrix)
#The output should be in the format 2x2 matrix

test_metrics= trainer.evaluate(test_token)

print("\nThe Test metrics\n", test_metrics)

eval_metrics= trainer.evaluate(eval_token)
print("\n Validation Metrics")
print(eval_metrics)
model.save_pretrained("./Model_1")
tokenizer.save_pretrained("./Model_1")

#We will output the results of the confusion matrix and the metrics
cols=("Predicted Fake","Predicted Real")
rows=("Actual Fake", "Actual Real" )

fig,ax=plt.subplots()
ax.set_axis_off()
table_matrix= ax.table(cellText=final_matrix, colLabels=cols,rowLabels=rows, loc='center', cellLoc='center')
table_matrix.auto_set_font_size(False)
table_matrix.set_fontsize(10)
table_matrix.scale(1.2, 1.2)
plt.savefig("Model_1.png")

#We will now save the results of the metrics into statistics.txt
text="Results of the Training (Model_1)"
text2="Results of the Validation (Model_1)"
text3="Results of the Test (Model_1)"
file_path="statistics.txt"
 
with open(file_path,'w') as file:
    file.write('\n')
    file.write(text)
    file.write(str(training_metrics.metrics))
    file.write("\n")
    file.write(text2 + "\n")
    file.write(f"Accuracy: {eval_metrics['eval_accuracy']}\n")
    file.write(f"Precision: {eval_metrics['eval_precision']}\n")
    file.write(f"Recall: {eval_metrics['eval_recall']}\n")
    file.write(f"F1: {eval_metrics['eval_f1']}\n\n")
    file.write(text3)
    file.write(f"Accuracy: {test_metrics['eval_accuracy']}\n")
    file.write(f"Precision: {test_metrics['eval_precision']}\n")
    file.write(f"Recall: {test_metrics['eval_recall']}\n")
    file.write(f"F1: {test_metrics['eval_f1']}\n")
print("Information has been saved")
