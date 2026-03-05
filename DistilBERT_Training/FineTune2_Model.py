from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from transformers import DataCollatorWithPadding
import pandas as pd
import numpy as np
import evaluate
from transformers import pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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

train_dataset2=pd.read_csv("train_dataset2.csv")
test_dataset2=pd.read_csv("test_dataset2.csv")

Trained_model= AutoModelForSequenceClassification.from_pretrained("./Model_1")
New_tokenizer = AutoTokenizer.from_pretrained("./Model_1")
#New function for the new tokenizer
def preprocess_function(examples):
    return New_tokenizer(examples["text"], examples["title"], truncation=True, max_length=512)
#Turn the test_data into dataset
train_data=Dataset.from_pandas(train_dataset2)
test_data= Dataset.from_pandas(test_dataset2)
#Use our new version of the tokenizer to tokenize the test_data
train_token= train_data.map(preprocess_function, batched=True)
test_token= test_data.map(preprocess_function, batched=True)
#Our training arguments
training_args= TrainingArguments(
    output_dir="Model_2",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
trainer=Trainer(
    model=Trained_model,
    args=training_args,
    train_dataset=train_token,
    eval_dataset=train_token,
    processing_class=New_tokenizer,
    compute_metrics=compute_metrics,
)
training_metrics= trainer.train()
metrics= trainer.evaluate()
print("\nThe information below is the Old method that was used to find the metrics")
print("The trainig metrics\n", training_metrics.metrics)
print("The metrics\n",metrics)

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

eval_metrics= trainer.evaluate()
print("\n Validation Metrics")
print(eval_metrics)
Trained_model.save_pretrained("./Model_2")
New_tokenizer.save_pretrained("./Model_2")

#We will output the results of the confusion matrix and the metrics
cols=("Predicted Fake","Predicted Real")
rows=("Actual Fake", "Actual Real" )

fig,ax=plt.subplots()
ax.set_axis_off()
table_matrix= ax.table(cellText=final_matrix, colLabels=cols,rowLabels=rows, loc='center', cellLoc='center')
table_matrix.auto_set_font_size(False)
table_matrix.set_fontsize(10)
table_matrix.scale(1.2, 1.2)
plt.savefig("Model_2.png")

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
