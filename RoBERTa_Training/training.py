import torch
from transformers import (
    AutoTokenizer,
    #RobertaForSequenceClassification,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    #DataCollatorWithPadding
)
from datasets import load_dataset, load_dataset_builder
import evaluate
import numpy as np
import pandas as pd

checkpoint="roberta-base"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

feverDataset=load_dataset("copenlu/fever_gold_evidence")

print(feverDataset)
#print(fever_train_dataset.features)

label_map={
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2
}

def map_labels(labels):
    return {"label": [label_map[m] for m in labels["label"]]}

def tokenization(claims):
    evidences=[]
    for e in claims["evidence"]:
        if len(e)>0:
            evidences.append(claims[0][2])      #fix later: account for multiple evidence senetnces.
        else:
            evidences.append("")
    return tokenizer(
        claims["claim"],
        evidences,
        truncation=True,
        padding="max_length",
        max_length=512
    )
#num_proc?

feverDataset=feverDataset.map(tokenization, batched=True)
feverDataset=feverDataset.map(map_labels, batched=True)

fever_train_dataset=feverDataset["train"]
fever_validation_dataset=feverDataset["validation"] # # will use to evaluate model
fever_test_dataset=feverDataset["test"] #will use to test the model later


fever_train_dataset=fever_train_dataset.select_columns(["input_ids", "attention_mask", "label"])
#collator=DataCollatorWithPadding(tokenizer=tokenizer)

#Fine-Tuning
model=AutoModelForSequenceClassification.from_pretrained(
    checkpoint, 
    num_labels=3
)

accuracy_metric=evaluate.load("accuracy")
f1_metric= evaluate.load("f1")

def compute_metrics(eval_preds):
    logits, labels=eval_preds
    predicts=np.argmax(logits, axis=-1)
    acc=accuracy_metric.compute(predictions=predicts, references=labels)
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


training_args=TrainingArguments(
    "test7",
    eval_strategy="steps",
    eval_steps=1000,
    learning_rate=2e-5,                 #only chnage when loss seems to chnage weirdly or stays the same
    per_device_train_batch_size=32,
    #gradient_accumulation_steps=8,
    per_device_eval_batch_size=256,     #GPU change:increase to 256
    num_train_epochs=3,                 #GPU change: increase to 5
    #weight_decay=0.01,
    logging_steps=500,   #should probbaly decrease this                   #228 per epoch but no real time logging
    save_strategy="steps",
    save_steps=10000,                      #GPU chnage: idecrease to 25
    #eval_steps #not needed for epoch
    warmup_steps=50

)

#add later: class weights for NEI
trainer = Trainer(
    model,
    training_args,
    train_dataset=fever_train_dataset,
    eval_dataset=fever_validation_dataset,
    processing_class=tokenizer,
    # data_collator=collator, #redundant
    compute_metrics=compute_metrics,
)


trainer.train()

trainer.save_model("Model_7")
tokenizer.save_pretrained("Tokenizer_7")
#we will test next with test split
testPredictions=trainer.predict(fever_test_dataset)
print(dir(testPredictions))
print(testPredictions.label_ids)
print("---------------------")
print(testPredictions.predictions)
print("---------------------")
print(testPredictions.metrics)
# labels of testPredictions: 'count', 'index', 'label_ids', 'metrics', 'predictions'

actual=testPredictions.label_ids
preds=np.argmax(testPredictions.predictions, axis=1)

df=pd.DataFrame({
    "actual": actual,
    "prediction": preds,
    "result": (actual==preds)

    })
df.to_csv("TestResults_2.csv", index=False)
print("METRICS: ", testPredictions.metrics)
print(df.head())
    
#we will test next with test split

#test_metrics=trainer.evaluate(fever_test_dataset)
#finalPredictions=trainer.predict(fever_test_dataset)