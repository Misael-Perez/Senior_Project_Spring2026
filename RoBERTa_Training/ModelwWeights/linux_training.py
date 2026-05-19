import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, load_dataset_builder
import evaluate
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
import json

checkpoint="roberta-base"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

feverDataset=load_dataset("copenlu/fever_gold_evidence")

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
            text=" ".join([ev[2] for ev in e])
        else:
            text=""
        evidences.append(text)

    return tokenizer(
        claims["claim"],
        evidences,
        truncation=True,
        padding="max_length",
        max_length=512
    )

feverDataset=feverDataset.map(tokenization, batched=True)
feverDataset=feverDataset.map(map_labels, batched=True)

fever_train_dataset=feverDataset["train"]
fever_train_dataset = fever_train_dataset.rename_column("label", "labels")
fever_validation_dataset=feverDataset["validation"] # will use to evaluate model
fever_test_dataset=feverDataset["test"] #will use to test the model later

supports_count=fever_train_dataset["labels"].count(0)
refutes_count=fever_train_dataset["labels"].count(1)
nei_count=fever_train_dataset["labels"].count(2)


fever_train_dataset=fever_train_dataset.select_columns(["input_ids", "attention_mask", "labels"])

#Fine-Tuning
model=AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=3,
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
    "test8wWeights",
    eval_strategy="steps",
    eval_steps=1000,
    learning_rate=2e-5,                 
    per_device_train_batch_size=32,
    per_device_eval_batch_size=256,     
    num_train_epochs=3,                 
    weight_decay=0.01,
    logging_steps=500,   
    save_strategy="steps",
    save_steps=1000,                     
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True
)

class_weights=torch.tensor([1.0/supports_count, 1.0/refutes_count, 1.0/nei_count])       #used inverse frequency
class myTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels=inputs.get("labels")
        outputs=model(**inputs)        
        logits=outputs.logits
        if labels is not None:
            cross_func = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = cross_func(logits, labels)
            if loss.dim() != 0:
                loss = loss.mean()
        else:
            loss = torch.tensor(0.0, device=logits.device)
        return (loss, outputs) if return_outputs else loss

trainer = myTrainer(
    model,
    training_args,
    train_dataset=fever_train_dataset,
    eval_dataset=fever_validation_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("Model_8")
tokenizer.save_pretrained("Tokenizer_8")

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
df.to_csv("TestResults_8.csv", index=False)
print("METRICS: ", testPredictions.metrics)
#print(df.head())

with open('metrics.txt', 'w') as f:
    f.write(json.dumps(testPredictions.metrics, indent=4))

#Time took to train: ~8 hours
                                                                                                                                                                    