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
    return tokenizer(
        claims["claim"],
        truncation=True,
        #padding="max_length",
        #max_length=512
    )
#num_proc?

feverDataset=feverDataset.map(tokenization, batched=True)
feverDataset=feverDataset.map(map_labels, batched=True)

fever_train_dataset=feverDataset["train"]
fever_validation_dataset=feverDataset["validation"] # will use to idk
fever_test_dataset=feverDataset["test"] #will use to test the model later


fever_train_dataset=fever_train_dataset.select_columns(["input_ids", "attention_mask", "label"])
#collator=DataCollatorWithPadding(tokenizer=tokenizer)

#Fine-Tuning
model=AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

accuracy_metric=evaluate.load("accuracy")
f1_metric= evaluate.load("f1")

def compute_metrics(eval_preds):
    logits, labels=eval_preds
    predicts=np.argmax(logits, axis=-1)
    acc=accuracy_metric.compute(predictions=predicts, references=labels)
    f1=f1_metric.compute(predictions=predicts, references=labels, average="macro" )    #macro since there are way more "SUPPORTS" then the other 2 labels
    return {**acc, **f1}


training_args=TrainingArguments(
    "test2",
    eval_strategy="epoch",
    learning_rate=2e-5,                 #only chnage when loss seems to chnage weirdly or stays the same
    per_device_train_batch_size=16,   
    gradient_accumulation_steps=62,
    per_device_eval_batch_size=128,     #GPU change:increase to 256
    num_train_epochs=3,                 #GPU change: increase to 5
    #weight_decay=0.01,
    logging_steps=50,   #should probbaly decrease this                   #228 per epoch but no real time logging
    save_strategy="steps",
    save_steps=50,                      #GPU chnage: idecrease to 25
    #eval_steps #not needed for epoch
)

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
    
#we will test next with test split

#test_metrics=trainer.evaluate(fever_test_dataset)
#finalPredictions=trainer.predict(fever_test_dataset)