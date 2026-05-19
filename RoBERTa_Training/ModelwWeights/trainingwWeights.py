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
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
import json
from sklearn.metrics import confusion_matrix

checkpoint="roberta-base"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

feverDataset=load_dataset("copenlu/fever_gold_evidence")

#print(feverDataset)
#print(fever_train_dataset.features)

label_map={
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2
}

id2label= {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT ENOUGH INFO"
}

label2id= {v: k for k, v in id2label.items()}

def map_labels(labels):
    return {"label": [label_map[m] for m in labels["label"]]}

def tokenization(batch):
    evidences=[]
    for i in range(len(batch["claim"])):
        e=batch["evidence"][i]
        if (e) and len(e[0])>2:
            evidences.append(e[0][2])
        else:
            evidences.append("")

    return tokenizer(
        batch["claim"],
        evidences,
        truncation=True,
        padding="max_length",
        max_length=256
    )

feverDataset=feverDataset.map(map_labels, batched=True)
feverDataset=feverDataset.map(tokenization, batched=True)
feverDataset=feverDataset.select_columns(["input_ids", "attention_mask", "label"])
feverDataset=feverDataset.rename_column("label", "labels")

fever_train_dataset=feverDataset["train"]
fever_validation_dataset=feverDataset["validation"] # will use to evaluate model
fever_test_dataset=feverDataset["test"] #will use to test the model later

#Fine-Tuning
model=AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
        )

accuracy_metric=evaluate.load("accuracy")
f1_metric= evaluate.load("f1")

def compute_metrics(eval_preds):
    logits, labels=eval_preds
    predicts=np.argmax(logits, axis=-1)
    acc=accuracy_metric.compute(predictions=predicts, references=labels)["accuracy"]
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
    "test9wWeights",
    eval_strategy="steps",
    eval_steps=1000,
    learning_rate=2e-5,                 #only chnage when loss seems to chnage weirdly or stays the same
    per_device_train_batch_size=32,
    #gradient_accumulation_steps=8,
    per_device_eval_batch_size=256,     #GPU change:increase to 256
    num_train_epochs=3,                 #GPU change: increase to 5
    weight_decay=0.01,
    logging_steps=500,   #should probbaly decrease this                   #228 per epoch but no real time logging
    save_strategy="steps",
    save_steps=1000,                      #GPU chnage: idecrease to 25
    #eval_steps #not needed for epoch
    #warmup_steps=50,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    max_grad_norm=1.0
)

labels= np.array(fever_train_dataset["labels"])
counts= np.bincount(labels, minlength=3)
weights= 1.0/ np.sqrt(counts+1e-6)
weights= weights/weights.min()
class_weights= torch.tensor(weights, dtype=torch.float)

class myTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels=inputs["labels"]
        outputs=model(**inputs)         #model() runs a forward pass so were returns logits, loss(only if labels was passed, hidden_states, and attentions)
        logits=outputs.logits
        #print("labels: ", labels.shape, "logits:", logits.shape)
        cross_func=nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss=cross_func(logits, labels)
        return (loss, outputs) if return_outputs else loss

trainer = myTrainer(
    model,
    training_args,
    train_dataset=fever_train_dataset,
    eval_dataset=fever_validation_dataset,
    processing_class=tokenizer,
    # data_collator=collator, #redundant
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("Model_9")
trainer.save_model("Model_9")
tokenizer.save_pretrained("Tokenizer_9")

#Test with test split
testPredictions=trainer.predict(fever_test_dataset)
actual=testPredictions.label_ids
preds=np.argmax(testPredictions.predictions, axis=1)


print("***TEST PREDICTIONS***")
print(dir(testPredictions))
print(testPredictions.label_ids)
print("---------------------")
print(testPredictions.predictions)
print("---------------------")
print(testPredictions.metrics)
# labels of testPredictions: 'count', 'index', 'label_ids', 'metrics', 'predictions'

confusionMatrix=confusion_matrix(actual, preds)
columnNames = [id2label[0], id2label[1], id2label[2]]
df_confusionMatrix= pd.DataFrame(
    confusionMatrix,
    index=columnNames,
    columns=columnNames
)
print("***CONFUSION MATRIX***")
print(df_confusionMatrix)


#df=pd.DataFrame({
#    "actual": actual,
#    "prediction": preds,
#    "result": (actual==preds)
#    })
#df.to_csv("TestResults_8.csv", index=False)
#print("METRICS: ", testPredictions.metrics)
#print(df.head())

with open('metrics_Model9.txt', 'w') as f:
    f.write(json.dumps(testPredictions.metrics, indent=4))



