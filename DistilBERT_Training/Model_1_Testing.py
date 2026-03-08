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
