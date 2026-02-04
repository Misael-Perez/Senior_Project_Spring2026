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
TrueData= pd.read_csv("True.csv")
FakeData=pd.read_csv("Fake.csv")
TrueData= TrueData.dropna()
FakeData= FakeData.dropna()
TrueData["Verify"]=True
FakeData["Verify"]=False
#If you want to save some texts for testing, put the code below

#end
#Merge
officialTable= pd.concat([TrueData,FakeData])
officialTable=officialTable.reset_index(drop=True)