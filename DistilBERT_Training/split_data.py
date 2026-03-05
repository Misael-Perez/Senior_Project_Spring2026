import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



#Start of the preprocessing
TrueData= pd.read_csv("News_dataset/True.csv")
FakeData=pd.read_csv("News_dataset/Fake.csv")
TrueData= TrueData.dropna()
FakeData= FakeData.dropna()
TrueData["label"]=True
FakeData["label"]=False

#Merge
officialTable= pd.concat([TrueData,FakeData])
officialTable=officialTable.reset_index(drop=True)
#We will shuffle the data and create separate dataframes
#More better splitting
train_data,remain= train_test_split(
    officialTable,
    test_size=0.33,
    stratify=officialTable["label"],
    random_state=42
)
eval_data,test_data= train_test_split(
    remain,
    test_size=0.5,
    stratify=remain["label"],
    random_state=42
)

train_data= train_data.reset_index(drop=True)
eval_data= eval_data.reset_index(drop=True)
test_data=test_data.reset_index(drop=True)

#"title" "date" "subject"
#Let's make the true and false numerical for the model
train_data["label"]= train_data["label"].astype(int)
eval_data["label"]= eval_data["label"].astype(int)
test_data["label"]= test_data["label"].astype(int)

#We will now turn the data into csv

train_data.to_csv("train.csv", index=False)
eval_data.to_csv("eval.csv", index=False)
test_data.to_csv("test.csv",index=False)
