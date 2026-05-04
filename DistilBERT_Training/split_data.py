import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

def remove_artifacts(article):
    if pd.isna(article):
        return ""
    article= article.replace("’", "'").replace("“", '"').replace("”", '"')
    article= re.sub(r"[\[\]\(\)]", "", article)
    article= re.sub(r"\s[-–—]\s", " ", article)
    article=re.sub(r"[!?.]{2,}", ".", article)
    article=re.sub(r"\s'\s", " ", article)
    article=re.sub(r"\s+", " ", article).strip()
    
    return article
    


#Start of the preprocessing
TrueData= pd.read_csv("News_dataset/True.csv")
FakeData=pd.read_csv("News_dataset/Fake.csv")
TrueData= TrueData.dropna()
FakeData= FakeData.dropna()


TrueData["text"] = TrueData["text"].apply(remove_artifacts)
FakeData["text"] = FakeData["text"].apply(remove_artifacts)

TrueData["title"] = TrueData["title"].apply(remove_artifacts)
FakeData["title"] = FakeData["title"].apply(remove_artifacts)

TrueData["labels"]=True
FakeData["labels"]=False
#Merge
officialTable= pd.concat([TrueData,FakeData])
officialTable=officialTable.reset_index(drop=True)
officialTable = officialTable.drop_duplicates(subset=["title","text"])
officialTable=officialTable.drop(columns=["subject","date"])

#We will shuffle the data and create separate dataframes
#More better splitting
train_data,remain= train_test_split(
    officialTable,
    test_size=0.33,
    stratify=officialTable["labels"],
    random_state=42
)
eval_data,test_data= train_test_split(
    remain,
    test_size=0.5,
    stratify=remain["labels"],
    random_state=42
)

train_data= train_data.reset_index(drop=True)
eval_data= eval_data.reset_index(drop=True)
test_data=test_data.reset_index(drop=True)

#"title" "date" "subject"
#Let's make the true and false numerical for the model
train_data["labels"]= train_data["labels"].astype(int)
eval_data["labels"]= eval_data["labels"].astype(int)
test_data["labels"]= test_data["labels"].astype(int)

#We will now turn the data into csv

train_data.to_csv("train.csv", index=False)
eval_data.to_csv("eval.csv", index=False)
test_data.to_csv("test.csv",index=False)
