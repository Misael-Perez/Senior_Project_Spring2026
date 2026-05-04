import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
#Should be used in the GPU server
def remove_artifacts(article):
    if pd.isna(article):
        return " "
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
print("Finished Part 1")

WEL_1_data= pd.read_csv("WEL1/Wel_PT1.csv")
WEL_2_data=pd.read_csv("WEL2/Wel_PT2.csv")
WEL_3_data=pd.read_csv("WEL3/Wel_PT3.csv")
#invert to right labels
WEL_1_data["labels"] = 1 - WEL_1_data["labels"]
WEL_2_data["labels"] = 1 - WEL_2_data["labels"]
WEL_3_data["labels"] = 1 - WEL_3_data["labels"]


WEL_1_data["text"] = WEL_1_data["text"].apply(remove_artifacts)
WEL_2_data["text"] = WEL_2_data["text"].apply(remove_artifacts)
WEL_3_data["text"] = WEL_3_data["text"].apply(remove_artifacts)

WEL_1_data["title"] = WEL_1_data["title"].apply(remove_artifacts)
WEL_2_data["title"] = WEL_2_data["title"].apply(remove_artifacts)
WEL_3_data["title"] = WEL_3_data["title"].apply(remove_artifacts)

WEL_1_data.to_csv("Wel_PT1.csv", index=False)
WEL_2_data.to_csv("Wel_PT2.csv", index=False)
WEL_3_data.to_csv("Wel_PT3.csv",index=False)
print("Finished Part 2")