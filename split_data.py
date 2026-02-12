import pandas as pd
import numpy as np



#Start of the preprocessing
TrueData= pd.read_csv("True.csv")
FakeData=pd.read_csv("Fake.csv")
TrueData= TrueData.dropna()
FakeData= FakeData.dropna()
TrueData["label"]=True
FakeData["label"]=False

#Merge
officialTable= pd.concat([TrueData,FakeData])
officialTable=officialTable.reset_index(drop=True)
#We will shuffle the data and create separate dataframes
shuffled= officialTable.sample(frac=1).reset_index(drop=True)
two_tables= np.array_split(shuffled,3)
train_data= pd.DataFrame(two_tables[0],columns=["title","text","subject","date","label"])
eval_data= pd.DataFrame(two_tables[1],columns=["title","text","subject","date","label"])
test_data= pd.DataFrame(two_tables[2],columns=["title","text","subject","date","label"])

train_data= train_data.reset_index(drop=True)
eval_data= eval_data.reset_index(drop=True)
test_data=test_data.reset_index(drop=True)

#"title" "date" "subject"
#Let's make the true and false numerical for the model
train_data["label"]= train_data["label"].astype(int)
eval_data["label"]= eval_data["label"].astype(int)
test_data["label"]= test_data["label"].astype(int)

#We will now turn the data into csv

train_data.to_csv("train.csv")
eval_data.to_csv("eval.csv")
test_data.to_csv("test.csv")