import pandas as pd
import numpy as np


test_dataset2=pd.read_csv("News_dataset_2/test_dataset2.csv")
train_dataset2= pd.read_csv("News_dataset_2/train_dataset2.csv")
train_dataset2= train_dataset2.dropna()
test_dataset2=test_dataset2.dropna()
print(train_dataset2.head())
"""
The reason we are doing this is because in the documentation of the dataset, Fake is 1 and Real is 0.
For our dataset, Fake is 0 and Real 1.
"""
train_dataset2["label"]= train_dataset2["label"].replace({0:1,1:0})
print(train_dataset2.head())

train_dataset2.to_csv("train_dataset2.csv")
test_dataset2.to_csv("test_dataset2.csv")
