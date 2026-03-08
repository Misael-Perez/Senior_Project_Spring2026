import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
print(os.listdir())

Wel_dataset= pd.read_csv("WEL_Dataset.csv")
Wel_dataset= Wel_dataset.dropna(subset=["title", "text", "label"])
Wel_PT1,remain= train_test_split(
    Wel_dataset,
    test_size=0.66,
    stratify=Wel_dataset["label"],
    random_state=42
)
Wel_PT2,Wel_PT3= train_test_split(
    remain,
    test_size=0.5,
    stratify=remain["label"],
    random_state=42
)
Wel_PT1.to_csv("Wel_PT1.csv", index=False)
Wel_PT2.to_csv("Wel_PT2.csv", index=False)
Wel_PT3.to_csv("Wel_PT3.csv",index=False)

