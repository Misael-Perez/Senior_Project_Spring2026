import pandas as pd
from sklearn.model_selection import train_test_split


Wel_dataset= pd.read_csv("WEL_Dataset.csv")
Wel_dataset= Wel_dataset.dropna(subset=["title", "text", "label"])
Wel_dataset=Wel_dataset.rename(columns={"label":"labels"})
Wel_dataset = Wel_dataset.drop_duplicates(subset=["text"])
Wel_dataset=Wel_dataset.drop_duplicates(subset=["title","text"])
Wel_dataset = Wel_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
Wel_PT1,remain= train_test_split(
    Wel_dataset,
    test_size=0.66,
    stratify=Wel_dataset["labels"],
    random_state=42
)
Wel_PT2,Wel_PT3= train_test_split(
    remain,
    test_size=0.5,
    stratify=remain["labels"],
    random_state=42
)
Wel_PT1.to_csv("Wel_PT1.csv", index=False)
Wel_PT2.to_csv("Wel_PT2.csv", index=False)
Wel_PT3.to_csv("Wel_PT3.csv",index=False)

