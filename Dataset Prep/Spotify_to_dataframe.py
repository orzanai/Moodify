import pandas as pd
import glob
import os
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

path = "/Users/abdullahorzan/Desktop/Miuul/DSML 11 Datavaders/final project/"
files = glob.glob(path + "/*.csv")
content = []

for filename in files:
    df = pd.read_csv(filename)
    df["labels"] = os.path.splitext(os.path.basename(filename))[0]
    content.append(df)

df = pd.concat(content, axis=0)
df.dropna(inplace=True)

df.to_csv("emotions_dataset_train.csv")