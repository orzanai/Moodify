import numpy as np
import pandas as pd

# Data Reading
df1 = pd.read_csv("before_adding")
df2 = pd.read_csv("songs_we_want_to_add")

df1.head()
df2.head()
df1.shape
df2.shape

# Feature selection
df2.columns = df2.columns.str.lower()
df2.drop(["unnamed: 0", "mode"], axis=1, inplace=True)
df1.drop(["album",  "release_date"], axis=1, inplace=True)
df1.rename(columns={'mood': 'labels', "name": "track", "id": "uri", "time_signature": "time signature", "length": "duration (ms)"}, inplace=True)

column_order = df2.columns
df1 = df1.reindex(columns=column_order)

df1["uri"] = df1["uri"].apply(lambda x: "spotify:track:" + x)
df1["labels"] = df1["labels"].apply(lambda x: x.lower())

# Observation selection
calm_ekle = df2[df2["labels"] == "calm"].iloc[:105]
sad_ekle = df2[df2["labels"] == "sad"].iloc[:103]
energ_ekle = df2[df2["labels"] == "energetic"].iloc[:146]
happy_ekle = df2[df2["labels"] == "happy"].iloc[:160]

# Final data frame
result = pd.concat([df1, calm_ekle, sad_ekle, energ_ekle, happy_ekle])

result.describe().T

# Mapping emotions
emotions_mapping = {'sad': 0, 'happy': 1, 'energetic': 2, 'calm': 3}
result["labels"] = result["labels"].map(emotions_mapping)

# Export CSV
result.to_csv("final_songs.csv")

