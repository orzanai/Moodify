import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 800)

df = pd.read_csv("labelled_csv_path_with_uri")
last_ = last

#Assuming X is the DataFrame with a single observation
#Assuming Y is the DataFrame with multiple observations
X = test.drop(["duration (ms)", "popularity", "time signature"], axis=1)
Y = last[last["labels"] == 1].drop(["labels", "duration (ms)", "spec_rate","Unnamed: 0" ,"Unnamed: 0.1", "uri"], axis=1)


#Compute cosine similarity
cosine_sim = cosine_similarity(X.values, Y.values)

#Create a DataFrame with the cosine similarity values
similarity_df = pd.DataFrame(cosine_sim, columns=Y.index)

#Retrieve the cosine similarity values for the first observation in X
similarities = similarity_df.iloc[0]

#Find the indices of the top 5 most similar observations
top_5_indices = similarities.nlargest(5).index

#Retrieve the top 5 most similar observations from Y
top_5_observations = last_.loc[top_5_indices]

#Print the top 5 most similar observations
print("Top 5 Most Similar Observations:")
print(top_5_observations)

#Rename uri column as Recommendations
top_5_observations.rename(columns={"uri": "Recommendations"}, inplace=True)


# Get song id and create Spotify link with id
track_id = get_track_id("https://open.spotify.com/track/4f8Mh5wuWHOsfXtzjrJB3t?si=22eacda57b5b45b0")
base_url = "https://open.spotify.com/track/"
new_link = base_url + track_id

# Check if song in the Recommendations
uris = top_5_observations["Recommendations"].to_list()
recommendations_list = []

for uri in uris:
    link = uri.replace("spotify:track:", "https://open.spotify.com/track/")
    recommendations_list.append(link)

if id_link in linked:
    #print(True)
    recommendations_list.remove(id_link)
    




