import joblib
from lightgbm import LGBMClassifier
import streamlit as st
from streamlit_extras.app_logo import add_logo
from streamlit_extras.colored_header import colored_header
from streamlit_extras.let_it_rain import rain
from streamlit_vertical_slider import vertical_slider
from IPython.display import HTML
from IPython.display import Audio
from PIL import Image
from streamlit_toggle import st_toggle_switch
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import requests
import base64
import io
import clipboard
import pyperclip
from PIL import Image
import webbrowser
from streamlit_extras.switch_page_button import switch_page
from sklearn.metrics.pairwise import cosine_similarity


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 800)


## Functions

def create_access_token():
    client_id = ""
    client_secret = ""

    # Encode client_id and client_secret
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")

    # Set the headers and data for the request
    headers = {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {"grant_type": "client_credentials"}

    # Make the POST request
    response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)

    # Parse the response
    token_data = response.json()
    access_token = token_data["access_token"]
    expires_in = token_data["expires_in"]

    print("Access Token:", access_token)
    print("Expires In:", expires_in)
    return access_token

def create_access_token_new():
    client_id = "client id form spofity api"
    client_secret = "client secret from spotify api"
    # Encode client_id and client_secret
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
    # Set the headers and data for the request
    headers = {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {"grant_type": "client_credentials"}

    # Make the POST request
    response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)


    # Parse the response
    token_data = response.json()
    access_token = token_data["access_token"]
    expires_in = token_data["expires_in"]

    print("Access Token:", access_token)
    print("Expires In:", expires_in)
    return access_token

# Function to retrieve additional artist and song features
def getArtistData(trackId):

    token_new = create_access_token_new()
    spNew = spotipy.Spotify(auth=token_new)
    print("trackId",trackId)
    try:
        if spNew.track(trackId):
            trackData = spNew.track(trackId)
        if trackData["album"]:
            album = trackData["album"]
        if album["name"]:
            albumName = album["name"]
        if album["artists"][0]["name"]:
            artistName = album["artists"][0]["name"]
        if album["images"][1]:
            trackThumbnail = album["images"][1]
        if trackData["duration_ms"]:
            trackDuration = trackData["duration_ms"]
        if album["release_date"]:
            releaseDate = album["release_date"]

        trackDuration = round(trackDuration / 1000 / 60 ,2)
        trackDuration = str(trackDuration).replace(".", ":")

        return trackThumbnail["url"],albumName, artistName, trackDuration, releaseDate
    except:
        st.write("I tried so hard to find recommendationa for you:( Can you try another one?")


# Function to retrieve song features
def get_song_features(song_link):
    # Set up your Spotify API access token
    token = create_access_token()
    sp = spotipy.Spotify(auth=token)
    # Extract the track ID from the song link
    track_id = song_link.split('/')[-1].split('?')[0]

    # Get the audio features of the track
    track_features = sp.audio_features([track_id])[0]

    # Get additional track details
    track_info = sp.track(track_id)
    duration_ms = track_info['duration_ms']
    popularity = track_info['popularity']

    # Add duration and popularity to the audio features
    track_features['duration_ms'] = duration_ms
    track_features['popularity'] = popularity

    df = pd.DataFrame([track_features])
    return df

# Funciton to format input song
def data_prep(test):
    ## Test Asamasi
    test.rename(columns={"time_signature": "time signature", "duration_ms": "duration (ms)"}, inplace=True)
    test = test[["duration (ms)", "popularity", "danceability", "energy", "loudness", "speechiness", "acousticness",
                 "instrumentalness", "liveness", "valence", "tempo", "time signature"]]
    return test


# Load the pre-trained model
# Replace this with your own model loading code
def load_model():
    # Load your model here
    model_path = r"../lgbm_model_last"  # Replace with the actual path to your model file
    model = joblib.load(model_path)
    return model


# Load the dataset
# Replace this with your own dataset loading code
def load_dataset():
    # Load your dataset here
    df = pd.read_csv(r"../278k_labelled_uri.csv")
    return df

# Function to predict the target using the pre-trained model
def predict_target(link):
    df = get_song_features(link)
    df = data_prep(df)
    model = load_model()
    result = model.predict(df)
    return int(result)

def convert_uri_to_link(uri):
    link = uri.replace("spotify:track:", "https://open.spotify.com/track/")
    return link

def get_track_id(url):
    # Split the URL by '/' and retrieve the last element
    print("url",url)
    segments = url.split('/')
    track_id = segments[-1].split('?')[0]
    return track_id

def get_random_observation(dataset, target):
    # Filter the dataset to include only rows with the specified target
    filtered_dataset = dataset[dataset['labels'] == target]

    # Get a random observation from the filtered dataset
    random_observation = filtered_dataset.sample(n=1)

    # Return the random observation
    random_observation = convert_uri_to_link(random_observation["uri"].iloc[0])
    #formatted_uri = "https://open.spotify.com/track" + random_observation.split(":")[-1]
    return random_observation




# Main Streamlit app
def main():
    # Logo container
    logo_path = './Moodify.png'
    logo = Image.open(logo_path)
    logo_resized = logo.resize((700, 700))

    # Logo encode
    buffer = io.BytesIO()
    logo_resized.save(buffer, format='PNG')
    logo_base64 = base64.b64encode(buffer.getvalue()).decode()

    st.markdown(
        """
        <style>
        .logo-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 10vh;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Logo görüntüsünü ve yazıyı sayfanın üst orta kısmında gösterin
    st.markdown(
        """
        <div class='logo-container'>
            <img src='data:image/png;base64,{}' alt='logo' width='200' height='200'>
        </div>
        """.format(logo_base64),
        unsafe_allow_html=True
    )
    # Logo dosyasını yükleyin
    logo = Image.open('Moodify.png')


    rain(
        emoji="♪",
        font_size=36,
        falling_speed=5,
        animation_length="infinite",
    )

    # Load the pre-trained model and dataset
    dataset = load_dataset()

    # Get user input
    link = st.text_input("Enter a link:")
    predict_button = st.button("Predict")

    if predict_button:
        # Predict the target using the pre-trained model
        prediction = predict_target(link)

        # Fetch a random observation with the same target
        # Map prediction values to corresponding labels
        if prediction == 0:
            prediction_label = "Sad"
        elif prediction == 1:
            prediction_label = "Happy"
        elif prediction == 2:
            prediction_label = "Energetic"
        elif prediction == 3:
            prediction_label = "Calm"
        else:
            prediction_label = "Unknown"
        random_observation = get_random_observation(dataset, int(prediction))

        st.markdown("## Prediction")
        st.markdown(
            f"<div style='background-color: #262730; padding: 10px;'>"
            f"<span style='color: white; font-size: 16px; font-weight: bold;'>Mood of this song is {prediction_label}.</br> Here are Moodify's best recommendations for your mood and the song:</span>"
            "</div>",
            unsafe_allow_html=True
        )
        ############################## cosine similarity ##############################

        # Assuming X is the DataFrame with a single observation
        # Assuming Y is the DataFrame with multiple observations
        data = load_dataset()
        song = get_song_features(link)
        song = data_prep(song)

        X = song.drop(["duration (ms)", "popularity", "time signature"], axis=1)
        Y = data[data["labels"] == 1].drop(["labels", "duration (ms)", "spec_rate", "Unnamed: 0.1","Unnamed: 0", "uri"], axis=1)

        # Compute cosine similarity
        cosine_sim = cosine_similarity(X.values, Y.values)

        # Create a DataFrame with the cosine similarity values
        similarity_df = pd.DataFrame(cosine_sim, columns=Y.index)

        # Retrieve the cosine similarity values for the first observation in X
        similarities = similarity_df.iloc[0]

        # Find the indices of the top 5 most similar observations
        top_5_indices = similarities.nlargest(5).index

        # Retrieve the top 5 most similar observations from Y
        top_5_observations = data.loc[top_5_indices]
        recoms = top_5_observations["uri"]


        track_id = get_track_id(link)
        base_url = "https://open.spotify.com/track/"
        new_link = base_url + track_id

        uris = recoms.to_list()
        linked = []

        for uri in uris:
            link = uri.replace("spotify:track:", "https://open.spotify.com/track/")
            print("link",link)
            linked.append(link)
            if new_link in linked:
                #print("linked",linked)
                linked.remove(new_link)


        ################### API visual request ##########################

        st.markdown("")
        try:
            for elems in linked:
                print("elems",elems)
                thumb_id = elems.strip("https://open.spotify.com/track/")
                imgUrl, songName, artistName, releaseDate, trackDuration = getArtistData(thumb_id)
                st.markdown(
                f"<div style='background-color: #262730; padding: 10px;display:flex;column-gap:24px; width:100%'>"
                f"<a href={elems} style='color: white; font-size: 16px; font-weight: bold;'>"
                    f"<img src={imgUrl} />"
                f"</a>"
                f"<div>"
                    f"<div style='font-size: 12px; '>Song</div>"
                        f"<div style='font-size: 36px; '>{songName}</div>"
                            f"<div style='display: flex; column-gap:8px;'>"
                                f"<div style='font-size: 12px; '>{artistName}</div>"
                                f"<div style='font-size: 12px; '>• {releaseDate[:4]}</div>"
                                f"<div style='font-size: 12px; '>• {trackDuration}</div>"
                            f"</div>"
                    f"</div>"
                "</div>",
                    unsafe_allow_html=True
                )
        except:
            print("ERROR")

        ##############################

def set_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://cdn.discordapp.com/attachments/1097587522994962502/1111071962689720331/Untitled_design-2.png");
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# Run the app
if __name__ == '__main__':
    set_custom_css()
    main()


#cd "project path"
#streamlit run streamlit_app.py
