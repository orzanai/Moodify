
# Moodify



Moodify is a project that aims to recognize the emotions of songs using the LGBM model and generate more emotion-focused recommendations by utilizing the obtained emotion output in conjunction with cosine similarity.

For mor detailed information, please check the documentation below.




## Dataset
Our train dataset was constructed using the principle of Wisdom of Crowds, which encompasses four distinct emotions: happiness, sadness, calmness, and energy. Leveraging the Spotify API, we accessed songs from extensively listened-to playlists and user-curated selections, employing statistical balancing techniques informed by relevant research papers to ensure the representation of diverse moods. [Robert Thayer’s traditional model of mood](https://sites.tufts.edu/eeseniordesignhandbook/2015/music-mood-classification/) study forms the basis of our investigation. 


The main features of train datasets are:

- Acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
- Danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
- Energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
- Instrumentalness: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
- Liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides a strong likelihood that the track is live.
- Loudness: the overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing the relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.
- Speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audiobook, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
- Valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
- Tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, the tempo is the speed or pace of a given piece and derives directly from the average beat duration.


Test Dataset link:

```bash
 https://www.kaggle.com/datasets/abdullahorzan/moodify-dataset
```

## Steps of Project

- Created labelled song dataset
- EDA and Feature Engineering
- Built best Machine Learning Model
- Hyperparameter Optimization
- Streamlit app for deploy


## Deployment

To deploy this project change Client ID and Secret ID and run

```bash
  streamlit run Streamlit_app.py
```


## Documentation

[Documentation](https://github.com/orzanai/Moodify/blob/main/Moodify.pdf)


## Team Members

- [@A1iakbar](https://github.com/A1iakbar)
- [@gundaesra]((https://github.com/gundaesra))
- [@YigitDncr](https://github.com/YigitDncr)





## Acknowledgements
This project completed as a part of Miuul Data Science & Machine Learning Bootcamp.
