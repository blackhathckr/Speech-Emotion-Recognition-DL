import joblib as jb 
import numpy as np
import librosa
import soundfile
import streamlit as st

model=jb.load('model.h5')

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        return result

#audio="path_to_audio"

st.title("Speech Emotion Recognition Web App")

audio=st.file_uploader("Choose the Audio File ['wav','mp3']")

if st.button('Speech Emotion'):
    feature=extract_feature(audio, mfcc=True, chroma=True, mel=True)
    speech_emotion=model.predict(np.array([feature]))
    st.title(speech_emotion.item().upper())
