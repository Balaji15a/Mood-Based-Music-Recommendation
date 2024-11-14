import cv2
import numpy as np
import tensorflow as tf
import random
import os
from pygame import mixer

# Load your pre-trained emotion detection model
model = tf.keras.models.load_model('emotion_detection_model.h5')

# Map emotions to music folder paths
emotion_music_map = {
    "angry": "music",
    "sad": "music",
    "happy": "music",
    "surprise": "music",
    "disgust": "music"
}

# Function to preprocess the image for emotion detection model
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (48, 48))  # Assuming model input size is 48x48
    image = np.reshape(image, (1, 48, 48, 1)) / 255.0  # Normalize
    return image

# Function to predict emotion based on image
def predict_emotion(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    emotion_index = np.argmax(prediction)
    emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    return emotion_labels[emotion_index]

# Function to play a random music file based on emotion
def play_music(emotion):
    mixer.init()
    music_folder = emotion_music_map.get(emotion, None)
    if music_folder and os.path.exists(music_folder):
        music_files = [f for f in os.listdir(music_folder) if f.endswith('.mp3')]
        if music_files:
            selected_music = random.choice(music_files)
            music_path = os.path.join(music_folder, selected_music)
            mixer.music.load(music_path)
            mixer.music.play()
            print(f"Playing {selected_music} for emotion: {emotion}")
        else:
            print(f"No music files found for emotion: {emotion}")
    else:
        print(f"Music folder for emotion '{emotion}' not found.")

# Main function to detect emotion and play corresponding music
def mood_based_music_recommendation(test_folder):
    for emotion_folder in os.listdir(test_folder):
        emotion_path = os.path.join(test_folder, emotion_folder)
        if os.path.isdir(emotion_path):
            for image_file in os.listdir(emotion_path):
                image_path = os.path.join(emotion_path, image_file)
                emotion = predict_emotion(image_path)
                play_music(emotion)
                input("Press Enter to analyze next image...")  # Wait for user input to continue
                mixer.music.stop()

# Run the function
test_folder_path = "test"
mood_based_music_recommendation(test_folder_path)
