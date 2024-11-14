import cv2
import numpy as np
import tensorflow as tf
import random
import os
from pygame import mixer
from collections import Counter
import time

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
def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray, (48, 48))  # Assuming model input size is 48x48
    image = np.reshape(resized_image, (1, 48, 48, 1)) / 255.0  # Normalize
    return image

# Function to predict emotion based on a frame
def predict_emotion(frame):
    processed_image = preprocess_image(frame)
    prediction = model.predict(processed_image)
    emotion_index = np.argmax(prediction)
    emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    return emotion_labels[emotion_index]

# Function to play a random music file based on emotion
def play_music(emotion):
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

# Function to collect live emotion and play music accordingly
def live_emotion_music():
    # Initialize the mixer and webcam
    mixer.init()
    cap = cv2.VideoCapture(0)  # Open the webcam
    last_emotion = None
    frame_count = 0
    emotion_buffer = []

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Predict emotion every 5 frames to avoid over-processing
        if frame_count % 5 == 0:
            emotion = predict_emotion(frame)
            emotion_buffer.append(emotion)

            # Keep only the last 10 predictions for smoothing
            if len(emotion_buffer) > 10:
                emotion_buffer.pop(0)

            # Use majority vote for the most common emotion in the buffer
            most_common_emotion = Counter(emotion_buffer).most_common(1)[0][0]

            # Check if the majority emotion has changed
            if most_common_emotion != last_emotion:
                if mixer.music.get_busy():
                    mixer.music.stop()  # Stop any previously playing music
                
                if most_common_emotion in emotion_music_map:  # Only play music if a known emotion is detected
                    play_music(most_common_emotion)
                    last_emotion = most_common_emotion  # Update the last detected emotion

        # Display the frame with the detected emotion
        cv2.putText(frame, f"Emotion: {last_emotion if last_emotion else 'Detecting...'}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Live Emotion Detection', frame)

        # Increment the frame count
        frame_count += 1

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    mixer.quit()

# Run the function
live_emotion_music()
