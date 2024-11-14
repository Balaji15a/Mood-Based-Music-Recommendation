import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Define paths
train_data_dir = 'test'  # Path to the dataset folder

# Image data generator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,       # Normalize pixel values between 0 and 1
    shear_range=0.2,       # Shear transformation
    zoom_range=0.2,        # Randomly zoom
    horizontal_flip=True,  # Randomly flip images
    validation_split=0.2   # 20% data for validation
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(48, 48),    # Resize images to 48x48 pixels
    batch_size=32,
    color_mode='grayscale',  # Use grayscale images for simplicity
    class_mode='categorical',
    subset='training'
)

# Load validation data
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Save the model to an .h5 file
model.save('emotion_detection_model.h5')
print("Model saved as 'emotion_detection_model.h5'")
