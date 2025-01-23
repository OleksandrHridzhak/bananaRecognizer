import os
import numpy as np
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image  import load_img, img_to_array, ImageDataGenerator
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.utils import to_categorical

# Paths
train_dir = 'MY_data/train'
validation_dir = 'MY_data/test'

# Image dimensions and parameters
img_size = (150, 150)
batch_size = 32
epochs = 10

# Data Preprocessing (Image Augmentation for training data)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data should only be rescaled
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow data from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Папка для навчання
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'  # Бінарна класифікація: є банан чи ні
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,  # Папка для тестування
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'  # Бінарна класифікація
)

# Build the model (CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Одна вихідна одиниця для бінарної класифікації
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save model weights
model.save('model_weights.h5')

print("Model training complete and weights saved!")