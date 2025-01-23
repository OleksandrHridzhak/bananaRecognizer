import numpy as np
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
import os

# Load the trained model
model = keras.models.load_model('model_weights.h5')

# Path to the directory containing images for prediction
predict_dir = 'MY_data/predict'

# Image size must match the size used during training
img_size = (150, 150)

# Class names (0 = not banana, 1 = banana)
class_names = ['Banana', 'Not Banana']


# Predict function
def predict_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    print(f"Prediction for {image_path}: {prediction[0]}")  # Друк значень ймовірностей
    return 1 if prediction[0] > 0.5 else 0

# Iterate over all images in the 'predict' directory
for img_name in os.listdir(predict_dir):
    img_path = os.path.join(predict_dir, img_name)
    
    # Ensure it's an image file
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        class_idx = predict_image(img_path)
        predicted_class = class_names[class_idx]
        print(f"{img_name}: Predicted class: {predicted_class}")