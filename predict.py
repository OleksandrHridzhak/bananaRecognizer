from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
import tkinter as tk
from tkinter import Label, Canvas, Scrollbar, Frame
from PIL import Image, ImageTk
import numpy as np
from tensorflow import keras
import os

model = keras.models.load_model('model_weights.h5')

predict_dir = 'MY_data/predict'

img_size = (150, 150)

class_names = ['Banana', 'Avocado']

def predict_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return 1 if prediction[0] > 0.5 else 0

def display_images():
    images = [f for f in os.listdir(predict_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not images:
        result_label.config(text="No images found in the folder.")
        return

    row, col = 0, 0
    for image_name in images:
        image_path = os.path.join(predict_dir, image_name)

        class_idx = predict_image(image_path)
        predicted_class = class_names[class_idx]

        img = Image.open(image_path)
        img = img.resize((150, 150))
        tk_img = ImageTk.PhotoImage(img)

        image_label = Label(scrollable_frame, image=tk_img)
        image_label.image = tk_img
        image_label.grid(row=row, column=col, padx=10, pady=10)

        text_label = Label(scrollable_frame, text=f"{image_name}: {predicted_class}", font=("Arial", 12))
        text_label.grid(row=row + 1, column=col, padx=10, pady=5)

        col += 1
        if col >= 3:
            col = 0
            row += 2

root = tk.Tk()
root.title("Batch Image Prediction")
root.geometry("600x600")

canvas = Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = Scrollbar(root, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

scrollable_frame = Frame(canvas)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

result_label = Label(scrollable_frame, text="", font=("Arial", 16))
result_label.grid(row=0, column=0, columnspan=3, pady=10)

display_images()

root.mainloop()