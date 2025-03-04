# Imported necessary libraries
import tkinter as tk
from tkinter import Canvas, filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

# Function to preprocess the image for prediction
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L') # Convert to grayscale
    image = image.resize((28, 28))  # Resize to match the model's input size
    image = np.array(image)  # Convert to numpy array for mathematical operations
    image = image.reshape((1, 28, 28, 1)).astype('float32') / 255  # Reshape and normalize
    return image

# Function to make predictions
def predict_digit(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    digit = np.argmax(prediction) 
    return digit

# Function to handle the "Open Image" button
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        digit = predict_digit(file_path)

        # Display the image on the canvas
        img = Image.open(file_path)
        img = img.resize((200, 200), Image.BILINEAR) # Resize using BILINEAR sampling
        img = ImageTk.PhotoImage(img)
        canvas.img = img  
        canvas.create_image(0, 0, anchor=tk.NW, image=img)

        # Update result label
        result_label.config(text=f"Predicted Digit: {digit}")

        # Print the predicted digit
        print(f"Predicted Digit: {digit}")

# Create the main GUI window
root = tk.Tk()
root.title("Handwritten Digit Recognition")

# Create a canvas to display the image
canvas = Canvas(root, width=200, height=200)
canvas.pack()

# Create a label to display the results
result_label = tk.Label(root, text="Predicted Digit: ")
result_label.pack()

# Create a button to open an image
open_button = tk.Button(root, text="Select Image", command=open_image)
open_button.pack()

# Run the GUI main loop
root.mainloop()
