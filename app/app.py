# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model("/home/user/Downloads/Darshan_data/Portfolio/digit-recognition-cicd/models/model.h5")

# Set up the Streamlit app interface
st.title("Handwritten Digit Recognition")
st.write("Upload a digit image to predict.")

# File uploader for image upload
uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image to match the model input shape
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img)  # Convert to numpy array
    img_array = img_array.astype('float32') / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model

    # Predict the digit
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    # Display the prediction
    st.write(f"Predicted Digit: {predicted_digit}")
