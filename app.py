import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("maize_disease_model.h5")
class_names = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']

treatment_map = {
    'Blight': 'Use Silver Nanoparticles',
    'Common Rust': 'Use Copper Nanoparticles',
    'Gray Leaf Spot': 'Use Silica Nanoparticles',
    'Healthy': 'No treatment needed'
}

st.title("Maize Leaf Disease Detection")
uploaded = st.file_uploader("Upload a maize leaf image...", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Suggested Nanotech Treatment: {treatment_map[predicted_class]}")
