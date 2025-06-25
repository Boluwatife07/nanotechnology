import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("maize_disease_model.h5")

model = load_model()
class_names = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']
treatment_map = {
    'Blight': 'Use Silver Nanoparticles',
    'Common Rust': 'Use Copper Nanoparticles',
    'Gray Leaf Spot': 'Use Silica Nanoparticles',
    'Healthy': 'No treatment needed'
}

# UI setup
st.set_page_config(page_title="Maize Disease Detection", layout="centered")
st.title("🌽 Maize Leaf Disease Detection Using Nanotech")

# Sidebar: Upload
with st.sidebar:
    st.header("📂 Upload Image")
    uploaded = st.file_uploader("Choose a maize leaf image", type=["jpg", "png", "jpeg"])

# Session state: image history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Placeholder before upload
placeholder = st.empty()

if not uploaded:
    with placeholder.container():
        st.info("📷 Please upload a maize leaf image to begin diagnosis.")
        # Optionally display a local sample image here if available:
        # st.image("sample_leaf.jpg", caption="Sample Maize Leaf", use_container_width=True)

if uploaded:
    placeholder.empty()  # Clear placeholder
    col1, col2 = st.columns([1, 2])

    with col1:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption='🖼️ Uploaded Image', use_container_width=True)
        st.session_state['history'].append(img.copy())  # Store for history

    with col2:
        st.info("🔍 Analyzing image...")
        img_resized = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        with st.spinner("Running model..."):
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = float(np.max(prediction)) * 100

        st.success(f"✅ Prediction: **{predicted_class}**")
        st.metric(label="🔬 Model Confidence", value=f"{confidence:.2f}%")
        st.info(f"🧪 Suggested Treatment: **{treatment_map[predicted_class]}**")

        # Bar chart of probabilities
        st.subheader("📊 Prediction Breakdown")
        probs_df = pd.DataFrame(prediction[0], index=class_names, columns=["Probability"])
        st.bar_chart(probs_df)

        # Expandable treatment info
        with st.expander("🧬 Learn more about the suggested nanotech treatment"):
            st.write(f"The model detected **{predicted_class}**.")
            st.write(f"Recommended treatment: **{treatment_map[predicted_class]}**.")

        # Downloadable report
        report_text = f"""
        Maize Disease Detection Report

        Prediction: {predicted_class}
        Confidence: {confidence:.2f}%
        Suggested Treatment: {treatment_map[predicted_class]}
        """
        b = BytesIO()
        b.write(report_text.encode())
        b.seek(0)
        st.download_button("📄 Download Report", b, file_name="maize_report.txt")

# Image history
if st.session_state['history']:
    st.subheader("🖼️ Image History This Session")
    cols = st.columns(min(4, len(st.session_state['history'])))
    for i, hist_img in enumerate(reversed(st.session_state['history'][-4:])):
        with cols[i % 4]:
            st.image(hist_img, caption=f"Previous {len(st.session_state['history']) - i}", use_container_width=True)
