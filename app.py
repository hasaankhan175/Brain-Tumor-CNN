import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Set app page
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Classification App")
st.markdown("""
Upload a brain MRI scan to classify the tumor type.  
Model Categories: **Glioma**, **Meningioma**, **Pituitary**, **No Tumor**
""")

# Path to model
MODEL_PATH = r"D:\portfolio_of_data_science\Brain_tumor_cnn\saved_model\brain_tumor_model.h5"

# Class labels
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("📤 Upload a Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((512, 512))
    img_array = np.array(img_resized)

    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove alpha channel if present

    img_normalized = img_array / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict
    with st.spinner("🔍 Classifying..."):
        predictions = model.predict(img_input)[0]
        predicted_index = np.argmax(predictions)
        predicted_label = class_names[predicted_index]
        confidence = predictions[predicted_index] * 100

    # Display prediction result
    st.markdown("## 🧪 Prediction Result:")
    st.success(f"🎯 **Predicted Tumor Type:** `{predicted_label}`")
    st.info(f"📊 **Confidence Score:** `{confidence:.2f}%`")

    st.caption("⚠️ This is a deep learning prediction and should not be used as a medical diagnosis. Always consult a doctor.")
    
    st.markdown("### 📈 Class Probabilities:")
    for i, score in enumerate(predictions):
        st.write(f"- **{class_names[i]}**: `{score*100:.2f}%`")
