import streamlit as st
from PIL import Image
import os
from model import predict_image  # From model.py

# Set Streamlit config
st.set_page_config(page_title="Mosquito Classifier", layout="centered")
st.title("🦟 Mosquito Species Classifier")

# Upload an image
uploaded_file = st.file_uploader("Upload a mosquito image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image temporarily
    temp_path = f"temp/{uploaded_file.name}"
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict using the model
    with st.spinner("Classifying..."):
        try:
            result = predict_image(temp_path)

            st.success(f"✅ Predicted Species: **{result['predicted_class']}**")
            st.info(f" Confidence: **{result['confidence']:.2f}**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

    # Clean up temp file
    os.remove(temp_path)