import streamlit as st
from PIL import Image
import os
from model import predict_image

st.set_page_config(page_title="Mosquito Classifier", layout="centered")

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .title {
        font-size: 42px;
        font-weight: bold;
        background: -webkit-linear-gradient(90deg, #FF512F, #DD2476);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
    }

    .confidence {
        font-size: 24px;
        font-weight: 600;
        color: #444;
        text-align: center;
        margin-top: 10px;
    }

    .shadow-box {
        background-color: #ffffff;
        padding: 25px;
        margin-top: 30px;
        border-radius: 18px;
        border: 2px solid #e2e2e2;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }

    .center {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- App Title ---
st.markdown('<div class="title">🦟 Mosquito Species Classifier</div>', unsafe_allow_html=True)
st.write("Upload an image of a mosquito, and let the model identify its species.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display layout with two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Save image temporarily
    temp_path = f"temp/{uploaded_file.name}"
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with col2:
        with st.spinner("🧠 Classifying..."):
            try:
                result = predict_image(temp_path)
                predicted = result["predicted_class"]
                confidence = result["confidence"]

                st.subheader(f"🔍 Prediction: {predicted}")
                st.markdown(f"<div class='confidence'>Confidence: {confidence:.2%}</div>", unsafe_allow_html=True)

                # Confidence progress bar
                st.progress(confidence)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    os.remove(temp_path)
