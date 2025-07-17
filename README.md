# 🦟 Mosquito Species Classification

This is a machine learning-based web API for classifying mosquito species from images. It uses a Convolutional Neural Network (CNN) trained on image data to identify whether an image contains **Aedes**, **Anopheles**, **Culex**, or flag uncertain/non-mosquito inputs.

## 📌 Features

- Classifies mosquito species from uploaded images.
- Built with **FastAPI** and served with **Uvicorn**.
- Trained using **TensorFlow** on a custom image dataset.
- Handles image uploads and returns predictions.
- Supports **Dockerization** for easy deployment.
- Accepts `.jpg`, `.jpeg`, or `.png` image formats.

---

## 🧠 Model Overview

- Architecture: Simple CNN with Conv2D, MaxPooling, Dense, Dropout
- Input Shape: `(128, 128, 3)`
- Classes: `Aedes`, `Anopheles`, `Culex`, and optionally `Unknown`
- Loss Function: `sparse_categorical_crossentropy`
- Optimizer: `adam`
- Metrics: `accuracy`

---

## 📁 Project Structure

mosquito-project/
│
├── model.py # Model training and saving script
├── predict.py # Prediction function for loading and using saved model
├── api.py # FastAPI application
├── requirements.txt # Project dependencies
├── Dockerfile # For containerization
├── README.md # You are here!
└── mosquito_classifier.h5 # Saved model (auto-generated after training)


---

## ⚙️ Installation

### 🔧 Local Setup

1. Clone the repo:
```bash
git clone https://github.com/yourusername/mosquito-project.git
cd mosquito-project

### Create a Virtual Environment 
python -m venv venv
source venv/bin/activate

#Train model
pip install -r requirements.txt

### Run API

🌍 Future Work
Integrate mobile camera capture

Use transfer learning (e.g., MobileNetV2)

Deploy online via Render or Hugging Face Spaces

Add CI/CD with GitHub Actions