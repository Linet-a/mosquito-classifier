# model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define paths
train_dir = "mosquito_dataset_split/train"
val_dir = "mosquito_dataset_split/val"
test_dir = "mosquito_dataset_split/test"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = "mosquito_classifier.h5"
CLASS_NAMES = ['AEDES', 'ANOPHELES', 'CULEX']

# Load datasets
def load_datasets():
    train_ds = image_dataset_from_directory(train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    val_ds = image_dataset_from_directory(val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    test_ds = image_dataset_from_directory(test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

# Build the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 3 classes
    ])
    return model

# Main training routine
def train_and_save_model():
    train_ds, val_ds, test_ds = load_datasets()
    model = build_model()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Evaluate
    loss, acc = model.evaluate(test_ds)
    print(f"\nTest Accuracy: {acc:.4f}")

    # Save the model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")

# Predict a single image using the trained model
def predict_image(img_path):
    if not os.path.exists(MODEL_SAVE_PATH):
        raise ValueError("Model not found. Please train the model first.")

    model = load_model(MODEL_SAVE_PATH)

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx])

    return {
        "predicted_class": CLASS_NAMES[class_idx],
        "confidence": round(confidence, 4)
    }
# Entry point
if __name__ == "__main__":
    train_and_save_model()
