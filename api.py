from fastapi import FastAPI, UploadFile, File
from model import predict_image
import os
import shutil

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = predict_image(temp_path)

    os.remove(temp_path)
    return result
