# Use official TensorFlow image with Python 3.10 and pip preinstalled
FROM tensorflow/tensorflow:2.11.0

# Set working directory
WORKDIR /app

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose Streamlit's port
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.enableCORS=false", "--server.port=8501"]

FROM tensorflow/tensorflow:2.13.0
