# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by TensorFlow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender1 libhdf5-dev libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.enableCORS=false", "--server.port=8501"]

