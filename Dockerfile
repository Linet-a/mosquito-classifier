# Use official TensorFlow image (2.11.0) with Python and pip
FROM tensorflow/tensorflow:2.11.0

# Set working directory inside container
WORKDIR /app

#Copy dependency file first to leverage caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the application code
COPY . .

# Expose Streamlit's default port (for local use; Render uses 8080 internally)
EXPOSE 8501

# Run the Streamlit app (set correct port + disable CORS)
CMD ["streamlit", "run", "app.py", "--server.enableCORS=false", "--server.port=8080", "--server.address=0.0.0.0"]
