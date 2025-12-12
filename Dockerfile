# Railway-optimized Dockerfile for PageCraftML
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files (exclude __pycache__, .git, etc.)
COPY nn_server.py .
COPY gnn_model.py .
COPY data_processor.py .
COPY decode_predictions.py .
COPY predict_with_model.py .
COPY model_checkpoint.pth .
COPY start.sh .

# Make start script executable
RUN chmod +x start.sh

# Create training_data directory if it doesn't exist
RUN mkdir -p training_data

# Expose port (Railway will set PORT env variable)
EXPOSE 8000

# Run the application using start script
CMD ["./start.sh"]
