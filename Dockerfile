FROM python:3.12-slim

# Install Tesseract OCR system binary
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Application directory
WORKDIR /app

# Install Python dependencies first for better Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Render expects the web service to listen on port 10000
EXPOSE 10000

# Start FastAPI
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "10000"]