FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"

# Copy application code
COPY app.py .
COPY ../testing/scripts /app/testing/scripts

# Expose port
EXPOSE 7860

# Run app
CMD ["python", "app.py"]

