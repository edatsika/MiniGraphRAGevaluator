FROM python:3.11-slim

# System dependencies
RUN apt-get update && \
    apt-get install -y gcc curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK resources (needed for tokenization)
RUN python -m nltk.downloader punkt punkt_tab

# Copy app code
COPY . .

# Expose ports
EXPOSE 8000 8001

# Entrypoint
CMD ["python", "graphrag_pipeline.py"]
