FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    build-essential \
    gcc \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE="/app/cache/transformers"
ENV HUGGINGFACE_HUB_CACHE="/app/cache/huggingface"

WORKDIR /app

COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Create cache directories with proper permissions
RUN mkdir -p /app/cache/transformers /app/cache/huggingface && \
    chmod -R 777 /app/cache

COPY . .

# Add the EXPOSE instruction with your application's port
EXPOSE 80
CMD ["python", "bot.py"]