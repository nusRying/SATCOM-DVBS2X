FROM python:3.11-slim

# Prevent Python from writing .pyc files and keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# System packages for scientific Python stack
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies separately for better build caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the project
COPY . /app

# Default command: show loopback test options
CMD ["python", "tests/test_tx_rx_loopback.py", "--help"]
