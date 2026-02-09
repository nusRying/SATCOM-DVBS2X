## DVB-S2/S2X TX-RX environment
## - Single-stage image optimized for reproducible CI/local runs
## - Includes BLAS/LAPACK toolchain so SciPy/NumPy wheels can build if wheels unavailable
## - Default CMD shows RX loopback help; override to run other scripts
FROM python:3.11-slim

# Prevent .pyc noise, keep logs unbuffered, make repo importable anywhere
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# System packages for scientific Python stack (BLAS/LAPACK + build tools)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps separately for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the project
COPY . /app

# Default command: display loopback test options (override with `docker run ... <your command>`)
CMD ["python", "tests/test_tx_rx_loopback.py", "--help"]
