FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD ["python", "tests/test_tx_rx_loopback.py", "--max-frames", "1"]