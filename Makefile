.PHONY: install test lint run

install:
    python -m pip install -r requirements.txt

test:
    pytest -q

lint:
    flake8 .

run:
    python tests/test_tx_rx_loopback.py --max-frames 1