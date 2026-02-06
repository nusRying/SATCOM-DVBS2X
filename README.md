# DVBS2X — DVB‑S2/X TX↔RX Loopback Suite

Lightweight DVB‑S2/X transmitter→receiver loopback testbench for development and experimentation.

Quickstart
1. Create venv and install:
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
2. Run a loopback:
   python tests/test_tx_rx_loopback.py --max-frames 1 --esn0-db 10
3. Run tests:
   pytest -q

Repository layout
- tx/, rx/, common/ — signal chain modules
- tests/ — loopback driver and integration tests
- config/, data/ — matrices and sample inputs
- docs/, examples/, results/

Contributing, CI and usage examples are in the repo. License: MIT.