# Getting started

1. Ensure config/ldpc_matrices/dvbs2xLDPCParityMatrices.mat exists.
2. Prepare input bits at data/GS_data/umair_gs_bits.csv.
3. Create venv and install:
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
4. Run loopback:
   python tests/test_tx_rx_loopback.py --max-frames 3 --esn0-db 8 --output-dir results/loopback
5. Inspect results/loopback and intermediate/ for stage outputs and plots.