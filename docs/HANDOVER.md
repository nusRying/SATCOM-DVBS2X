# SATCOM-DVBS2X Handover Report

## Project at a Glance
- **Purpose:** DVB‑S2/S2X baseband TX/RX chain with loopback testing and reporting.
- **Stack:** Python 3.10/3.11, NumPy, SciPy, matplotlib, pytest, flake8.
- **Entry points:** `tx/` (transmitter), `rx/` (receiver), `tests/` (loopbacks), `docs/` (guides).

## Repository Layout (essentials)
- `tx/` — TX stages: BB frame, stream adaptation, BCH encode, LDPC encode, interleave/map, PL header, pilot insertion, scrambler.
- `rx/` — RX stages: PL descramble, pilot removal, pilot phase correction, demap, de‑interleave, LDPC decode, BCH check, BB deframe. Orchestrator: `rx/receiver_Chain.py`.
- `common/` — Shared utilities: bit interleaver, constellation mapper, pilot insertion constants, PL scrambler.
- `config/ldpc_matrices/` — `dvbs2xLDPCParityMatrices.mat` (20 MB) parity tables.
- `tests/` — Loopback/system tests (`test_rx_loopback.py`, `test_tx_rx_loopback.py`, demos).
- `config/archived_code/` — Legacy merger/slicer pipeline; tests in `test_pl_frame.py`.
- `docs/` — How‑tos and per‑stage RX docs (`docs/rx_chain/*`), Docker instructions.

## How to Run
- All tests: `PYTHONPATH=. pytest -q`
- TX→RX loopback: `PYTHONPATH=. python tests/test_tx_rx_loopback.py --max-frames 1 --esn0-db 5`
- RX-only loopback: `PYTHONPATH=. python tests/test_rx_loopback.py`
- Docker build: `docker build -t dvbs2x .`
- Docker run loopback:  
  `docker run --rm -v ${PWD}/results:/app/results -v ${PWD}/loopback_output:/app/loopback_output dvbs2x python tests/test_tx_rx_loopback.py --max-frames 1 --esn0-db 5`
- docker-compose (defaults in `docker-compose.yml`): `docker compose up --build`

## Continuous Integration
- Workflow: `.github/workflows/ci.yml`
- Matrix: Python `3.10`, `3.11`; `PYTHONPATH=.` set.
- Commands: install `requirements.txt` (no cache) then `pytest -q`.
- Naming: tests must follow `test_*.py`; legacy file renamed to `config/archived_code/test_pl_frame.py` for discovery.

## Key Files (by function)
- RX orchestrator: `rx/receiver_Chain.py`
- TX core: `tx/_01_BB_Frame.py`, `tx/_02_stream_adaptation.py`, `tx/_03_bch_encoding.py`, `tx/_04_ldpc_Encoding.py`, `tx/_05_pl_header.py`
- Shared math: `common/constellation_mapper.py`, `common/pilot_insertion.py`, `common/bit_interleaver.py`, `common/pl_scrambler.py`
- Tests: `tests/test_tx_rx_loopback.py`, `tests/test_rx_loopback.py`, `config/archived_code/test_pl_frame.py`
- Docs: `DOCKER.md`, `DETAILED_REPORT_GUIDE.md`, `ENHANCED_REPORTING_README.md`, RX stage docs in `docs/rx_chain/`

## RX Chain Docs (docs/rx_chain/)
- `overview.md` — Flow (Stages 12–19), dimensions, interfaces.
- `stage12_pl_descrambler.md` … `stage19_bb_deframer.md` — Per-stage math, inputs/outputs.
- `stage16_bit_deinterleave.md` — Inverse interleaver details.

## Current Status
- Local pytest: **13 passed** (`PYTHONPATH=. pytest -q`).
- CI fixed for naming/imports; rerun GitHub Actions after pull.
- Dockerfile and DOCKER.md updated for reproducible runs.

## Known Gaps / TODOs
- BCH decoder is detect‑only; implement error correction (Berlekamp–Massey/Chien) for spec completeness.
- APSK gamma tables cover higher‑rate modes; extend for lower‑rate APSK MODCODs if needed.
- LDPC MAT path assumed at `config/ldpc_matrices/dvbs2xLDPCParityMatrices.mat`; keep or update references.

## Quick Handover Checklist
1) Rerun CI (`.github/workflows/ci.yml`) on Python 3.10/3.11.  
2) Ensure `dvbs2xLDPCParityMatrices.mat` remains available at expected path.  
3) Name new tests `test_*.py` so pytest and CI discover them.  
4) Update `requirements.txt` and rebuild Docker if adding dependencies.

## Useful One-Liners
- `PYTHONPATH=. pytest -q`
- `PYTHONPATH=. python tests/test_tx_rx_loopback.py --max-frames 1 --esn0-db 5`
- `docker build -t dvbs2x .`
- `docker run --rm -v ${PWD}/results:/app/results dvbs2x pytest -q`
