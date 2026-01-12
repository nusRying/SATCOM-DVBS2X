# SATCOM DVB-S2 Baseband Chain

This repo contains a Python implementation of a DVB-S2 baseband processing chain (BB frame generation, BCH/LDPC, bit interleaving, constellation mapping, PL header, and PL scrambling). It also includes sample data and reports used to validate each stage.

## Layout
- `run_dvbs2.py` - interactive transmitter run that drives the full chain
- `BB_Frame.py` - BB frame building and stream adaptation helpers
- `stream_adaptation.py` - rate adaption utilities
- `bch_encoding.py` - BCH encoding
- `ldpc_Encoding.py` - LDPC encoding (uses parity matrices in `s2xLDPCParityMatrices/`)
- `bit_interleaver.py` - DVB-S2 bit interleaver
- `constellation_mapper.py` - modulation mapping
- `pl_header.py` - PL header creation
- `pl_scrambler.py` - PL scrambling
- `GS_data/` and `TS_data/` - input examples
- `Data/` - generated or intermediate data (some files tracked with Git LFS)

## Quick start (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_dvbs2.py
```

## Configuration notes
- Update `BITS_CSV_PATH` in `run_dvbs2.py` to point to your input CSV file.
- Update `MAT_PATH` in `run_dvbs2.py` to point to the LDPC parity matrix file.

## Outputs
The run generates text reports and intermediate bit/symbol files in the repo root, including `dvbs2_full_report.txt`.

## Git LFS
Large data files (for example `Data/bits_single_column.csv`) are tracked with Git LFS.