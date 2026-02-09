# Docker Usage

Reproducible environment for the DVB-S2/S2X TX/RX toolchain. Image is built from `python:3.11-slim` with BLAS/LAPACK build tooling; repo lives at `/app` and `PYTHONPATH` is set accordingly.

## Build
```bash
docker build -t dvbs2x .
```

## Quick run: loopback test
Persists outputs to host volumes so artifacts survive container exit.
```bash
docker run --rm -it \
  -v ${PWD}/results:/app/results \
  -v ${PWD}/loopback_output:/app/loopback_output \
  dvbs2x \
  python tests/test_tx_rx_loopback.py --max-frames 1 --esn0-db 5
```
List options:
```bash
docker run --rm dvbs2x
docker run --rm dvbs2x python tests/test_tx_rx_loopback.py --help
```

## docker-compose
```bash
docker compose up --build
```
Defaults to `--max-frames 1 --esn0-db 5` and mounts:
- `./results` -> `/app/results`
- `./loopback_output` -> `/app/loopback_output`
- `./dvbs2x_output` -> `/app/dvbs2x_output`

Edit `docker-compose.yml` to change the command or add volumes (e.g., `demo_output/`).

## Custom commands
- Transmitter only:
  ```bash
  docker run --rm dvbs2x python tx/run_dvbs2.py
  ```
- Interactive shell:
  ```bash
  docker run --rm -it dvbs2x bash
  ```

## What’s in the image
- Base: `python:3.11-slim`.
- Apt: `build-essential`, `gfortran`, `libopenblas-dev`, `liblapack-dev`.
- Python deps: `requirements.txt` installed during build.
- Default CMD: `python tests/test_tx_rx_loopback.py --help` (override with your own command).

## Volumes & outputs
- Mount `results/`, `loopback_output/`, `dvbs2x_output/` to capture BER reports, plots, and intermediate dumps. Add more mounts if you write elsewhere.

## Tips / Troubleshooting
- Adding deps: update `requirements.txt` then rebuild. Layer cache is preserved if `requirements.txt` is unchanged.
- Need different Python version? Change `FROM` line and rebuild.
- To inspect intermediate files, launch a shell with the same volume mounts you use for tests.
