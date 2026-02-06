# Docker Usage

This image provides a reproducible environment for the DVB-S2X TX/RX code and tests.

## Build

```bash
docker build -t dvbs2x .
```

## Run a loopback test (example)

Outputs land in mounted host folders so they persist after the container exits.

```bash
docker run --rm -it \
  -v ${PWD}/results:/app/results \
  -v ${PWD}/loopback_output:/app/loopback_output \
  dvbs2x \
  python tests/test_tx_rx_loopback.py --max-frames 1 --esn0-db 5
```

To see available arguments:

```bash
docker run --rm dvbs2x
# or
docker run --rm dvbs2x python tests/test_tx_rx_loopback.py --help
```

## Using docker-compose

```bash
docker compose up --build
```

Defaults to `--max-frames 1 --esn0-db 5` and mounts:
- `./results` → `/app/results`
- `./loopback_output` → `/app/loopback_output`
- `./dvbs2x_output` → `/app/dvbs2x_output`

Edit `docker-compose.yml` to change the command or add more volumes (e.g., to expose other output folders).

## Custom commands

- Run transmitter only:  
  ```bash
  docker run --rm dvbs2x python tx/run_dvbs2.py
  ```
- Open a shell for debugging:  
  ```bash
  docker run --rm -it dvbs2x bash
  ```

## Notes

- Base image: `python:3.10-slim` with BLAS/LAPACK build tools for SciPy/NumPy.
- `PYTHONPATH` is set to `/app` so imports work anywhere in the repo.
- If you add new Python dependencies, update `requirements.txt` and rebuild.
