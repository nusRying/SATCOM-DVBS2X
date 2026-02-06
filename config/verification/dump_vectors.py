"""
Generate MATLAB-friendly test vectors for DVB-S2 BCH + LDPC cross-checks.

This script:
  1) Reads a 0/1 bitstream file (any length <= Kbch).
  2) Pads and BB-scrambles to Kbch (rate-aware).
  3) Runs BCH outer code to Nbch.
  4) Runs LDPC inner code to Nldpc.
  5) Writes a single MAT file with all stages + metadata, plus human-readable .txt dumps.

Usage (from repo root):
    python verification/dump_vectors.py --bits GS_data/umair_gs_bits.csv \
        --fecframe short --rate 1/2 \
        --mat-path s2xLDPCParityMatrices/dvbs2xLDPCParityMatrices.mat \
        --outdir verification/out

Defaults match your existing paths; override as needed.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import scipy.io

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tx._02_stream_adaptation import (
    get_kbch,
    pad_bbframe_rate,
    stream_adaptation_rate,
)
from tx._03_bch_encoding import bch_encode_bbframe, BCH_PARAMS
from tx._04_ldpc_Encoding import DVB_LDPC_Encoder


def _read_bits_file(path: str) -> np.ndarray:
    """Load a text file containing 0/1 characters (whitespace ignored)."""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    bits = [1 if ch == "1" else 0 for ch in data if ch in ("0", "1")]
    if not bits:
        raise ValueError(f"No bits found in file: {path}")
    return np.array(bits, dtype=np.uint8)


def _write_bitstring(path: str, bits: np.ndarray) -> None:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join("1" if b else "0" for b in bits))


def build_vectors(
    bits_in: np.ndarray,
    fecframe: str,
    rate: str,
    mat_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Run stream adaptation -> BCH -> LDPC and return stage outputs plus metadata.
    bits_in can be any length <= Kbch; it is padded then scrambled inside.
    """
    Kbch, Nbch, _t = BCH_PARAMS[(fecframe, rate)]
    if bits_in.size > Kbch:
        raise ValueError(f"Input length {bits_in.size} exceeds Kbch {Kbch}")

    padded = pad_bbframe_rate(bits_in, fecframe, rate)
    scrambled = stream_adaptation_rate(bits_in, fecframe, rate)

    bch_codeword = bch_encode_bbframe(scrambled, fecframe, rate)

    ldpc_encoder = DVB_LDPC_Encoder(mat_path)
    ldpc_codeword = ldpc_encoder.encode(bch_codeword, fecframe, rate)

    meta = {
        "fecframe": fecframe,
        "rate": rate,
        "Kbch": Kbch,
        "Nbch": Nbch,
        "nldpc": int(ldpc_codeword.size),
        "kldpc": int(bch_codeword.size),
        "input_bits": int(bits_in.size),
        "mat_path": os.path.abspath(mat_path),
    }
    return padded, scrambled, bch_codeword, ldpc_codeword, meta


def main():
    parser = argparse.ArgumentParser(description="Dump DVB-S2 BCH/LDPC vectors for MATLAB.")
    parser.add_argument("--bits", default="GS_data/umair_gs_bits.csv",
                        help="Path to input 0/1 text file (whitespace ignored).")
    parser.add_argument("--fecframe", default="short", choices=["normal", "short"])
    parser.add_argument("--rate", default="1/2", help="Code rate string, e.g., 1/2, 3/5.")
    parser.add_argument("--mat-path", default="s2xLDPCParityMatrices/dvbs2xLDPCParityMatrices.mat",
                        help="Path to DVB-S2 LDPC parity matrix .mat file.")
    parser.add_argument("--outdir", default="verification/out",
                        help="Directory to write MAT/TXT outputs.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    bits_path = args.bits
    if not os.path.isabs(bits_path):
        bits_path = os.path.join(ROOT, bits_path)
    if not os.path.isfile(bits_path):
        raise FileNotFoundError(f"Input bits file not found: {bits_path}")

    mat_path = args.mat_path
    if not os.path.isabs(mat_path):
        mat_path = os.path.join(ROOT, mat_path)
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"LDPC parity matrix .mat not found: {mat_path}")

    bits_in = _read_bits_file(bits_path)
    padded, scrambled, bch_codeword, ldpc_codeword, meta = build_vectors(
        bits_in, args.fecframe, args.rate, mat_path
    )

    out_mat = os.path.join(args.outdir, "vectors.mat")
    scipy.io.savemat(
        out_mat,
        {
            "raw_bits": bits_in,
            "padded_kbch": padded,
            "scrambled_kbch": scrambled,
            "bch_codeword": bch_codeword,
            "ldpc_codeword": ldpc_codeword,
            "meta": meta,
        },
    )

    _write_bitstring(os.path.join(args.outdir, "raw_bits.txt"), bits_in)
    _write_bitstring(os.path.join(args.outdir, "padded_kbch.txt"), padded)
    _write_bitstring(os.path.join(args.outdir, "scrambled_kbch.txt"), scrambled)
    _write_bitstring(os.path.join(args.outdir, "bch_codeword.txt"), bch_codeword)
    _write_bitstring(os.path.join(args.outdir, "ldpc_codeword.txt"), ldpc_codeword)

    print(f"Wrote vectors to {out_mat}")
    print(f"Metadata: {meta}")


if __name__ == "__main__":
    main()
