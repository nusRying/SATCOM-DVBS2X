# Stage 15 - Constellation Demapper (LLRs)
# constellation_demapper.py
# =============================================================================
# DVB-S2 constellation soft-demapper (LLRs). Currently supports QPSK.
# Noise model: complex AWGN with variance noise_var = E[|n|^2].
# For unit-energy constellations, noise_var = 1 / EsN0_linear.
# =============================================================================

from __future__ import annotations

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from typing import Tuple, Optional, List

from common.bit_interleaver import BITS_PER_SYMBOL
from common.constellation_mapper import (
    DEFAULT_GAMMA_16,
    DEFAULT_GAMMA_32,
    map_qpsk,
    map_8psk,
    map_16apsk,
    map_32apsk,
)


def _resolve_noise_var(noise_var: Optional[float], esn0_db: Optional[float]) -> float:
    if noise_var is not None:
        if noise_var <= 0:
            raise ValueError("noise_var must be > 0")
        return float(noise_var)
    if esn0_db is None:
        raise ValueError("Provide noise_var or esn0_db")
    esn0_lin = 10 ** (esn0_db / 10.0)
    # For unit-energy symbols, E|n|^2 = 1 / EsN0
    return 1.0 / esn0_lin


def demap_qpsk_llr(
    symbols: np.ndarray,
    noise_var: Optional[float] = None,
    esn0_db: Optional[float] = None,
) -> np.ndarray:
    """
    Soft demap QPSK symbols to LLRs (Gray, MSB=I, LSB=Q).

    LLR per bit uses analytical expression for QPSK over AWGN:
        LLR_I = (2*sqrt(2)/noise_var) * Re(s)
        LLR_Q = (2*sqrt(2)/noise_var) * Im(s)
    where noise_var = E[|n|^2].
    """
    s = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    nv = _resolve_noise_var(noise_var, esn0_db)

    factor = (2.0 * np.sqrt(2.0)) / nv
    llr_i = factor * s.real
    llr_q = factor * s.imag

    out = np.empty(s.size * 2, dtype=np.float64)
    out[0::2] = llr_i
    out[1::2] = llr_q
    return out


# -----------------------------------------------------------------------------#
# Max-log generic demapper for PSK/APSK
# -----------------------------------------------------------------------------#

def _build_constellation(mod: str, code_rate: Optional[str], apsk_gammas: Optional[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (points, bits_labels) where:
      points: complex array length M
      bits_labels: uint8 array shape (M, m) MSB-first
    """
    mod = mod.upper()
    if mod == "QPSK":
        bits = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.uint8)
        pts = map_qpsk(bits.reshape(-1))
        return pts, bits
    if mod == "8PSK":
        bits = np.array([
            [0,0,0],
            [0,0,1],
            [1,0,1],
            [1,1,1],
            [0,1,1],
            [0,1,0],
            [1,1,0],
            [1,0,0],
        ], dtype=np.uint8)
        pts = map_8psk(bits.reshape(-1))
        return pts, bits
    if mod == "16APSK":
        if apsk_gammas is None:
            if code_rate is None or code_rate not in DEFAULT_GAMMA_16:
                raise ValueError("16APSK demap needs apsk_gammas or code_rate for gamma")
            gamma = DEFAULT_GAMMA_16[code_rate]
        else:
            gamma = apsk_gammas[0]
        # Build labels 0..15 with Gray mapping as in mapper dict order
        labels = list(range(16))
        bits = np.array([[ (v>>3)&1, (v>>2)&1, (v>>1)&1, v&1 ] for v in labels], dtype=np.uint8)
        pts = map_16apsk(bits.reshape(-1), gamma)
        return pts, bits
    if mod == "32APSK":
        if apsk_gammas is None:
            if code_rate is None or code_rate not in DEFAULT_GAMMA_32:
                raise ValueError("32APSK demap needs apsk_gammas or code_rate for gammas")
            gamma1, gamma2 = DEFAULT_GAMMA_32[code_rate]
        else:
            gamma1, gamma2 = apsk_gammas
        labels = list(range(32))
        bits = np.array([[ (v>>4)&1, (v>>3)&1, (v>>2)&1, (v>>1)&1, v&1 ] for v in labels], dtype=np.uint8)
        pts = map_32apsk(bits.reshape(-1), gamma1, gamma2)
        return pts, bits
    raise ValueError(f"Unsupported modulation '{mod}'")


def demap_maxlog(
    symbols: np.ndarray,
    mod: str,
    noise_var: Optional[float],
    esn0_db: Optional[float],
    code_rate: Optional[str] = None,
    apsk_gammas: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Max-log LLR demapper for PSK/APSK constellations.
    LLR_k = (1/nv) * (min_{b=0}|r-s|^2 - min_{b=1}|r-s|^2)
    """
    pts, labels = _build_constellation(mod, code_rate, apsk_gammas)
    nv = _resolve_noise_var(noise_var, esn0_db)

    r = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    m = labels.shape[1]
    llr = np.empty(r.size * m, dtype=np.float64)

    # Precompute distances per received symbol
    for i, sym in enumerate(r):
        d2 = np.abs(sym - pts)**2
        for k in range(m):
            mask1 = labels[:, k] == 1
            mask0 = ~mask1
            d1 = d2[mask1].min()
            d0 = d2[mask0].min()
            llr[i*m + k] = (d0 - d1) / nv
    return llr


def dvbs2_constellation_demapper(
    symbols: np.ndarray,
    modulation: str,
    noise_var: Optional[float] = None,
    esn0_db: Optional[float] = None,
    code_rate: Optional[str] = None,
    apsk_gammas: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Demap symbols -> bit LLRs for the given modulation.

    Returns
    -------
    llrs : np.ndarray float64
        Length = m * Ns, where m is bits/symbol.
    meta : dict
        Useful info (noise_var, modulation, symbols_in, bits_per_symbol)
    """
    mod = modulation.upper()
    if mod not in BITS_PER_SYMBOL:
        raise ValueError(f"Unsupported modulation '{mod}'")

    if mod == "QPSK":
        llrs = demap_qpsk_llr(symbols, noise_var=noise_var, esn0_db=esn0_db)
    else:
        llrs = demap_maxlog(
            symbols,
            mod,
            noise_var=noise_var,
            esn0_db=esn0_db,
            code_rate=code_rate,
            apsk_gammas=apsk_gammas,
        )

    meta = {
        "modulation": mod,
        "noise_var": float(_resolve_noise_var(noise_var, esn0_db)),
        "symbols_in": int(np.asarray(symbols).size),
        "bits_per_symbol": int(BITS_PER_SYMBOL[mod]),
        "code_rate": code_rate,
    }
    return llrs, meta


def _self_test() -> None:
    rng = np.random.default_rng(0)
    s = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex128) / np.sqrt(2)
    bits = np.array([0,0,0,1,1,0,1,1], dtype=np.uint8)  # pairs per symbol

    nv = 0.5  # arbitrary
    llr, meta = dvbs2_constellation_demapper(s, "QPSK", noise_var=nv)
    # Signs should match transmitted bits
    hard = (llr < 0).astype(np.uint8)
    assert hard.tolist() == bits.tolist()
    assert meta["bits_per_symbol"] == 2
    print("constellation_demapper.py self-test PASSED")


if __name__ == "__main__":
    _self_test()
