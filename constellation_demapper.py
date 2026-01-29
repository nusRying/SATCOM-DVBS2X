# constellation_demapper.py
# =============================================================================
# DVB-S2 constellation soft-demapper (LLRs). Currently supports QPSK.
# Noise model: complex AWGN with variance noise_var = E[|n|^2].
# For unit-energy constellations, noise_var = 1 / EsN0_linear.
# =============================================================================

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from bit_interleaver import BITS_PER_SYMBOL


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


def dvbs2_constellation_demapper(
    symbols: np.ndarray,
    modulation: str,
    noise_var: Optional[float] = None,
    esn0_db: Optional[float] = None,
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
        raise NotImplementedError(f"{mod} demapper not yet implemented")

    meta = {
        "modulation": mod,
        "noise_var": float(_resolve_noise_var(noise_var, esn0_db)),
        "symbols_in": int(np.asarray(symbols).size),
        "bits_per_symbol": int(BITS_PER_SYMBOL[mod]),
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
