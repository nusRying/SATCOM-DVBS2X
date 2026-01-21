# bb_filter.py
# =============================================================================
# DVB-S2 Baseband Filter & Quadrature Modulation
# ETSI EN 302 307-1 — Clause 5.6
#
# - Root Raised Cosine (RRC) pulse shaping
# - Upsampling
# - I/Q baseband output
#
# Input  : complex PLFRAME symbols
# Output : complex baseband samples
# =============================================================================

import numpy as np
from typing import Tuple


# -----------------------------------------------------------------------------
# Root Raised Cosine (RRC) Filter
# -----------------------------------------------------------------------------

def rrc_filter(
    alpha: float,
    sps: int,
    span: int
) -> np.ndarray:
    """
    Generate Root Raised Cosine (RRC) FIR filter taps.

    Parameters
    ----------
    alpha : float
        Roll-off factor (0.20, 0.25, 0.35)
    sps : int
        Samples per symbol
    span : int
        Filter span in symbols (typically 8–12)

    Returns
    -------
    h : np.ndarray
        RRC filter coefficients
    """

    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")

    N = span * sps
    t = np.arange(-N / 2, N / 2 + 1) / sps
    h = np.zeros_like(t, dtype=np.float64)

    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - alpha + (4 * alpha / np.pi)
        elif abs(ti) == 1 / (4 * alpha):
            h[i] = (
                alpha / np.sqrt(2)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
                )
            )
        else:
            num = (
                np.sin(np.pi * ti * (1 - alpha))
                + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            )
            den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i] = num / den

    # Normalize energy
    h /= np.sqrt(np.sum(h ** 2))
    return h


# -----------------------------------------------------------------------------
# DVB-S2 BB Filter & Quadrature Modulation
# -----------------------------------------------------------------------------

def dvbs2_bb_filter(
    symbols: np.ndarray,
    alpha: float,
    sps: int = 4,
    span: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply DVB-S2 baseband filtering and quadrature modulation.

    Parameters
    ----------
    symbols : np.ndarray
        Complex PLFRAME symbols
    alpha : float
        Roll-off factor (0.20 / 0.25 / 0.35)
    sps : int
        Samples per symbol
    span : int
        RRC filter span (symbols)

    Returns
    -------
    samples : np.ndarray
        Complex baseband waveform
    rrc_taps : np.ndarray
        RRC filter taps (for receiver matched filter)
    """

    if symbols.ndim != 1:
        raise ValueError("symbols must be 1-D complex array")

    # -----------------------------
    # Upsample (zero insertion)
    # -----------------------------
    up = np.zeros(len(symbols) * sps, dtype=np.complex128)
    up[::sps] = symbols

    # -----------------------------
    # RRC pulse shaping
    # -----------------------------
    rrc_taps = rrc_filter(alpha, sps, span)
    samples = np.convolve(up, rrc_taps, mode="same")

    # -----------------------------
    # Normalize average power
    # -----------------------------
    power = np.mean(np.abs(samples) ** 2)
    samples /= np.sqrt(power)

    return samples, rrc_taps


# -----------------------------------------------------------------------------
# Self-Test
# -----------------------------------------------------------------------------

def _self_test():
    np.random.seed(0)

    # Random QPSK symbols
    symbols = (np.random.randn(1000) + 1j * np.random.randn(1000)) / np.sqrt(2)

    for alpha in [0.35, 0.25, 0.20]:
        samples, taps = dvbs2_bb_filter(
            symbols,
            alpha=alpha,
            sps=4,
            span=10
        )

        # Power should be ~1
        p = np.mean(np.abs(samples) ** 2)
        if not np.isclose(p, 1.0, atol=1e-2):
            raise AssertionError("Power normalization failed")

        assert samples.dtype == np.complex128
        assert taps.ndim == 1

    print("BB Filter self-test PASSED")


if __name__ == "__main__":
    _self_test()
