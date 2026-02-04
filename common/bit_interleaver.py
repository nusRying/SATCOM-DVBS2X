# Stage 06/16 - Bit Interleaver (TX interleave / RX deinterleave)
"""
# bit_interleaver.py
# ============================================================
# DVB-S2 Bit Interleaver
# ETSI EN 302 307-1, Clause 5.3.3
# ============================================================
"""

import numpy as np

# ------------------------------------------------------------
# Modulation → bits per symbol (m)
# ------------------------------------------------------------

BITS_PER_SYMBOL = {
    "QPSK": 2,
    "8PSK": 3,
    "16APSK": 4,
    "32APSK": 5,
}

# Variant for soft metrics (LLRs); uses same permutation but keeps dtype/values.
def dvbs2_llr_deinterleave(llr: np.ndarray, modulation: str) -> np.ndarray:
    modulation = modulation.upper()
    y = np.asarray(llr).reshape(-1)

    if modulation == "QPSK":
        return y

    if modulation not in BITS_PER_SYMBOL:
        raise ValueError(f"Unsupported modulation '{modulation}'")

    m = BITS_PER_SYMBOL[modulation]
    if y.size % m != 0:
        raise ValueError(
            f"Interleaved length {y.size} not divisible by bits-per-symbol {m}"
        )

    Ns = y.size // m
    out = np.empty_like(y, dtype=np.float64)
    for j in range(m):
        out[j::m] = y[j * Ns : (j + 1) * Ns]
    return out

# ------------------------------------------------------------
# Utility: enforce clean 1-D uint8 {0,1}
# ------------------------------------------------------------

def _as_1d_bits(bits: np.ndarray, name="bits") -> np.ndarray:
    arr = np.asarray(bits).reshape(-1)

    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8, copy=False)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)

    uniq = np.unique(arr)
    if not np.all((uniq == 0) | (uniq == 1)):
        raise ValueError(f"{name} must contain only 0/1, got {uniq}")

    return arr

# ------------------------------------------------------------
# DVB-S2 Bit Interleaver (Tx)
# ------------------------------------------------------------

def dvbs2_bit_interleave(ldpc_bits: np.ndarray, modulation: str) -> np.ndarray:
    """
    DVB-S2 bit interleaver (ETSI EN 302 307-1, Clause 5.3.3)

    Parameters
    ----------
    ldpc_bits : np.ndarray
        LDPC encoded bits (length = nldpc)
    modulation : str
        "QPSK", "8PSK", "16APSK", or "32APSK"

    Returns
    -------
    interleaved_bits : np.ndarray
        Interleaved bits (length = nldpc)
    """

    modulation = modulation.upper()
    bits = _as_1d_bits(ldpc_bits, "ldpc_bits")

    # QPSK: no bit interleaver
    if modulation == "QPSK":
        return bits

    if modulation not in BITS_PER_SYMBOL:
        raise ValueError(f"Unsupported modulation '{modulation}'")

    m = BITS_PER_SYMBOL[modulation]

    if bits.size % m != 0:
        raise ValueError(
            f"LDPC length {bits.size} not divisible by bits-per-symbol {m}"
        )

    # ETSI definition:
    # y = [c(0), c(m), c(2m), ...,
    #      c(1), c(m+1), c(2m+1), ...,
    #      ...
    #      c(m-1), c(2m-1), ...]
    return np.concatenate([bits[j::m] for j in range(m)])

# ------------------------------------------------------------
# DVB-S2 Bit De-Interleaver (Rx)
# ------------------------------------------------------------

def dvbs2_bit_deinterleave(interleaved_bits: np.ndarray, modulation: str) -> np.ndarray:
    """
    Inverse DVB-S2 bit interleaver.

    Parameters
    ----------
    interleaved_bits : np.ndarray
        Interleaved bits
    modulation : str
        "QPSK", "8PSK", "16APSK", or "32APSK"

    Returns
    -------
    ldpc_bits : np.ndarray
        Original LDPC bit order
    """

    modulation = modulation.upper()
    y = _as_1d_bits(interleaved_bits, "interleaved_bits")

    # QPSK: no interleaver
    if modulation == "QPSK":
        return y

    if modulation not in BITS_PER_SYMBOL:
        raise ValueError(f"Unsupported modulation '{modulation}'")

    m = BITS_PER_SYMBOL[modulation]

    if y.size % m != 0:
        raise ValueError(
            f"Interleaved length {y.size} not divisible by bits-per-symbol {m}"
        )

    Ns = y.size // m
    c = np.empty_like(y)

    for j in range(m):
        c[j::m] = y[j * Ns : (j + 1) * Ns]

    return c

# ------------------------------------------------------------
# Self-test (bit-exact)
# ------------------------------------------------------------

def _self_test():
    print("Running DVB-S2 bit interleaver self-test...")

    for mod, m in BITS_PER_SYMBOL.items():
        N = m * 10
        x = np.arange(N) % 2

        y = dvbs2_bit_interleave(x, mod)
        z = dvbs2_bit_deinterleave(y, mod)

        assert np.array_equal(x, z), f"FAILED for {mod}"
        print(f"  {mod}: OK")

    print("All DVB-S2 bit interleaver tests PASSED")

# ------------------------------------------------------------
# Run self-test
# ------------------------------------------------------------

if __name__ == "__main__":
    _self_test()
