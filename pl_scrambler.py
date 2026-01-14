# pl_scrambler.py
# =============================================================================
# DVB-S2 PL Scrambler (ETSI EN 302 307 V1.3.1)
#
# Manual basis (Figure 14 / PL Scrambling):
#   - Build two m-sequences x and y of degree 18:
#       x: primitive polynomial 1 + x^7 + x^18
#          init: x(0)=1, x(1)=...=x(17)=0
#          rec : x(i+18) = x(i+7) + x(i) (mod 2)
#
#       y: polynomial 1 + y^5 + y^7 + y^10 + y^18
#          init: y(0)=...=y(17)=1
#          rec : y(i+18) = y(i+10)+y(i+7)+y(i+5)+y(i) (mod 2)
#
#   - Gold sequence z_n for scrambling code number n:
#       z_n(i) = [ x((i+n) mod (2^18-1)) + y(i) ] mod 2, i=0..2^18-2
#
#   - Convert to integer sequence R_n (values 0..3):
#       R_n(i) = 2*z_n((i+131072) mod (2^18-1)) + z_n(i), i=0..66419
#
#   - Complex scrambler:
#       C_n(i) = exp(j * R_n(i) * pi/2)
#       where R=0->1, R=1->j, R=2->-1, R=3->-j
#
# Usage:
#   - For a PLFRAME, scrambling is RESET after PLHEADER (90 symbols).
#   - The first symbol AFTER PLHEADER is multiplied by C_n(0), next by C_n(1), ...
# =============================================================================

from __future__ import annotations
import numpy as np
from functools import lru_cache
from typing import Tuple


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
_MSEQ_LEN = (1 << 18) - 1          # 2^18 - 1 = 262143
_R_OFFSET = 1 << 17               # 131072
_R_MAX_I  = 66419                 # i = 0..66419 inclusive => 66420 symbols


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _as_complex_1d(x: np.ndarray, name: str = "symbols") -> np.ndarray:
    a = np.asarray(x).reshape(-1)
    if not np.issubdtype(a.dtype, np.complexfloating):
        a = a.astype(np.complex128, copy=False)
    return a


def _validate_scrambling_code(n: int) -> int:
    if not isinstance(n, (int, np.integer)):
        raise TypeError("scrambling_code must be an int")
    n = int(n)
    # Standard defines n in [0, 2^18-2]
    if n < 0 or n > (_MSEQ_LEN - 1):
        raise ValueError(f"scrambling_code n must be in [0, {_MSEQ_LEN-1}], got {n}")
    return n


# -----------------------------------------------------------------------------
# m-sequence builders (degree 18)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _mseq_x() -> np.ndarray:
    """
    x sequence with polynomial 1 + x^7 + x^18 (degree 18), length 2^18-1.
    x(0)=1, x(1..17)=0
    x(i+18)=x(i+7)+x(i) mod2
    """
    L = _MSEQ_LEN
    x = np.zeros(L, dtype=np.uint8)
    x[0] = 1
    # generate up to L-1; recurrence defined for i=0..(2^18-20)
    for i in range(L - 18):
        x[i + 18] = (x[i + 7] ^ x[i])  # mod2 add == XOR for bits
    return x


@lru_cache(maxsize=1)
def _mseq_y() -> np.ndarray:
    """
    y sequence with polynomial 1 + y^5 + y^7 + y^10 + y^18 (degree 18), length 2^18-1.
    y(0..17)=1
    y(i+18)=y(i+10)+y(i+7)+y(i+5)+y(i) mod2
    """
    L = _MSEQ_LEN
    y = np.ones(L, dtype=np.uint8)
    # generate
    for i in range(L - 18):
        y[i + 18] = y[i + 10] ^ y[i + 7] ^ y[i + 5] ^ y[i]
    return y


# -----------------------------------------------------------------------------
# Gold sequence / R_n / C_n
# -----------------------------------------------------------------------------
def gold_zn_bits(scrambling_code: int, start: int = 0, length: int = None) -> np.ndarray:
    """
    Return z_n(i) bits for i = start..start+length-1 (modulo period 2^18-1).

    z_n(i) = x((i+n) mod (2^18-1)) XOR y(i)
    """
    n = _validate_scrambling_code(scrambling_code)
    if start < 0:
        raise ValueError("start must be >= 0")
    if length is None:
        length = _MSEQ_LEN
    if length < 0:
        raise ValueError("length must be >= 0")

    L = _MSEQ_LEN
    x = _mseq_x()
    y = _mseq_y()

    ii = (np.arange(start, start + length, dtype=np.int64) % L)
    zn = x[(ii + n) % L] ^ y[ii]
    return zn.astype(np.uint8)


def rn_sequence(scrambling_code: int, length: int = _R_MAX_I + 1, start: int = 0) -> np.ndarray:
    """
    Return integer sequence R_n(i) in {0,1,2,3} for i=start..start+length-1.

    R_n(i) = 2*z_n((i+131072) mod (2^18-1)) + z_n(i)

    Notes:
      - ETSI specifies R_n(i) for i=0..66419 (length 66420).
      - You may request shorter/longer lengths; indexing is modulo (2^18-1).
    """
    n = _validate_scrambling_code(scrambling_code)
    if start < 0:
        raise ValueError("start must be >= 0")
    if length < 0:
        raise ValueError("length must be >= 0")

    L = _MSEQ_LEN
    x = _mseq_x()
    y = _mseq_y()

    i0 = (np.arange(start, start + length, dtype=np.int64) % L)
    i1 = ((i0 + _R_OFFSET) % L)

    z0 = x[(i0 + n) % L] ^ y[i0]
    z1 = x[(i1 + n) % L] ^ y[i1]

    R = (2 * z1 + z0).astype(np.uint8)  # 0..3
    return R


def cn_sequence(scrambling_code: int, length: int = _R_MAX_I + 1, start: int = 0) -> np.ndarray:
    """
    Return complex scrambler C_n(i) = exp(j * R_n(i) * pi/2) for i=start..start+length-1.

    Mapping:
      R=0 ->  1
      R=1 ->  1j
      R=2 -> -1
      R=3 -> -1j
    """
    R = rn_sequence(scrambling_code, length=length, start=start).astype(np.uint8)

    # LUT for exp(j*R*pi/2)
    lut = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128)
    return lut[R]


# -----------------------------------------------------------------------------
# Scrambling application
# -----------------------------------------------------------------------------
def pl_scramble_symbols(
    symbols_after_plheader: np.ndarray,
    scrambling_code: int,
    start_index: int = 0,
) -> np.ndarray:
    """
    Scramble complex symbols AFTER PLHEADER by multiplying with C_n(i).

    Parameters
    ----------  
    symbols_after_plheader : np.ndarray (complex), shape (N,)
        Symbols starting immediately after PLHEADER (i.e., first symbol uses C_n(0)).
        If you want to scramble a later portion, use start_index.
    scrambling_code : int
        n in [0, 2^18-2] per ETSI.
    start_index : int
        i offset into the scrambler (default 0). Useful for chunked processing.

    Returns
    -------
    scrambled_symbols : np.ndarray complex128, shape (N,)
    """
    s = _as_complex_1d(symbols_after_plheader, "symbols_after_plheader")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")

    c = cn_sequence(scrambling_code, length=s.size, start=start_index)
    return s * c


# -----------------------------------------------------------------------------
# Convenience: scramble full PLFRAME (keep PLHEADER untouched)
# -----------------------------------------------------------------------------
def pl_scramble_full_plframe(
    plframe_symbols: np.ndarray,
    scrambling_code: int,
    plheader_len: int = 90,
) -> np.ndarray:
    """
    Scramble a full PLFRAME symbol vector while leaving the PLHEADER (first 90 symbols) unchanged.
    Scrambling RESET occurs after PLHEADER: symbol at index plheader_len uses C_n(0).

    Parameters
    ----------
    plframe_symbols : np.ndarray complex
        Full PLFRAME symbols including PLHEADER at the beginning.
    scrambling_code : int
        n in [0, 2^18-2].
    plheader_len : int
        PLHEADER length in symbols (90 in DVB-S2).

    Returns
    -------
    out : np.ndarray complex128
    """
    s = _as_complex_1d(plframe_symbols, "plframe_symbols")
    if plheader_len < 0 or plheader_len > s.size:
        raise ValueError("plheader_len out of range")

    out = s.copy()
    if s.size > plheader_len:
        out[plheader_len:] = pl_scramble_symbols(out[plheader_len:], scrambling_code, start_index=0)
    return out


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------
def _self_test() -> None:
    # Basic lengths
    assert _mseq_x().size == _MSEQ_LEN
    assert _mseq_y().size == _MSEQ_LEN

    # R_n and C_n default length per ETSI clause (66420)
    R0 = rn_sequence(0)
    C0 = cn_sequence(0)
    assert R0.size == _R_MAX_I + 1
    assert C0.size == _R_MAX_I + 1
    assert np.all((R0 >= 0) & (R0 <= 3))

    # Scramble should be unit-magnitude rotations
    assert np.allclose(np.abs(C0), 1.0)

    # Full PLFRAME helper sanity
    x = np.ones(200, dtype=np.complex128) * (1 + 1j)
    y = pl_scramble_full_plframe(x, scrambling_code=0, plheader_len=90)
    assert np.allclose(y[:90], x[:90])  # header untouched
    assert not np.allclose(y[90:], x[90:])  # scrambled part changes generally

    print("pl_scrambler.py self-test PASSED")


if __name__ == "__main__":
    _self_test()
