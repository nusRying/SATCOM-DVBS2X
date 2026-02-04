# Stage 12 - PL Descrambler (inverse of PL scrambling)
# pl_descrambler.py
# =============================================================================
# DVB-S2 PL Descrambler (inverse of PL scrambling)
#
# This mirrors pl_scrambler.py and uses the exact same C_n(i) generator so that
# y = s * C_n(i) at TX is inverted by s = y * conj(C_n(i)) at RX.
# =============================================================================

from __future__ import annotations
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np

from common.pl_scrambler import (
    cn_sequence,
    _as_complex_1d,
    _validate_scrambling_code,
)


def pl_descramble_symbols(
    symbols_after_plheader: np.ndarray,
    scrambling_code: int,
    start_index: int = 0,
) -> np.ndarray:
    """
    Descramble complex symbols AFTER PLHEADER by multiplying with conj(C_n(i)).

    Parameters
    ----------
    symbols_after_plheader : np.ndarray (complex), shape (N,)
        Symbols starting immediately after PLHEADER (i.e., first symbol uses C_n(0)).
        If you want to descramble a later portion, use start_index.
    scrambling_code : int
        n in [0, 2^18-2] per ETSI.
    start_index : int
        i offset into the scrambler (default 0). Useful for chunked processing.

    Returns
    -------
    descrambled_symbols : np.ndarray complex128, shape (N,)
    """
    s = _as_complex_1d(symbols_after_plheader, "symbols_after_plheader")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    _validate_scrambling_code(scrambling_code)

    c = cn_sequence(scrambling_code, length=s.size, start=start_index)
    return s * np.conjugate(c)


def pl_descramble_full_plframe(
    rx_plframe: np.ndarray,
    scrambling_code: int,
    plheader_len: int = 90,
) -> np.ndarray:
    """
    DVB-S2 PL descrambling (ETSI 5.5.4): inverse of PL scrambling.
    - PLHEADER (first plheader_len symbols) is NOT scrambled.
    - Everything after PLHEADER (payload + pilots) IS descrambled.

    Parameters
    ----------
    rx_plframe : np.ndarray complex
        Full PLFRAME symbols including PLHEADER at the beginning.
    scrambling_code : int
        n in [0, 2^18-2].
    plheader_len : int
        PLHEADER length in symbols (90 in DVB-S2).

    Returns
    -------
    out : np.ndarray complex128
    """
    x = _as_complex_1d(rx_plframe, "rx_plframe")
    if plheader_len < 0 or plheader_len > x.size:
        raise ValueError("plheader_len out of range")
    _validate_scrambling_code(scrambling_code)

    out = x.copy()
    if x.size > plheader_len:
        out[plheader_len:] = pl_descramble_symbols(
            out[plheader_len:],
            scrambling_code,
            start_index=0,
        )
    return out


def _self_test() -> None:
    # Round-trip sanity: descramble(scramble(x)) == x
    from common.pl_scrambler import pl_scramble_full_plframe

    rng = np.random.default_rng(0)
    x = rng.standard_normal(200) + 1j * rng.standard_normal(200)
    y = pl_scramble_full_plframe(x, scrambling_code=5, plheader_len=90)
    z = pl_descramble_full_plframe(y, scrambling_code=5, plheader_len=90)
    assert np.allclose(z, x)

    print("pl_descrambler.py self-test PASSED")


if __name__ == "__main__":
    _self_test()
