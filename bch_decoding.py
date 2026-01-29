# bch_decoding.py
# =============================================================================
# Simple DVB-S2 BCH decoder (detect-only). It verifies the BCH parity using the
# same generator polynomial as the encoder and returns the first Kbch bits.
# For clean loopback tests (no noise after LDPC), this suffices.
# =============================================================================

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict

from bch_encoding import (
    BCH_PARAMS,
    dvbs2_bch_generator_poly,
    bits_msb_to_poly_int,
    poly_deg,
    poly_mod,
)


def bch_check_and_strip(
    codeword_bits: np.ndarray,
    fecframe: str,
    rate: str,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Verify BCH codeword and return payload (Kbch bits).
    Detect-only: raises ValueError on parity failure.
    """
    key = (fecframe, rate)
    if key not in BCH_PARAMS:
        raise ValueError(f"Unsupported (fecframe, rate) = {key}")
    Kbch, Nbch, t = BCH_PARAMS[key]

    cw = np.asarray(codeword_bits, dtype=np.uint8).reshape(-1)
    if cw.size != Nbch:
        raise ValueError(f"BCH codeword length {cw.size} != Nbch {Nbch}")

    g = dvbs2_bch_generator_poly(fecframe, t)
    if poly_deg(g) != (Nbch - Kbch):
        raise RuntimeError("Generator degree mismatch; check BCH_PARAMS.")

    # Syndrome (remainder) on full codeword
    c_poly = bits_msb_to_poly_int(cw)
    rem = poly_mod(c_poly, g)
    if rem != 0:
        raise ValueError("BCH parity check failed (syndrome != 0)")

    payload = cw[:Kbch].copy()
    meta = {"Kbch": Kbch, "Nbch": Nbch, "t": t}
    return payload, meta


def _self_test():
    # Encode-known valid codeword (all-zero + parity -> all-zero)
    fec = "short"
    rate = "1/2"
    Kbch, Nbch, _ = BCH_PARAMS[(fec, rate)]
    cw = np.zeros(Nbch, dtype=np.uint8)
    payload, meta = bch_check_and_strip(cw, fec, rate)
    assert payload.size == Kbch
    print("bch_decoding.py self-test PASSED")


if __name__ == "__main__":
    _self_test()
