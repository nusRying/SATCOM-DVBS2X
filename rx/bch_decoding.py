# Stage 18 - BCH Decoder (check and strip)
# bch_decoding.py
# =============================================================================
# BCH decoder (detect-only). Correction attempt deferred for stability.
# =============================================================================

from __future__ import annotations

import os
import sys
from typing import Tuple, Dict
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tx.bch_encoding import (
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

    # Syndrome
    c_poly = bits_msb_to_poly_int(cw)
    rem = poly_mod(c_poly, g)
    if rem != 0:
        raise ValueError("BCH parity check failed (syndrome != 0)")

    payload = cw[:Kbch].copy()
    meta = {"Kbch": Kbch, "Nbch": Nbch, "t": t, "corrected": False, "errors": 0}
    return payload, meta


def _self_test():
    fec = "short"
    rate = "1/2"
    Kbch, Nbch, _ = BCH_PARAMS[(fec, rate)]
    cw = np.zeros(Nbch, dtype=np.uint8)
    payload, meta = bch_check_and_strip(cw, fec, rate)
    assert payload.size == Kbch
    print("bch_decoding.py self-test PASSED (detect-only)")


if __name__ == "__main__":
    _self_test()
