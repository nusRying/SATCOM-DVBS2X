# Stage 08 - Pilot Insertion (insert pilot blocks into PLFRAME)
# pilot_insertion.py
# =============================================================================
# DVB-S2 Pilot Insertion (ETSI EN 302 307 V1.3.1)
# Clause 5.5.3: Pilot insertion in PLFRAME
#
# A PLFRAME is organized into SLOTS of 90 symbols.
# When pilots are enabled (PILOT=1 in TYPE field), a pilot block of 36 symbols
# is inserted after every 16 slots of data.
#
# Structure (data-only slots):
#   data: Nslots * 90 symbols
#
# Structure (with pilots):
#   after slots 16, 32, 48, ... insert 36 pilot symbols.
#
# For DVB-S2:
#   Normal FECFRAME:  Nslots = 360
#   Short  FECFRAME:  Nslots = 90
#
# Pilot blocks count:
#   Normal: floor(360/16) = 22 pilot blocks
#   Short : floor(90/16)  = 5 pilot blocks
#
# =============================================================================

from __future__ import annotations
import numpy as np
from typing import Tuple


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SLOT_LEN = 90
PILOT_BLOCK_LEN = 36
PILOT_PERIOD_SLOTS = 16

# ETSI-defined pilot sequence (36 symbols).
# In DVB-S2, pilot symbols are a known π/2-BPSK sequence.
#
# ETSI-defined pilot symbols:
# All 36 pilot symbols are identical π/2-BPSK symbols:
#   I = +1/sqrt(2), Q = +1/sqrt(2)
#
# The pilots are constant (unmodulated) and are used by the receiver
# for carrier frequency, phase, and timing recovery.

# (Receiver correlates against this known periodic sequence.)
# NOTE: Per ETSI, pilots are constant pi/2-BPSK; all 36 symbols identical.
_P = 1.0 / np.sqrt(2.0)
PILOT_SYMBOLS_36 = np.full(
    (PILOT_BLOCK_LEN,),
    complex(_P, _P),
    dtype=np.complex128
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _as_1d_complex(x: np.ndarray, name: str = "symbols") -> np.ndarray:
    s = np.asarray(x).reshape(-1)
    if not np.issubdtype(s.dtype, np.complexfloating):
        s = s.astype(np.complex128, copy=False)
    return s

def n_slots_for_fecframe(fecframe: str) -> int:
    fecframe = fecframe.strip().lower()
    if fecframe == "normal":
        return 360
    if fecframe == "short":
        return 90
    raise ValueError("fecframe must be 'normal' or 'short'")

def expected_data_symbols(fecframe: str) -> int:
    return n_slots_for_fecframe(fecframe) * SLOT_LEN

def pilot_block_count(fecframe: str) -> int:
    nslots = n_slots_for_fecframe(fecframe)
    return (nslots - 1) // PILOT_PERIOD_SLOTS if nslots > 0 else 0

def expected_total_symbols_with_pilots(fecframe: str) -> int:
    return expected_data_symbols(fecframe) + pilot_block_count(fecframe) * PILOT_BLOCK_LEN


# -----------------------------------------------------------------------------
# Core: Pilot insertion
# -----------------------------------------------------------------------------

def insert_pilots(data_symbols: np.ndarray) -> np.ndarray:
    """
    Insert DVB-S2 pilot blocks into PLFRAME data symbols.

    Parameters
    ----------  
    data_symbols : np.ndarray (complex), length = Nslots*90
        Data symbols for the PLFRAME (output of constellation mapping),
        excluding PLHEADER and excluding pilots.

    Returns
    -------
    out : np.ndarray (complex)
        Data symbols with pilot blocks inserted.
    """
    s = _as_1d_complex(data_symbols, "data_symbols")
    if s.size % SLOT_LEN != 0:
        raise ValueError(
            f"data_symbols length must be a multiple of {SLOT_LEN}, got {s.size}"
        )
    nslots = s.size // SLOT_LEN
    needed = nslots * SLOT_LEN

    n_pil = (nslots - 1) // PILOT_PERIOD_SLOTS if nslots > 0 else 0
    if n_pil == 0:
        return s.copy()

    out_len = needed + n_pil * PILOT_BLOCK_LEN
    out = np.empty(out_len, dtype=np.complex128)

    in_idx = 0
    out_idx = 0

    # Process blocks of 16 slots (= 16*90 symbols), then insert pilots
    chunk_len = PILOT_PERIOD_SLOTS * SLOT_LEN

    for _ in range(n_pil):
        out[out_idx:out_idx + chunk_len] = s[in_idx:in_idx + chunk_len]
        in_idx += chunk_len
        out_idx += chunk_len

        out[out_idx:out_idx + PILOT_BLOCK_LEN] = PILOT_SYMBOLS_36
        out_idx += PILOT_BLOCK_LEN

    # Remainder slots after the last pilot insertion (if any)
    if in_idx < needed:
        out[out_idx:] = s[in_idx:]

    return out


def insert_pilots_into_payload(
    data_symbols: np.ndarray,
    pilots_on: bool,
    fecframe: str | None = None
) -> Tuple[np.ndarray, dict]:
    """
    Insert pilots if enabled and return (payload_symbols_out, meta_dict).
    """
    s = _as_1d_complex(data_symbols, "data_symbols")
    if s.size % SLOT_LEN != 0:
        raise ValueError(
            f"data_symbols length must be a multiple of {SLOT_LEN}, got {s.size}"
        )

    if fecframe is not None:
        expected = expected_data_symbols(fecframe)
        if s.size != expected:
            raise ValueError(
                f"data_symbols length must be {expected} for fecframe={fecframe}, got {s.size}"
            )

    nslots = s.size // SLOT_LEN
    n_pil = (nslots - 1) // PILOT_PERIOD_SLOTS if (pilots_on and nslots > 0) else 0

    if pilots_on:
        out = insert_pilots(s)
    else:
        out = s.copy()

    meta = {
        "pilots_on": bool(pilots_on),
        "S_slots": int(nslots),
        "pilot_blocks": int(n_pil),
        "pilot_symbol": PILOT_SYMBOLS_36[0],
        "payload_symbols_in": int(s.size),
        "payload_symbols_out": int(out.size),
    }
    return out, meta


def remove_pilots(rx_symbols: np.ndarray, fecframe: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Receiver-side helper: remove pilot blocks and return:
      (data_symbols, pilots_matrix)

    pilots_matrix shape = (Npilot_blocks, 36)
    """
    r = _as_1d_complex(rx_symbols, "rx_symbols")

    nslots = n_slots_for_fecframe(fecframe)
    n_pil = (nslots - 1) // PILOT_PERIOD_SLOTS if nslots > 0 else 0
    data_needed = nslots * SLOT_LEN

    total_needed = data_needed + n_pil * PILOT_BLOCK_LEN
    if r.size != total_needed:
        raise ValueError(
            f"rx_symbols length must be {total_needed} for fecframe={fecframe}, got {r.size}"
        )

    data = np.empty(data_needed, dtype=np.complex128)
    pilots = np.empty((n_pil, PILOT_BLOCK_LEN), dtype=np.complex128)

    chunk_len = PILOT_PERIOD_SLOTS * SLOT_LEN
    in_idx = 0
    data_idx = 0

    for k in range(n_pil):
        data[data_idx:data_idx + chunk_len] = r[in_idx:in_idx + chunk_len]
        in_idx += chunk_len
        data_idx += chunk_len

        pilots[k, :] = r[in_idx:in_idx + PILOT_BLOCK_LEN]
        in_idx += PILOT_BLOCK_LEN

    # remainder
    if data_idx < data_needed:
        data[data_idx:] = r[in_idx:]

    return data, pilots


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    rng = np.random.default_rng(0)

    for fec in ["normal", "short"]:
        n = expected_data_symbols(fec)
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)

        y = insert_pilots(x)
        assert y.size == expected_total_symbols_with_pilots(fec)

        x2, p = remove_pilots(y, fec)
        assert np.allclose(x, x2)
        assert p.shape[1] == PILOT_BLOCK_LEN
        # pilots equal the known pattern
        assert np.allclose(p[0], PILOT_SYMBOLS_36)

    print("Pilot insertion self-test PASSED")


if __name__ == "__main__":
    _self_test()
