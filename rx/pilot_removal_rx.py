# Stage 13 - Pilot Removal (RX)
# pilot_removal_rx.py
# =============================================================================
# DVB-S2 Pilot Removal (Receiver) - ETSI EN 302 307 V1.3.1 Clause 5.5.3
#
# Input:  PLFRAME after PL descrambling (complex symbols)
# Output: payload data-only symbols (no pilots) + extracted pilot blocks
#
# Notes:
# - PLHEADER is NOT part of pilot structure (first 90 symbols).
# - Pilot blocks (36 symbols) are inserted after every 16 slots of 90 data symbols.
# - For short FECFRAME: 90 slots => 5 pilot blocks
# - For normal FECFRAME: 360 slots => 22 pilot blocks
# =============================================================================

from __future__ import annotations
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
from typing import Tuple, Dict, Any

from common.pilot_insertion import (
    SLOT_LEN,
    PILOT_BLOCK_LEN,
    PILOT_PERIOD_SLOTS,
    n_slots_for_fecframe,
)

PLHEADER_LEN_SYMS = 90  # DVB-S2 PLHEADER length in symbols


def _as_1d_complex(x: np.ndarray, name: str = "symbols") -> np.ndarray:
    s = np.asarray(x).reshape(-1)
    if not np.issubdtype(s.dtype, np.complexfloating):
        s = s.astype(np.complex128, copy=False)
    return s


def expected_payload_with_pilots_len(fecframe: str) -> int:
    """Length of (payload+pilots) excluding PLHEADER."""
    nslots = n_slots_for_fecframe(fecframe)
    if nslots <= 0:
        return 0
    n_pil = (nslots - 1) // PILOT_PERIOD_SLOTS
    data_len = nslots * SLOT_LEN
    return data_len + n_pil * PILOT_BLOCK_LEN


def _pilot_positions(nslots: int) -> np.ndarray:
    """
    Return slot indices after which pilots are inserted.
    For nslots=90 => [16, 32, 48, 64, 80]
    """
    if nslots <= 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(PILOT_PERIOD_SLOTS, nslots, PILOT_PERIOD_SLOTS, dtype=np.int64)


def remove_pilots_from_plframe(
    rx_plframe: np.ndarray,
    fecframe: str,
    plheader_len: int = PLHEADER_LEN_SYMS,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Remove pilot blocks from a DVB-S2 PLFRAME (after descrambling).

    Parameters
    ----------
    rx_plframe : np.ndarray complex, shape (N,)
        Full PLFRAME symbols including PLHEADER at the beginning.
        Must already be PL-DESCRAMBLED.
    fecframe : str
        "normal" or "short"
    plheader_len : int
        PLHEADER length in symbols (default 90 for DVB-S2).

    Returns
    -------
    payload_data : np.ndarray complex128
        Payload symbols with pilots removed (data-only), length = Nslots*90
    pilots : np.ndarray complex128
        Extracted pilot blocks, shape = (Npilot_blocks, 36)
    meta : dict
        Useful metadata and sanity info.
    """
    x = _as_1d_complex(rx_plframe, "rx_plframe")

    if plheader_len < 0 or plheader_len > x.size:
        raise ValueError("plheader_len out of range")

    payload_with_pilots = x[plheader_len:]

    expected_len = expected_payload_with_pilots_len(fecframe)
    if payload_with_pilots.size != expected_len:
        raise ValueError(
            f"(payload+pilots) length mismatch: expected {expected_len} "
            f"for fecframe={fecframe}, got {payload_with_pilots.size}. "
            f"(Full PLFRAME expected {plheader_len + expected_len} symbols)"
        )

    # Derive structure
    nslots = n_slots_for_fecframe(fecframe)
    n_pil = (nslots - 1) // PILOT_PERIOD_SLOTS if nslots > 0 else 0
    data_needed = nslots * SLOT_LEN

    # Allocate outputs
    payload_data = np.empty(data_needed, dtype=np.complex128)
    pilots = (
        np.empty((n_pil, PILOT_BLOCK_LEN), dtype=np.complex128)
        if n_pil > 0
        else np.empty((0, PILOT_BLOCK_LEN), dtype=np.complex128)
    )

    chunk_len = PILOT_PERIOD_SLOTS * SLOT_LEN
    in_idx = 0
    out_idx = 0

    # Process each pilot block
    for k in range(n_pil):
        payload_data[out_idx:out_idx + chunk_len] = payload_with_pilots[in_idx:in_idx + chunk_len]
        in_idx += chunk_len
        out_idx += chunk_len

        pilots[k, :] = payload_with_pilots[in_idx:in_idx + PILOT_BLOCK_LEN]
        in_idx += PILOT_BLOCK_LEN

    # Remainder after last pilot block
    if out_idx < data_needed:
        payload_data[out_idx:] = payload_with_pilots[in_idx:in_idx + (data_needed - out_idx)]

    meta = {
        "fecframe": fecframe,
        "plheader_len": int(plheader_len),
        "plframe_len_total": int(x.size),
        "payload_with_pilots_len": int(payload_with_pilots.size),
        "payload_data_len": int(payload_data.size),
        "S_slots": int(nslots),
        "pilot_blocks": int(n_pil),
        "slot_len": int(SLOT_LEN),
        "pilot_block_len": int(PILOT_BLOCK_LEN),
        "pilot_slots_after": _pilot_positions(nslots),
    }
    return payload_data, pilots, meta


def _self_test() -> None:
    # Round-trip test against TX insertion (structure only)
    from common.pilot_insertion import insert_pilots  # inserts pilots into data-only payload

    rng = np.random.default_rng(0)
    for fec in ["short", "normal"]:
        nslots = n_slots_for_fecframe(fec)
        data_len = nslots * SLOT_LEN

        payload = (rng.standard_normal(data_len) + 1j * rng.standard_normal(data_len)).astype(np.complex128)

        payload_with_pilots = insert_pilots(payload)
        plheader = (rng.standard_normal(PLHEADER_LEN_SYMS) + 1j * rng.standard_normal(PLHEADER_LEN_SYMS)).astype(np.complex128)

        rx_plframe = np.concatenate([plheader, payload_with_pilots])

        out_payload, pilots, meta = remove_pilots_from_plframe(rx_plframe, fecframe=fec)

        assert out_payload.size == data_len
        assert np.allclose(out_payload, payload)
        assert meta["pilot_blocks"] == pilots.shape[0]

    print("pilot_removal_rx.py self-test PASSED")


if __name__ == "__main__":
    _self_test()
