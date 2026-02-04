# Stage 14 - Pilot Phase Correction (CPE removal)
# pilot_phase_correction.py
# =============================================================================
# Pilot-aided common phase error (CPE) correction for DVB-S2 payload symbols.
# Uses the known π/2-BPSK pilot blocks inserted every 16 slots (ETSI 5.5.3).
# =============================================================================

from __future__ import annotations
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from typing import Dict, Any, Tuple

from common.pilot_insertion import (
    SLOT_LEN,
    PILOT_BLOCK_LEN,
    PILOT_PERIOD_SLOTS,
    PILOT_SYMBOLS_36,
    n_slots_for_fecframe,
)


def estimate_pilot_phase(pilot_block: np.ndarray) -> float:
    """
    Return the estimated common phase (radians) of one 36-symbol pilot block.
    """
    # Rotate by known pilot to isolate channel-induced phase, then average.
    ref = PILOT_SYMBOLS_36[0]
    rotated = pilot_block * np.conjugate(ref)
    ph = np.angle(np.mean(rotated))
    return float(ph)


def apply_pilot_phase_correction(
    payload_data: np.ndarray,
    pilots: np.ndarray,
    fecframe: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply per-pilot common phase correction to payload symbols.

    Parameters
    ----------
    payload_data : np.ndarray complex, shape (N,)
        Data-only payload symbols (no pilots, no PLHEADER).
    pilots : np.ndarray complex, shape (Npilots, 36)
        Extracted pilot blocks corresponding to this payload.
    fecframe : str
        'normal' or 'short'

    Returns
    -------
    corrected : np.ndarray complex128
        Phase-corrected payload symbols.
    phases : np.ndarray float64, shape (Npilots,)
        Estimated phase (radians) per pilot block.
    meta : dict
        Details of mapping and chunk sizes.
    """
    data = np.asarray(payload_data, dtype=np.complex128).reshape(-1)
    p = np.asarray(pilots, dtype=np.complex128)

    nslots = n_slots_for_fecframe(fecframe)
    chunk_len = PILOT_PERIOD_SLOTS * SLOT_LEN  # symbols between pilots
    expected_pilots = (nslots - 1) // PILOT_PERIOD_SLOTS if nslots > 0 else 0

    if p.shape[0] != expected_pilots:
        raise ValueError(f"Expected {expected_pilots} pilot blocks for fecframe={fecframe}, got {p.shape[0]}")

    phases = np.array([estimate_pilot_phase(pb) for pb in p], dtype=np.float64)

    corrected = data.copy()
    idx = 0
    for k, phi in enumerate(phases):
        seg_end = idx + chunk_len
        corrected[idx:seg_end] *= np.exp(-1j * phi)
        idx = seg_end

    # Tail after final pilot uses last phase estimate (common practice)
    if idx < corrected.size and phases.size > 0:
        corrected[idx:] *= np.exp(-1j * phases[-1])

    meta = {
        "fecframe": fecframe,
        "chunk_len": int(chunk_len),
        "pilot_blocks": int(expected_pilots),
        "payload_symbols": int(corrected.size),
    }
    return corrected, phases, meta


def _self_test() -> None:
    """
    Simple sanity: apply known phase to data+pilots and verify removal.
    """
    rng = np.random.default_rng(0)
    for fec in ("short", "normal"):
        nslots = n_slots_for_fecframe(fec)
        data_len = nslots * SLOT_LEN
        chunk_len = PILOT_PERIOD_SLOTS * SLOT_LEN
        npil = (nslots - 1) // PILOT_PERIOD_SLOTS

        data = (rng.standard_normal(data_len) + 1j * rng.standard_normal(data_len)).astype(np.complex128)
        pilots = np.tile(PILOT_SYMBOLS_36, (npil, 1))

        # Impose a linear phase drift over pilot blocks
        phis = np.linspace(0.1, 0.4, npil)
        data_noisy = data.copy()
        pilots_noisy = pilots.copy()

        idx = 0
        for k, phi in enumerate(phis):
            data_noisy[idx:idx + chunk_len] *= np.exp(1j * phi)
            pilots_noisy[k, :] *= np.exp(1j * phi)
            idx += chunk_len
        if idx < data_noisy.size:
            data_noisy[idx:] *= np.exp(1j * phis[-1])

        corrected, est_phis, _ = apply_pilot_phase_correction(data_noisy, pilots_noisy, fec)

        assert np.allclose(corrected, data, atol=1e-9)
        assert np.allclose(est_phis, phis, atol=1e-3)

    print("pilot_phase_correction.py self-test PASSED")


if __name__ == "__main__":
    _self_test()
