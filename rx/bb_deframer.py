# Stage 19 - BB Deframer (extract DF bits after stream deadaptation)
# bb_deframer.py
# =============================================================================
# Parse DVB-S2 BBHEADER and extract DF from a Kbch-length BBFRAME (after
# descrambling). Checks CRC-8 on BBHEADER.
# =============================================================================

from __future__ import annotations
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from typing import Dict, Any, Tuple

from tx.BB_Frame import dvbs2_crc8
from tx.stream_adaptation import get_kbch


def _as_bits(x: np.ndarray, name: str = "bits") -> np.ndarray:
    b = np.asarray(x).reshape(-1)
    if b.dtype == np.bool_:
        b = b.astype(np.uint8, copy=False)
    elif b.dtype != np.uint8:
        b = b.astype(np.uint8, copy=False)
    u = np.unique(b)
    if not np.all((u == 0) | (u == 1)):
        raise ValueError(f"{name} must be 0/1 bits")
    return b


def parse_bbheader(bbheader_bits: np.ndarray) -> Dict[str, Any]:
    b = _as_bits(bbheader_bits, "bbheader_bits")
    if b.size != 80:
        raise ValueError(f"BBHEADER must be 80 bits, got {b.size}")

    matype1 = int("".join(str(x) for x in b[0:8]), 2)
    matype2 = int("".join(str(x) for x in b[8:16]), 2)
    upl = int("".join(str(x) for x in b[16:32]), 2)
    dfl = int("".join(str(x) for x in b[32:48]), 2)
    sync = int("".join(str(x) for x in b[48:56]), 2)
    syncd = int("".join(str(x) for x in b[56:72]), 2)
    crc_rx = int("".join(str(x) for x in b[72:80]), 2)

    crc_calc = dvbs2_crc8(b[:72])
    if crc_calc != crc_rx:
        raise ValueError(f"BBHEADER CRC mismatch: got {crc_rx:02X}, expected {crc_calc:02X}")

    return {
        "MATYPE1": matype1,
        "MATYPE2": matype2,
        "UPL": upl,
        "DFL": dfl,
        "SYNC": sync,
        "SYNCD": syncd,
        "CRC": crc_rx,
    }


def deframe_bb(scrambled_kbch: np.ndarray, fecframe: str, rate: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Input: descrambled BBFRAME of length Kbch (still contains padding).
    Output: (DF bits length = DFL, meta including header fields)
    """
    kbch = get_kbch(fecframe, rate)
    bits = _as_bits(scrambled_kbch, "scrambled_kbch")
    if bits.size != kbch:
        raise ValueError(f"descrambled BBFRAME length {bits.size} != Kbch {kbch}")

    header = bits[:80]
    df_with_pad = bits[80:]
    meta = parse_bbheader(header)
    dfl = meta["DFL"]
    if dfl > df_with_pad.size:
        raise ValueError(f"DFL {dfl} exceeds available payload {df_with_pad.size}")

    df_bits = df_with_pad[:dfl].copy()
    meta["padding_bits"] = int(df_with_pad.size - dfl)
    return df_bits, meta


def _self_test():
    # Build a dummy BBHEADER with DFL=10, UPL=0, CRC correct
    from BB_Frame import build_bbheader
    fec = "short"
    rate = "1/2"
    dfl = 10
    bbheader = build_bbheader(0, 0, upl=0, dfl=dfl, sync=0x47, syncd=0)
    df = np.zeros(dfl, dtype=np.uint8)
    kbch = get_kbch(fec, rate)
    pad = np.zeros(kbch - 80 - dfl, dtype=np.uint8)
    bbframe = np.concatenate([bbheader, df, pad])
    out_df, meta = deframe_bb(bbframe, fec, rate)
    assert out_df.size == dfl
    assert meta["DFL"] == dfl
    print("bb_deframer.py self-test PASSED")


if __name__ == "__main__":
    _self_test()
