# pl_header.py
# =============================================================================
# DVB-S2 PLHEADER (ETSI EN 302 307 V1.3.1)
# Clause 5.5.2: SOF + PLSC generation and π/2-BPSK modulation
#
# PLHEADER bits: y1..y90  =  SOF(26 bits) || PLSC(64 bits)
# SOF: 0x18D2E82 (26 bits, MSB is y1)
# PLSC: (64,7) code built from RM(1,5) (32,6) + repetition/alternation + scramble
#
# This file includes:
# - strict MODCOD validation (Table 12)
# - strict TYPE bits handling (Table 12 note)
# - correct RM(1,5) encoding with MSB-first bit ordering
# - correct 64-bit PLS scrambler sequence
# - correct π/2-BPSK mapping of y1..y90 to 90 complex symbols
# =============================================================================

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _as_1d_bits(x: np.ndarray, name: str = "bits") -> np.ndarray:
    b = np.asarray(x).reshape(-1)
    if b.dtype == np.bool_:
        b = b.astype(np.uint8, copy=False)
    elif b.dtype != np.uint8:
        b = b.astype(np.uint8, copy=False)
    u = np.unique(b)
    if not np.all((u == 0) | (u == 1)):
        raise ValueError(f"{name} must be 0/1. Unique: {u[:20]}")
    return b

def _int_to_bits_msb(v: int, width: int) -> np.ndarray:
    if v < 0 or v >= (1 << width):
        raise ValueError(f"value {v} does not fit in {width} bits")
    s = format(v, f"0{width}b")
    return np.fromiter((1 if ch == "1" else 0 for ch in s), count=width, dtype=np.uint8)

def _bits_to_int_msb(b: np.ndarray) -> int:
    b = _as_1d_bits(b, "b")
    v = 0
    for bit in b:
        v = (v << 1) | int(bit)
    return v


# -----------------------------------------------------------------------------
# Table 12: MODCOD mapping / validity
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModcodInfo:
    modcod: int
    modulation: str   # QPSK / 8PSK / 16APSK / 32APSK / DUMMY
    code_rate: str    # "1/4", ... or "" for dummy/reserved
    efficiency: float # η (Table 12), informational only

# ETSI EN 302 307 V1.3.1 Table 12
# MODCOD 0..31. (29..31 reserved; 0 is dummy PLFRAME)
MODCOD_TABLE: Dict[int, ModcodInfo] = {
    0:  ModcodInfo(0,  "DUMMY",  "",     0.0000),
    1:  ModcodInfo(1,  "QPSK",   "1/4",  0.4902),
    2:  ModcodInfo(2,  "QPSK",   "1/3",  0.6566),
    3:  ModcodInfo(3,  "QPSK",   "2/5",  0.7890),
    4:  ModcodInfo(4,  "QPSK",   "1/2",  0.9889),
    5:  ModcodInfo(5,  "QPSK",   "3/5",  1.1883),
    6:  ModcodInfo(6,  "QPSK",   "2/3",  1.3223),
    7:  ModcodInfo(7,  "QPSK",   "3/4",  1.4875),
    8:  ModcodInfo(8,  "QPSK",   "4/5",  1.5874),
    9:  ModcodInfo(9,  "QPSK",   "5/6",  1.6547),
    10: ModcodInfo(10, "QPSK",   "8/9",  1.7665),
    11: ModcodInfo(11, "QPSK",   "9/10", 1.7886),
    12: ModcodInfo(12, "8PSK",   "3/5",  2.2824),
    13: ModcodInfo(13, "8PSK",   "2/3",  2.4962),
    14: ModcodInfo(14, "8PSK",   "3/4",  2.6792),
    15: ModcodInfo(15, "8PSK",   "5/6",  2.7886),
    16: ModcodInfo(16, "8PSK",   "8/9",  2.9167),
    17: ModcodInfo(17, "8PSK",   "9/10", 2.9412),
    18: ModcodInfo(18, "16APSK", "2/3",  3.1623),
    19: ModcodInfo(19, "16APSK", "3/4",  3.5231),
    20: ModcodInfo(20, "16APSK", "4/5",  3.7288),
    21: ModcodInfo(21, "16APSK", "5/6",  3.8501),
    22: ModcodInfo(22, "16APSK", "8/9",  3.9956),
    23: ModcodInfo(23, "16APSK", "9/10", 4.0274),
    24: ModcodInfo(24, "32APSK", "3/4",  4.4365),
    25: ModcodInfo(25, "32APSK", "4/5",  4.6410),
    26: ModcodInfo(26, "32APSK", "5/6",  4.7592),
    27: ModcodInfo(27, "32APSK", "8/9",  4.9167),
    28: ModcodInfo(28, "32APSK", "9/10", 4.9524),
    29: ModcodInfo(29, "RESERVED", "",   -1.0),
    30: ModcodInfo(30, "RESERVED", "",   -1.0),
    31: ModcodInfo(31, "RESERVED", "",   -1.0),
}

def validate_modcod(modcod: int, allow_reserved: bool = False) -> ModcodInfo:
    if modcod not in MODCOD_TABLE:
        raise ValueError(f"MODCOD must be 0..31, got {modcod}")
    info = MODCOD_TABLE[modcod]
    if info.modulation == "RESERVED" and not allow_reserved:
        raise ValueError(f"MODCOD {modcod} is RESERVED by ETSI Table 12")
    return info


def modcod_from_modulation_rate(modulation: str, code_rate: str) -> int:
    """
    Resolve MODCOD from modulation and code rate per ETSI Table 12.
    """
    mod = modulation.strip().upper()
    rate = code_rate.strip()
    for modcod, info in MODCOD_TABLE.items():
        if info.modulation.upper() == mod and info.code_rate == rate:
            if info.modulation == "RESERVED":
                continue
            return modcod
    raise ValueError(f"No MODCOD for modulation={modulation}, rate={code_rate}")


# -----------------------------------------------------------------------------
# Clause 5.5.2.1: SOF (26 bits)
# -----------------------------------------------------------------------------

# SOF = 18D2E82_hex (26 bits). MSB is the leftmost PLHEADER bit y1.
SOF_HEX = 0x18D2E82
SOF_BITS_26 = _int_to_bits_msb(SOF_HEX, 26)  # y1..y26


# -----------------------------------------------------------------------------
# Clause 5.5.2.2: PLSC encoding
# -----------------------------------------------------------------------------
# RM(1,5) generator matrix G (6 x 32) per ETSI (the classic first-order RM)
# Rows correspond to MSB-first of the 6-bit input u = [MODCOD(5 bits MSB->LSB), TYPE_MSB]
# y_rm = sum_{k=1..6} u_k * G_k  (mod 2)
#
# Then build 64 bits:
#  - if TYPE_LSB == 0: output (y1 y1 y2 y2 ... y32 y32)
#  - if TYPE_LSB == 1: output (y1 ~y1 y2 ~y2 ... y32 ~y32)
#
# Then XOR with 64-bit scrambling sequence (MSB-first) from ETSI.
# -----------------------------------------------------------------------------

RM15_G = np.array(
[
    [0,1]*16,                        # Row 1
    [0,0,1,1]*8,                     # Row 2
    [0,0,0,0,1,1,1,1]*4,             # Row 3
    [0]*8 + [1]*8 + [0]*8 + [1]*8,   # Row 4
    [0]*16 + [1]*16,                 # Row 5
    [1]*32                           # Row 6
],
    dtype=np.uint8
)

# 64-bit PLS scrambler sequence from ETSI (MSB-first)
PLS_SCRAMBLE_64 = np.fromiter(
    (1 if c == "1" else 0 for c in "0111000110011101100000111100100101010011010000100010110111111010"),
    count=64,
    dtype=np.uint8
)

def rm_1_5_encode(u6_msb_first: np.ndarray) -> np.ndarray:
    """
    Encode 6 bits using RM(1,5) to 32 bits, MSB-first u mapping to rows of G as per ETSI.
    """
    u = _as_1d_bits(u6_msb_first, "u6")
    if u.size != 6:
        raise ValueError(f"RM(1,5) input must be 6 bits, got {u.size}")
    # y = u @ G mod 2
    y = (u.astype(np.uint8)[:, None] * RM15_G).sum(axis=0) & 1
    return y.astype(np.uint8)

def plsc_bits(modcod: int, fecframe: str, pilots: bool, allow_reserved_modcod: bool = False) -> np.ndarray:
    """
    Build the 64-bit scrambled PLSC (y27..y90) from MODCOD + TYPE (2 bits).
    TYPE[1] (MSB): FECFRAME size (0=normal 64800, 1=short 16200)
    TYPE[0] (LSB): Pilots (0=no pilots, 1=pilots)
    """
    info = validate_modcod(modcod, allow_reserved=allow_reserved_modcod)

    fecframe = fecframe.strip().lower()
    if fecframe not in ("normal", "short"):
        raise ValueError("fecframe must be 'normal' or 'short'")

    type_msb = 0 if fecframe == "normal" else 1
    type_lsb = 1 if bool(pilots) else 0

    # MODCOD is 5 bits MSB-first
    modcod_bits = _int_to_bits_msb(modcod, 5)
    u6 = np.concatenate([modcod_bits, np.array([type_msb], dtype=np.uint8)])  # 6 bits MSB-first

    y32 = rm_1_5_encode(u6)  # 32 bits

    # Build 64 bits per TYPE_LSB rule
    out64 = np.empty(64, dtype=np.uint8)
    if type_lsb == 0:
        # y1 y1 y2 y2 ... y32 y32
        out64[0::2] = y32
        out64[1::2] = y32
    else:
        # y1 ~y1 y2 ~y2 ... y32 ~y32
        out64[0::2] = y32
        out64[1::2] = (y32 ^ 1)

    # Scramble
    scrambled = out64 ^ PLS_SCRAMBLE_64

    # Extra strictness: if MODCOD is dummy, pilots should be 0 in practice (optional).
    if info.modulation == "DUMMY" and pilots:
        raise ValueError("Dummy PLFRAME (MODCOD=0) must not request pilots (TYPE_LSB must be 0)")

    return scrambled


# -----------------------------------------------------------------------------
# Clause 5.5.2: PLHEADER bits and π/2-BPSK modulation
# -----------------------------------------------------------------------------
# ETSI rule (for i = 1..45):
# I_{2i-1} = Q_{2i-1} = (1/2) * (1 - 2*y_{2i-1})
# I_{2i}   = -Q_{2i}  = -(1/2) * (1 - 2*y_{2i})
# -----------------------------------------------------------------------------

def plheader_bits(modcod: int, fecframe: str, pilots: bool,
                  allow_reserved_modcod: bool = False) -> np.ndarray:
    """
    Return the 90 PLHEADER bits y1..y90 (uint8), MSB-first (y1 first).
    """
    plsc64 = plsc_bits(modcod, fecframe, pilots, allow_reserved_modcod=allow_reserved_modcod)
    y90 = np.concatenate([SOF_BITS_26, plsc64]).astype(np.uint8)
    if y90.size != 90:
        raise RuntimeError(f"Internal error: PLHEADER size is {y90.size}, expected 90")
    return y90

def plheader_pi_over_2_bpsk_symbols(y90: np.ndarray) -> np.ndarray:
    """
    Map PLHEADER bits y1..y90 to 90 π/2-BPSK complex symbols per ETSI formula.
    Output: complex128 array length 90.
    """
    y = _as_1d_bits(y90, "y90")
    if y.size != 90:
        raise ValueError(f"PLHEADER must be 90 bits, got {y.size}")

    s = np.empty(90, dtype=np.complex128)
    a = (1.0 - 2.0*y) / np.sqrt(2.0)


    # Odd indices (1-based): I=Q= a
    # Even indices (1-based): I=-a, Q=+a  (since I = -Q = -(1/2)*(1-2*y))
    # Careful with 0-based indexing:
    #   k=0 -> y1 (odd), k=1 -> y2 (even), ...
    for k in range(90):
        if (k % 2) == 0:
            # odd (1-based)
            s[k] = a[k] + 1j * a[k]
        else:
            # even (1-based)
            s[k] = (-a[k]) + 1j * (a[k])

    return s

def build_plheader(modcod: int, fecframe: str, pilots: bool,
                   allow_reserved_modcod: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience: returns (y90_bits, s90_symbols).
    """
    y90 = plheader_bits(modcod, fecframe, pilots, allow_reserved_modcod=allow_reserved_modcod)
    s90 = plheader_pi_over_2_bpsk_symbols(y90)
    return y90, s90


# -----------------------------------------------------------------------------
# Self-test / sanity checks
# -----------------------------------------------------------------------------

def _self_test() -> None:
    # Basic compliance checks
    assert SOF_BITS_26.size == 26
    assert PLS_SCRAMBLE_64.size == 64
    assert RM15_G.shape == (6, 32)

    # Try a few known MODCODs
    for modcod in [0, 1, 4, 11, 12, 23, 28]:
        for fecframe in ["normal", "short"]:
            for pilots in [False, True]:
                if modcod == 0 and pilots:
                    # should raise (strict)
                    try:
                        _ = build_plheader(modcod, fecframe, pilots)
                        raise AssertionError("Expected error for dummy+pilots")
                    except ValueError:
                        pass
                    continue

                y90, s90 = build_plheader(modcod, fecframe, pilots)
                assert y90.size == 90
                assert s90.size == 90
                # symbols should have |s| = sqrt((1/2)^2 + (1/2)^2) = sqrt(1/2)
                mag = np.abs(s90)
                if not np.allclose(mag, np.sqrt(0.5), atol=1e-12):
                    raise AssertionError("π/2-BPSK symbols not expected magnitude")

    print("PLHEADER self-test PASSED")

if __name__ == "__main__":
    _self_test()
