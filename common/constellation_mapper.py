# Stage 07 - Constellation Mapper (TX symbol mapping)
# constellation_mapper.py
# ============================================================
# DVB-S2 Constellation Mapper (ETSI EN 302 307-1)
#  - QPSK:  Fig. 9
#  - 8PSK:  Fig. 10
#  - 16APSK/32APSK: mapping tables + gamma from ETSI tables
#
# IMPORTANT FIX vs your version:
#  - APSK normalization is CONSTELLATION-level (manual compliant),
#    NOT data-dependent (mean over transmitted symbols).
# ============================================================

import numpy as np
from typing import Tuple, Optional

# ============================================================
# Utilities
# ============================================================

def _as_1d_bits(bits: np.ndarray, name="bits") -> np.ndarray:
    b = np.asarray(bits).reshape(-1)
    if b.dtype == np.bool_:
        b = b.astype(np.uint8, copy=False)
    elif b.dtype != np.uint8:
        b = b.astype(np.uint8, copy=False)

    u = np.unique(b)
    if not np.all((u == 0) | (u == 1)):
        raise ValueError(f"{name} must contain only 0/1. Unique: {u[:20]}")
    return b

def bits_to_int_msb(group: np.ndarray) -> int:
    v = 0
    for bit in group:
        v = (v << 1) | int(bit)
    return v

def group_bits_exact(bits: np.ndarray, m: int, name="bits") -> np.ndarray:
    bits = _as_1d_bits(bits, name)
    if bits.size % m != 0:
        raise ValueError(f"{name} length {bits.size} not divisible by m={m}")
    Ns = bits.size // m
    if Ns == 0:
        return np.zeros((0, m), dtype=np.uint8)
    return bits.reshape(Ns, m)

# ============================================================
# QPSK (ETSI Fig. 9)
# MSB = I, LSB = Q
# 00 -> +1 + j1
# 01 -> +1 - j1
# 10 -> -1 + j1
# 11 -> -1 - j1
# Normalized: divide by sqrt(2)
# ============================================================

def map_qpsk(bits: np.ndarray) -> np.ndarray:
    g = group_bits_exact(bits, 2, "qpsk_bits")
    if g.shape[0] == 0:
        return np.zeros((0,), dtype=np.complex128)

    idx = (g[:, 0] << 1) | g[:, 1]
    lut = {
        0b00:  1 + 1j,
        0b01:  1 - 1j,
        0b10: -1 + 1j,
        0b11: -1 - 1j,
    }
    syms = np.array([lut[int(i)] for i in idx], dtype=np.complex128)
    return syms / np.sqrt(2.0)

# ============================================================
# 8PSK (ETSI Fig. 10)
# Unit circle points at specified angles (degrees)
# ============================================================

ETSI_8PSK_ANGLE_DEG = {
    0b000: 45.0,
    0b001: 0.0,
    0b101: 315.0,
    0b111: 270.0,
    0b011: 225.0,
    0b010: 180.0,
    0b110: 135.0,
    0b100: 90.0,
}

def map_8psk(bits: np.ndarray) -> np.ndarray:
    g = group_bits_exact(bits, 3, "8psk_bits")
    if g.shape[0] == 0:
        return np.zeros((0,), dtype=np.complex128)

    angles = np.array([ETSI_8PSK_ANGLE_DEG[bits_to_int_msb(row)] for row in g], dtype=np.float64)
    return np.exp(1j * np.deg2rad(angles))  # already unit energy

# ============================================================
# APSK mapping tables (ETSI mapping figures)
# Use (ring, angle_deg) representation (easy + explicit)
# Rings:
#  - 16APSK: inner (4 pts), outer (12 pts)
#  - 32APSK: inner (4 pts), mid (12 pts), outer (16 pts)
# ============================================================

ETSI_16APSK_MAP = {
    0b1100: ("inner", 45.0),
    0b1110: ("inner", 135.0),
    0b1111: ("inner", 225.0),
    0b1101: ("inner", 315.0),
    0b0100: ("outer", 15.0),
    0b0000: ("outer", 45.0),
    0b1000: ("outer", 75.0),
    0b1010: ("outer", 105.0),
    0b0010: ("outer", 135.0),
    0b0110: ("outer", 165.0),
    0b0111: ("outer", 195.0),
    0b0011: ("outer", 225.0),
    0b1011: ("outer", 255.0),
    0b1001: ("outer", 285.0),
    0b0001: ("outer", 315.0),
    0b0101: ("outer", 345.0),
}

ETSI_32APSK_MAP = {
    0b10001: ("inner", 45.0),
    0b10101: ("inner", 135.0),
    0b10111: ("inner", 225.0),
    0b10011: ("inner", 315.0),
    0b10000: ("mid", 15.0),
    0b00000: ("mid", 45.0),
    0b00001: ("mid", 75.0),
    0b00101: ("mid", 105.0),
    0b00100: ("mid", 135.0),
    0b10100: ("mid", 165.0),
    0b10110: ("mid", 195.0),
    0b00110: ("mid", 225.0),
    0b00111: ("mid", 255.0),
    0b00011: ("mid", 285.0),
    0b00010: ("mid", 315.0),
    0b10010: ("mid", 345.0),
    0b11000: ("outer", 0.0),
    0b01000: ("outer", 22.5),
    0b11001: ("outer", 45.0),
    0b01001: ("outer", 67.5),
    0b01101: ("outer", 90.0),
    0b11101: ("outer", 112.5),
    0b01100: ("outer", 135.0),
    0b11100: ("outer", 157.5),
    0b11110: ("outer", 180.0),
    0b01110: ("outer", 202.5),
    0b11111: ("outer", 225.0),
    0b01111: ("outer", 247.5),
    0b01011: ("outer", 270.0),
    0b11011: ("outer", 292.5),
    0b01010: ("outer", 315.0),
    0b11010: ("outer", 337.5),
}

# ============================================================
# ETSI-compliant APSK normalization (constellation-level)
# For 16APSK:
#   avgE = (4*r1^2 + 12*r2^2)/16
# For 32APSK:
#   avgE = (4*r1^2 + 12*r2^2 + 16*r3^2)/32
# Normalize by sqrt(avgE)
# ============================================================

def _norm_16apsk_scale(gamma: float) -> float:
    r1 = 1.0
    r2 = float(gamma)
    avgE = (4*(r1**2) + 12*(r2**2)) / 16.0
    return np.sqrt(avgE)

def _norm_32apsk_scale(gamma1: float, gamma2: float) -> float:
    r1 = 1.0
    r2 = float(gamma1)
    r3 = float(gamma2)
    avgE = (4*(r1**2) + 12*(r2**2) + 16*(r3**2)) / 32.0
    return np.sqrt(avgE)

def map_16apsk(bits: np.ndarray, gamma: float) -> np.ndarray:
    g = group_bits_exact(bits, 4, "16apsk_bits")
    if g.shape[0] == 0:
        return np.zeros((0,), dtype=np.complex128)

    r1, r2 = 1.0, float(gamma)
    scale = _norm_16apsk_scale(gamma)

    syms = np.zeros((g.shape[0],), dtype=np.complex128)
    for i, row in enumerate(g):
        ring, ang = ETSI_16APSK_MAP[bits_to_int_msb(row)]
        r = r1 if ring == "inner" else r2
        syms[i] = (r * np.exp(1j * np.deg2rad(ang))) / scale
    return syms

def map_32apsk(bits: np.ndarray, gamma1: float, gamma2: float) -> np.ndarray:
    g = group_bits_exact(bits, 5, "32apsk_bits")
    if g.shape[0] == 0:
        return np.zeros((0,), dtype=np.complex128)

    r1, r2, r3 = 1.0, float(gamma1), float(gamma2)
    scale = _norm_32apsk_scale(gamma1, gamma2)

    syms = np.zeros((g.shape[0],), dtype=np.complex128)
    for i, row in enumerate(g):
        ring, ang = ETSI_32APSK_MAP[bits_to_int_msb(row)]
        if ring == "inner":
            r = r1
        elif ring == "mid":
            r = r2
        else:
            r = r3
        syms[i] = (r * np.exp(1j * np.deg2rad(ang))) / scale
    return syms

# ============================================================
# Default gamma tables (DVB-S2) used when you provide code_rate
# (These must match your manual tables; keep them centralized.)
# ============================================================

DEFAULT_GAMMA_16 = {
    "2/3": 3.15,
    "3/4": 2.85,
    "4/5": 2.75,
    "5/6": 2.70,
    "8/9": 2.60,
    "9/10": 2.57,
}

DEFAULT_GAMMA_32 = {
    "3/4": (2.84, 5.27),
    "4/5": (2.72, 4.87),
    "5/6": (2.64, 4.64),
    "8/9": (2.54, 4.33),
    "9/10": (2.53, 4.30),
}

# ============================================================
# Top-level mapper
# ============================================================

def dvbs2_constellation_map(
    interleaved_bits: np.ndarray,
    mod: str,
    code_rate: Optional[str] = None,
    apsk_gammas: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Map interleaved bits -> complex symbols (ETSI-compliant).

    Parameters
    ----------
    interleaved_bits : np.ndarray
        Output of DVB-S2 bit interleaver (or LDPC bits directly for QPSK)
    mod : str
        "QPSK", "8PSK", "16APSK", "32APSK"
    code_rate : str
        e.g., "3/4", used only to infer default gammas for APSK if apsk_gammas is None
    apsk_gammas : tuple
        For 16APSK: (gamma,)
        For 32APSK: (gamma1, gamma2)

    Returns
    -------
    syms : np.ndarray of complex128
    """
    mod = mod.upper()
    b = _as_1d_bits(interleaved_bits, "interleaved_bits")

    if mod == "QPSK":
        return map_qpsk(b)

    if mod == "8PSK":
        return map_8psk(b)

    if mod == "16APSK":
        if apsk_gammas is None:
            if code_rate is None or code_rate not in DEFAULT_GAMMA_16:
                raise ValueError("16APSK needs apsk_gammas or a supported code_rate")
            gamma = DEFAULT_GAMMA_16[code_rate]
        else:
            gamma = float(apsk_gammas[0])
        return map_16apsk(b, gamma)

    if mod == "32APSK":
        if apsk_gammas is None:
            if code_rate is None or code_rate not in DEFAULT_GAMMA_32:
                raise ValueError("32APSK needs apsk_gammas or a supported code_rate")
            gamma1, gamma2 = DEFAULT_GAMMA_32[code_rate]
        else:
            gamma1, gamma2 = float(apsk_gammas[0]), float(apsk_gammas[1])
        return map_32apsk(b, gamma1, gamma2)

    raise ValueError("Unsupported modulation. Use QPSK, 8PSK, 16APSK, 32APSK")

# ============================================================
# Convenience helper: LDPC bits -> interleave -> map
# ============================================================

def dvbs2_map_from_ldpc_bits(
    ldpc_bits: np.ndarray,
    mod: str,
    code_rate: Optional[str] = None,
    apsk_gammas: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correct chain:
      LDPC bits -> bit interleaver (except QPSK) -> mapper

    Returns:
      (interleaved_bits, symbols)
    """
    from common.bit_interleaver import dvbs2_bit_interleave

    interleaved = dvbs2_bit_interleave(ldpc_bits, mod)
    symbols = dvbs2_constellation_map(
        interleaved,
        mod,
        code_rate=code_rate,
        apsk_gammas=apsk_gammas
    )
    return interleaved, symbols
