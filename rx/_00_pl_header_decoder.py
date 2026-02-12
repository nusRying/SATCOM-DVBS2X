"""
Stage 11 - PLHEADER demod + PLSC decode (SOF + MODCOD/TYPE).

This module recovers the 90 PLHEADER bits from the first 90 symbols of a
PLFRAME (π/2-BPSK), verifies the Start Of Frame (SOF) pattern, and decodes the
64-bit Physical Layer Signalling Code (PLSC) to extract MODCOD, FECFRAME type,
and pilots flag. It does not estimate the scrambling sequence index `n`
because DVB-S2 does not signal it in the PLHEADER (defaults to n=0 in this
receiver).
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache
from typing import Any, Dict

from tx._05_pl_header import (
    SOF_BITS_26,
    PLS_SCRAMBLE_64,
    rm_1_5_encode,
    _int_to_bits_msb,
    validate_modcod,
)

PLHEADER_LEN_SYMS = 90  # DVB-S2 PLHEADER length (symbols and bits)

_REF_ODD = (1.0 + 1.0j) / np.sqrt(2.0)   # maps to y=0 on odd positions
_REF_EVEN = (-1.0 + 1.0j) / np.sqrt(2.0)  # maps to y=0 on even positions
_REFS = np.empty(PLHEADER_LEN_SYMS, dtype=np.complex128)
_REFS[0::2] = _REF_ODD
_REFS[1::2] = _REF_EVEN


def _as_bits(x: np.ndarray, name: str) -> np.ndarray:
    b = np.asarray(x).reshape(-1)
    if b.dtype == np.bool_:
        b = b.astype(np.uint8)
    elif b.dtype != np.uint8:
        b = b.astype(np.uint8, copy=False)
    if not np.all((b == 0) | (b == 1)):
        raise ValueError(f"{name} must be 0/1 bits")
    return b


def demap_plheader_bits(plheader_syms: np.ndarray) -> np.ndarray:
    """
    Hard-decision demap π/2-BPSK PLHEADER symbols -> 90 bits.
    Uses matched projections to the expected constellation for odd/even indices.
    """
    s = np.asarray(plheader_syms, dtype=np.complex128).reshape(-1)
    if s.size < PLHEADER_LEN_SYMS:
        raise ValueError(f"Expected at least {PLHEADER_LEN_SYMS} PLHEADER symbols, got {s.size}")
    s90 = s[:PLHEADER_LEN_SYMS]
    proj = np.real(s90 * np.conj(_REFS))
    bits = (proj < 0).astype(np.uint8)
    return bits


@lru_cache(maxsize=1)
def _rm_codebook() -> list[Dict[str, Any]]:
    """
    Precompute all 2^6 RM(1,5) codewords (32 bits) for MODCOD(5) + TYPE_MSB(1).
    """
    book: list[Dict[str, Any]] = []
    for modcod in range(32):
        mod_bits = _int_to_bits_msb(modcod, 5)
        for type_msb in (0, 1):
            u6 = np.concatenate([mod_bits, np.array([type_msb], dtype=np.uint8)])
            cw = rm_1_5_encode(u6)
            book.append(
                {
                    "modcod": modcod,
                    "type_msb": type_msb,
                    "y32": cw,
                }
            )
    return book


def _nearest_rm_codeword(y32: np.ndarray) -> Dict[str, Any]:
    """
    Return the closest RM(1,5) codeword (Hamming) to the 32-bit vector y32.
    """
    y = _as_bits(y32, "y32")
    best: Dict[str, Any] | None = None
    for entry in _rm_codebook():
        dist = int(np.count_nonzero(entry["y32"] != y))
        if best is None or dist < best["dist"]:
            best = {**entry, "dist": dist}
            if dist == 0:
                break
    assert best is not None
    return best


def decode_plsc(plsc_scrambled: np.ndarray) -> Dict[str, Any]:
    """
    Decode 64-bit scrambled PLSC -> signalling fields.

    Returns:
        {
            "plsc_bits_descrambled": ndarray(64),
            "type_lsb": 0/1 (pilots flag),
            "type_msb": 0/1 (fecframe: 0=normal,1=short),
            "pilots": bool,
            "fecframe": "normal"/"short",
            "modcod": int,
            "modulation": str,
            "code_rate": str,
            "rm_distance": int (Hamming to nearest RM codeword),
            "type_lsb_mismatches": int (Hamming against pair pattern),
            "pair_equal": int,
            "pair_invert": int,
        }
    """
    b = _as_bits(plsc_scrambled, "plsc_bits")
    if b.size != 64:
        raise ValueError(f"PLSC must be 64 bits, got {b.size}")

    descrambled = b ^ PLS_SCRAMBLE_64

    pair_equal = int(np.count_nonzero(descrambled[0::2] == descrambled[1::2]))
    pair_invert = 32 - pair_equal
    type_lsb = 0 if pair_equal >= pair_invert else 1  # 0: (y,y), 1: (y,~y)

    if type_lsb == 0:
        y32 = (descrambled[0::2] + descrambled[1::2] >= 1).astype(np.uint8)
        recon64 = np.repeat(y32, 2)
    else:
        y32 = (descrambled[0::2] + (1 - descrambled[1::2]) >= 1).astype(np.uint8)
        recon64 = np.empty(64, dtype=np.uint8)
        recon64[0::2] = y32
        recon64[1::2] = y32 ^ 1

    type_lsb_mismatches = int(np.count_nonzero(descrambled != recon64))

    cw = _nearest_rm_codeword(y32)
    modcod = cw["modcod"]
    type_msb = cw["type_msb"]

    info = validate_modcod(modcod, allow_reserved=True)
    fecframe = "short" if type_msb else "normal"
    pilots = bool(type_lsb)

    return {
        "plsc_bits_descrambled": descrambled,
        "type_lsb": type_lsb,
        "type_msb": type_msb,
        "pilots": pilots,
        "fecframe": fecframe,
        "modcod": modcod,
        "modulation": info.modulation,
        "code_rate": info.code_rate,
        "rm_distance": int(cw["dist"]),
        "type_lsb_mismatches": type_lsb_mismatches,
        "pair_equal": pair_equal,
        "pair_invert": pair_invert,
    }


def decode_plheader(plheader_syms: np.ndarray, sof_max_errors: int = 4) -> Dict[str, Any]:
    """
    Demap + decode PLHEADER.

    Args:
        plheader_syms: first 90 complex symbols of PLFRAME.
        sof_max_errors: allowable bit errors in the 26-bit SOF before raising.

    Returns dict merging demap + PLSC fields. Raises if SOF mismatch exceeds
    sof_max_errors.
    """
    bits = demap_plheader_bits(plheader_syms)
    sof_bits = bits[:26]
    plsc_bits_scrambled = bits[26:90]

    sof_errors = int(np.count_nonzero(sof_bits != SOF_BITS_26))
    sof_ok = sof_errors <= sof_max_errors
    if not sof_ok:
        raise ValueError(f"SOF mismatch: {sof_errors} errors (max {sof_max_errors})")

    plsc_fields = decode_plsc(plsc_bits_scrambled)

    return {
        "plheader_bits": bits,
        "sof_bits": sof_bits,
        "sof_errors": sof_errors,
        "sof_ok": sof_ok,
        "sof_max_errors": sof_max_errors,
        "plsc_bits_scrambled": plsc_bits_scrambled,
        **plsc_fields,
    }

