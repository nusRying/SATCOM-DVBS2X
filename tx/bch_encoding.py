# Stage 04 - BCH Encoding (outer BCH code)
# bch_encoder.py
import numpy as np

# ============================================================
# DVB-S2 BCH parameters (Tables 5a/5b)
# ============================================================

# Mapping: (fecframe, rate_str) -> (Kbch, Nbch, t)
# Note: Nbch == kldpc in DVB-S2 tables
BCH_PARAMS = {
    # -------- NORMAL (nldpc = 64800) --------
    ("normal", "1/4"):  (16008, 16200, 12),
    ("normal", "1/3"):  (21408, 21600, 12),
    ("normal", "2/5"):  (25728, 25920, 12),
    ("normal", "1/2"):  (32208, 32400, 12),
    ("normal", "3/5"):  (38688, 38880, 12),
    ("normal", "2/3"):  (43040, 43200, 10),
    ("normal", "3/4"):  (48408, 48600, 12),
    ("normal", "4/5"):  (51648, 51840, 12),
    ("normal", "5/6"):  (53840, 54000, 10),
    ("normal", "8/9"):  (57472, 57600, 8),
    ("normal", "9/10"): (58192, 58320, 8),

    # -------- SHORT (nldpc = 16200) --------
    ("short", "1/4"):  (3072,  3240,  12),
    ("short", "1/3"):  (5232,  5400,  12),
    ("short", "2/5"):  (6312,  6480,  12),
    ("short", "1/2"):  (7032,  7200,  12),
    ("short", "3/5"):  (9552,  9720,  12),
    ("short", "2/3"):  (10632, 10800, 12),
    ("short", "3/4"):  (11712, 11880, 12),
    ("short", "4/5"):  (12432, 12600, 12),
    ("short", "5/6"):  (13152, 13320, 12),
    ("short", "8/9"):  (14232, 14400, 12),
}

# ============================================================
# BCH polynomials (Tables 6a / 6b)
# Represent each gi(x) by a list of exponents where coeff = 1
# ============================================================

# Table 6a (normal, degree 16)
G_NORMAL = [
    [0, 2, 3, 5, 16],                                    # g1
    [0, 1, 4, 5, 6, 8, 16],                              # g2
    [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 16],                # g3
    [0, 2, 4, 6, 9, 11, 12, 14, 16],                     # g4
    [0, 1, 2, 3, 5, 8, 9, 10, 11, 12, 16],               # g5
    [0, 2, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16],       # g6
    [0, 2, 5, 6, 8, 9, 10, 11, 13, 15, 16],              # g7
    [0, 1, 2, 5, 6, 8, 9, 12, 13, 14, 16],               # g8
    [0, 5, 7, 9, 10, 11, 16],                            # g9
    [0, 1, 2, 5, 7, 8, 10, 12, 13, 14, 16],              # g10
    [0, 2, 3, 5, 9, 11, 12, 13, 16],                     # g11
    [0, 1, 5, 6, 7, 9, 11, 12, 16],                      # g12
]

# Table 6b (short, degree 14)
G_SHORT = [
    [0, 1, 3, 5, 14],                                    # g1
    [0, 6, 8, 11, 14],                                   # g2
    [0, 1, 2, 6, 9, 10, 14],                             # g3
    [0, 4, 7, 8, 10, 12, 14],                            # g4
    [0, 2, 4, 6, 8, 9, 11, 13, 14],                      # g5
    [0, 3, 7, 8, 9, 13, 14],                             # g6
    [0, 2, 5, 6, 7, 10, 11, 13, 14],                     # g7
    [0, 5, 8, 9, 10, 11, 14],                            # g8
    [0, 1, 2, 3, 9, 10, 14],                             # g9
    [0, 3, 6, 9, 11, 12, 14],                            # g10
    [0, 4, 11, 12, 14],                                  # g11
    [0, 1, 2, 3, 5, 6, 7, 8, 10, 13, 14],                # g12
]

# ============================================================
# Polynomial helpers over GF(2)
# ============================================================

def exps_to_poly_int(exps):
    """Convert exponent list to integer bitmask poly (bit i => x^i)."""
    p = 0
    for e in exps:
        p ^= (1 << e)
    return p

def poly_deg(p: int) -> int:
    return p.bit_length() - 1

def poly_mul(a: int, b: int) -> int:
    """GF(2) polynomial multiply."""
    res = 0
    x = a
    y = b
    shift = 0
    while y:
        if y & 1:
            res ^= (x << shift)
        y >>= 1
        shift += 1
    return res

def poly_mod(dividend: int, divisor: int) -> int:
    """GF(2) polynomial remainder dividend % divisor."""
    if divisor == 0:
        raise ValueError("divisor polynomial cannot be 0")
    dd = poly_deg(divisor)
    r = dividend
    while r != 0 and poly_deg(r) >= dd:
        shift = poly_deg(r) - dd
        r ^= (divisor << shift)
    return r

def bits_msb_to_poly_int(bits: np.ndarray) -> int:
    """MSB-first bits -> polynomial int where MSB is x^(L-1)."""
    p = 0
    L = len(bits)
    for i, b in enumerate(bits):
        if int(b) & 1:
            p ^= (1 << (L - 1 - i))
    return p

def poly_int_to_bits_msb(p: int, L: int) -> np.ndarray:
    """Polynomial int -> MSB-first bits length L."""
    out = np.zeros(L, dtype=np.uint8)
    for i in range(L):
        out[i] = (p >> (L - 1 - i)) & 1
    return out

# ============================================================
# Generator polynomial g(x) for given fecframe and t
# ============================================================

def dvbs2_bch_generator_poly(fecframe: str, t: int) -> int:
    """Multiply first t polynomials from Table 6a/6b."""
    if fecframe not in ("normal", "short"):
        raise ValueError("fecframe must be 'normal' or 'short'")
    table = G_NORMAL if fecframe == "normal" else G_SHORT
    if not (1 <= t <= len(table)):
        raise ValueError(f"t must be 1..{len(table)}")

    g = 1  # start with '1'
    for i in range(t):
        gi = exps_to_poly_int(table[i])
        g = poly_mul(g, gi)
    return g

# ============================================================
# DVB-S2 systematic BCH encoder (Clause 5.3.1)
# ============================================================

def bch_encode_bbframe(bbframe_kbch_bits: np.ndarray, fecframe: str, rate: str) -> np.ndarray:
    """
    Systematic BCH outer encoding:
      input:  BBFRAME (Kbch bits)
      output: [BBFRAME || BCHFEC] with length Nbch bits

    Uses Table 5a/5b (Kbch, Nbch, t) and Table 6a/6b polynomials.
    """
    key = (fecframe, rate)
    if key not in BCH_PARAMS:
        raise ValueError(f"Unsupported (fecframe, rate) = {key}")

    Kbch, Nbch, t = BCH_PARAMS[key]

    if len(bbframe_kbch_bits) != Kbch:
        raise ValueError(f"Expected BBFRAME length Kbch={Kbch}, got {len(bbframe_kbch_bits)}")

    n_minus_k = Nbch - Kbch  # parity bits count

    # generator polynomial g(x)
    g = dvbs2_bch_generator_poly(fecframe, t)
    if poly_deg(g) != n_minus_k:
        raise RuntimeError(
            f"Generator degree mismatch: deg(g)={poly_deg(g)} but (Nbch-Kbch)={n_minus_k}"
        )

    # message polynomial m(x)
    m = bits_msb_to_poly_int(bbframe_kbch_bits)

    # dividend = m(x) * x^(n-k)
    dividend = m << n_minus_k

    # remainder d(x)
    d = poly_mod(dividend, g)

    # codeword c(x) = dividend + d(x)
    c = dividend ^ d

    # output bits (Nbch length, MSB first)
    return poly_int_to_bits_msb(c, Nbch)
