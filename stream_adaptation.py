import numpy as np

# ============================================================
#  FECFRAME lengths (ETSI EN 302 307-1)
# ============================================================

KBCH_NORMAL = 64800
KBCH_SHORT  = 16200


# ============================================================
#  Internal utility: ensure 1-D {0,1} uint8 bits
# ============================================================

def _as_1d_bits(x: np.ndarray, name: str = "bits") -> np.ndarray:
    """
    Force input into a clean 1-D uint8 array of 0/1 values.
    This prevents (N,1)/(1,N) shapes and object arrays that break logging/saving.
    """
    arr = np.asarray(x)

    # Flatten any (N,1), (1,N), etc.
    arr = arr.reshape(-1)

    # Convert booleans/ints to uint8
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8, copy=False)
    elif arr.dtype != np.uint8:
        # allow ints like int64, etc.
        arr = arr.astype(np.uint8, copy=False)

    # Enforce values are 0/1 (avoid 255 etc if something weird slipped in)
    # This is safe because XOR logic expects 0/1.
    uniq = np.unique(arr)
    if not np.all((uniq == 0) | (uniq == 1)):
        raise ValueError(f"{name} must contain only 0/1 bits. Found unique values: {uniq[:20]}")

    return arr


# ============================================================
#  Padding (Clause 5.2.1)
# ============================================================

def pad_bbframe(bbframe_bits: np.ndarray, fecframe: str) -> np.ndarray:
    """
    Append zero padding bits so that BBFRAME length becomes Kbch.
    Input must be a 1-D bit array: BBHEADER (80) + DF (DFL).
    """
    if fecframe not in ("normal", "short"):
        raise ValueError("fecframe must be 'normal' or 'short'")

    bbframe_bits = _as_1d_bits(bbframe_bits, "bbframe_bits")

    Kbch = KBCH_NORMAL if fecframe == "normal" else KBCH_SHORT
    current_len = int(bbframe_bits.size)

    if current_len > Kbch:
        raise ValueError(f"BBFRAME length ({current_len}) exceeds Kbch ({Kbch})")

    padding_len = Kbch - current_len
    if padding_len == 0:
        return bbframe_bits.copy()

    padding = np.zeros(padding_len, dtype=np.uint8)
    return np.concatenate([bbframe_bits, padding])


# ============================================================
#  BB Scrambler (Clause 5.2.2)
# ============================================================

def bb_scramble(bits: np.ndarray) -> np.ndarray:
    """
    Scramble BBFRAME bits using ETSI PRBS:
      Polynomial: 1 + x^14 + x^15
      Initial state: 100101010000000 (15 bits)
    Input must be 1-D bit array (length = Kbch).
    """
    bits = _as_1d_bits(bits, "bits")

    # Initial register (MSB first) per ETSI
    reg = [
        1, 0, 0, 1, 0,
        1, 0, 1, 0, 0,
        0, 0, 0, 0, 0
    ]

    scrambled = np.empty_like(bits)

    for i in range(bits.size):
        prbs_bit = reg[-1]                 # output bit
        scrambled[i] = bits[i] ^ prbs_bit  # XOR

        # Feedback = x^14 XOR x^15 (last two bits)
        feedback = reg[-1] ^ reg[-2]

        # Shift register: insert feedback at front
        reg = [feedback] + reg[:-1]

    return scrambled


# ============================================================
#  Stream Adaptation (Clause 5.2)
# ============================================================

def stream_adaptation(bbframe_bits: np.ndarray, fecframe: str) -> np.ndarray:
    """
    Complete Stream Adaptation:
      1) Padding to Kbch
      2) BB scrambling
    Returns a 1-D uint8 bit array of length Kbch.
    """
    padded = pad_bbframe(bbframe_bits, fecframe)
    scrambled = bb_scramble(padded)

    # Final safety: must be exactly Kbch length
    Kbch = KBCH_NORMAL if fecframe == "normal" else KBCH_SHORT
    if scrambled.size != Kbch:
        raise RuntimeError(f"Stream adaptation output length {scrambled.size} != Kbch {Kbch}")

    return scrambled


# ============================================================
#  Save BBFRAME to File (robust)
# ============================================================

def save_bbframe_to_file(
    original_bbframe: np.ndarray,
    scrambled_bbframe: np.ndarray,
    output_file: str,
    fecframe: str = "normal",
    bits_per_line: int = 80,
    output_mode: str = "full"
) -> None:
    """
    Save original (padded) and scrambled BBFRAME bits to a text file.
    Both should be length Kbch, 1-D, 0/1 bits.
    """

    if fecframe not in ("normal", "short"):
        raise ValueError("fecframe must be 'normal' or 'short'")

    Kbch = KBCH_NORMAL if fecframe == "normal" else KBCH_SHORT

    original_bbframe = _as_1d_bits(original_bbframe, "original_bbframe")
    scrambled_bbframe = _as_1d_bits(scrambled_bbframe, "scrambled_bbframe")

    if original_bbframe.size != Kbch:
        raise ValueError(f"Original BBFRAME must be Kbch={Kbch} bits, got {original_bbframe.size}")
    if scrambled_bbframe.size != Kbch:
        raise ValueError(f"Scrambled BBFRAME must be Kbch={Kbch} bits, got {scrambled_bbframe.size}")

    def _write_bitlines(f, arr: np.ndarray):
        s = "".join("1" if b else "0" for b in arr.tolist())
        for i in range(0, len(s), bits_per_line):
            f.write(s[i:i + bits_per_line] + "\n")

    with open(output_file, "w", encoding="utf-8") as f:
        if output_mode == "scrambled_single_line":
            s = "".join("1" if b else "0" for b in scrambled_bbframe.tolist())
            f.write(s)
            return
        if output_mode == "both_single_line":
            s_orig = "".join("1" if b else "0" for b in original_bbframe.tolist())
            s_scr = "".join("1" if b else "0" for b in scrambled_bbframe.tolist())
            f.write(s_orig + "\n" + s_scr)
            return

        f.write("=" * 70 + "\n")
        f.write("BBFRAME DATA - ORIGINAL (PADDED) vs SCRAMBLED\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"FECFRAME Type           : {fecframe.upper()}\n")
        f.write(f"Kbch                    : {Kbch} bits\n")
        f.write(f"Original (padded) length: {original_bbframe.size} bits\n")
        f.write(f"Scrambled length        : {scrambled_bbframe.size} bits\n\n")

        f.write("-" * 70 + "\n")
        f.write("ORIGINAL BBFRAME (PADDED, BEFORE SCRAMBLING)\n")
        f.write("-" * 70 + "\n")
        _write_bitlines(f, original_bbframe)

        f.write("\n" + "-" * 70 + "\n")
        f.write("SCRAMBLED BBFRAME (AFTER BB SCRAMBLING)\n")
        f.write("-" * 70 + "\n")
        _write_bitlines(f, scrambled_bbframe)

        f.write("\n" + "=" * 70 + "\n")


# ============================================================
#  DVB-S2 BCH parameters (ETSI EN 302 307-1, Tables 5a/5b)
#  (fecframe, rate_str) -> (Kbch, Nbch, t)
# ============================================================

BCH_PARAMS = {
    # -------- NORMAL (Nldpc = 64800) --------
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

    # -------- SHORT (Nldpc = 16200) --------
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


def get_kbch(fecframe: str, rate: str) -> int:
    key = (fecframe, rate)
    if key not in BCH_PARAMS:
        raise ValueError(f"Unsupported (fecframe, rate) = {key}")
    Kbch, _, _ = BCH_PARAMS[key]
    return Kbch

# ============================================================
#  Inverse operations (descramble + keep padding)
# ============================================================

def bb_descramble(scrambled_bits: np.ndarray) -> np.ndarray:
    """
    Inverse of bb_scramble (same PRBS, XOR again).
    Length must equal Kbch (padding kept).
    """
    scrambled_bits = _as_1d_bits(scrambled_bits, "scrambled_bits")

    # Same register as scrambler
    reg = [
        1, 0, 0, 1, 0,
        1, 0, 1, 0, 0,
        0, 0, 0, 0, 0
    ]

    out = np.empty_like(scrambled_bits)
    for i in range(scrambled_bits.size):
        prbs_bit = reg[-1]
        out[i] = scrambled_bits[i] ^ prbs_bit
        feedback = reg[-1] ^ reg[-2]
        reg = [feedback] + reg[:-1]
    return out


def stream_deadaptation_rate(scrambled_kbch: np.ndarray, fecframe: str, rate: str) -> np.ndarray:
    """
    Reverse of stream_adaptation_rate: descramble only, keeps padding.
    Caller is responsible for trimming padding using BBHEADER/DFL if needed.
    """
    Kbch = get_kbch(fecframe, rate)
    s = _as_1d_bits(scrambled_kbch, "scrambled_kbch")
    if s.size != Kbch:
        raise ValueError(f"scrambled_kbch length {s.size} != Kbch {Kbch}")
    return bb_descramble(s)


# ============================================================
#  Rate-aware padding (ETSI 5.2.1): pad to Kbch
# ============================================================

def pad_bbframe_rate(bbframe_bits: np.ndarray, fecframe: str, rate: str) -> np.ndarray:
    bbframe_bits = _as_1d_bits(bbframe_bits, "bbframe_bits")
    Kbch = get_kbch(fecframe, rate)

    current_len = int(bbframe_bits.size)
    if current_len > Kbch:
        raise ValueError(f"BBFRAME length ({current_len}) exceeds Kbch ({Kbch})")

    pad_len = Kbch - current_len
    if pad_len == 0:
        return bbframe_bits.copy()

    return np.concatenate([bbframe_bits, np.zeros(pad_len, dtype=np.uint8)])


# ============================================================
#  Rate-aware stream adaptation (ETSI 5.2)
# ============================================================

def stream_adaptation_rate(bbheader_plus_df: np.ndarray, fecframe: str, rate: str) -> np.ndarray:
    padded = pad_bbframe_rate(bbheader_plus_df, fecframe, rate)
    scrambled = bb_scramble(padded)

    Kbch = get_kbch(fecframe, rate)
    if scrambled.size != Kbch:
        raise RuntimeError(f"stream_adaptation_rate output {scrambled.size} != Kbch {Kbch}")

    return scrambled


# ============================================================
#  Save BBFRAME to File (rate-aware)
# ============================================================

def save_bbframe_to_file_rate(
    original_kbch: np.ndarray,
    scrambled_kbch: np.ndarray,
    output_file: str,
    fecframe: str,
    rate: str,
    bits_per_line: int = 80,
    output_mode: str = "full",
) -> None:
    original_kbch = _as_1d_bits(original_kbch, "original_kbch")
    scrambled_kbch = _as_1d_bits(scrambled_kbch, "scrambled_kbch")

    Kbch = get_kbch(fecframe, rate)
    if original_kbch.size != Kbch or scrambled_kbch.size != Kbch:
        raise ValueError("Both original and scrambled must be exactly Kbch bits")

    def _write_lines(f, arr: np.ndarray):
        s = "".join("1" if b else "0" for b in arr.tolist())
        for i in range(0, len(s), bits_per_line):
            f.write(s[i:i + bits_per_line] + "\n")

    with open(output_file, "w", encoding="utf-8") as f:
        if output_mode == "scrambled_single_line":
            s = "".join("1" if b else "0" for b in scrambled_kbch.tolist())
            f.write(s)
            return
        if output_mode == "both_single_line":
            s_orig = "".join("1" if b else "0" for b in original_kbch.tolist())
            s_scr = "".join("1" if b else "0" for b in scrambled_kbch.tolist())
            f.write(s_orig + "\n" + s_scr)
            return

        f.write("=" * 70 + "\n")
        f.write("DVB-S2 STREAM ADAPTATION OUTPUT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"fecframe: {fecframe}\n")
        f.write(f"rate    : {rate}\n")
        f.write(f"Kbch    : {Kbch}\n\n")

        f.write("-" * 70 + "\n")
        f.write("ORIGINAL (PADDED TO Kbch, BEFORE SCRAMBLING)\n")
        f.write("-" * 70 + "\n")
        _write_lines(f, original_kbch)

        f.write("\n" + "-" * 70 + "\n")
        f.write("SCRAMBLED (AFTER BB SCRAMBLING)\n")
        f.write("-" * 70 + "\n")
        _write_lines(f, scrambled_kbch)

        f.write("\n" + "=" * 70 + "\n")
