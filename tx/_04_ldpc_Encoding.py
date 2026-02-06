# Stage 05 - LDPC Encoding (inner LDPC code)
# ldpc_encoder.py
import os
import re
import numpy as np

try:
    import scipy.io
except ImportError as e:
    raise ImportError(
        "scipy is required to load .mat parity matrices. Install with: pip install scipy"
    ) from e


# ------------------------------------------------------------
# DVB-S2 (and DVB-S2X-style naming) frame lengths
# ------------------------------------------------------------
NLDPC_NORMAL = 64800
NLDPC_SHORT  = 16200


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _as_1d_bits(x: np.ndarray, name: str = "bits") -> np.ndarray:
    arr = np.asarray(x).reshape(-1)

    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8, copy=False)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)

    uniq = np.unique(arr)
    if not np.all((uniq == 0) | (uniq == 1)):
        raise ValueError(f"{name} must contain only 0/1 bits. Found unique values: {uniq[:20]}")
    return arr


def _rate_to_key(rate: str) -> str:
    """
    Convert '1/2' -> '1_2', '3/5' -> '3_5', etc.
    """
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", rate)
    if not m:
        raise ValueError("rate must look like '1/2', '3/5', etc.")
    return f"{m.group(1)}_{m.group(2)}"


def _frame_to_suffix(fecframe: str) -> str:
    fecframe = fecframe.strip().lower()
    if fecframe == "normal":
        return "N"
    if fecframe == "short":
        return "S"
    raise ValueError("fecframe must be 'normal' or 'short'")


def _nldpc_from_fecframe(fecframe: str) -> int:
    return NLDPC_NORMAL if fecframe == "normal" else NLDPC_SHORT


# ------------------------------------------------------------
# LDPC Encoder class (precomputes row-wise connectivity)
# ------------------------------------------------------------
class DVB_LDPC_Encoder:
    """
    LDPC systematic encoder using H in sparse (row,col) form from .mat.

    The .mat is expected to contain arrays like:
      PT_1_2_N  -> (row, col) pairs, 1-based indices, for NORMAL rate 1/2
      PT_3_5_S  -> (row, col) pairs, 1-based indices, for SHORT  rate 3/5

    Encoding is done by solving:
      H * c^T = 0
      c = [u | p]
      => Hp * p = Hs * u   (mod 2)
    In DVB-S2 parity matrices, Hp is lower-triangular with 1s on diagonal,
    so parity p is computed by forward substitution.
    """

    def __init__(self, mat_path: str):
        if not os.path.isfile(mat_path):
            raise FileNotFoundError(f"MAT file not found: {mat_path}")

        self.mat_path = mat_path
        self._mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        self._cache = {}  # (fecframe, rate) -> prepared structure

    def available_codes(self):
        keys = [k for k in self._mat.keys() if k.startswith("PT_")]
        return sorted(keys)

    def _load_pairs(self, fecframe: str, rate: str) -> np.ndarray:
        rate_key = _rate_to_key(rate)
        suffix = _frame_to_suffix(fecframe)
        mat_key = f"PT_{rate_key}_{suffix}"

        if mat_key not in self._mat:
            raise KeyError(
                f"Parity table '{mat_key}' not found in MAT.\n"
                f"Available examples: {self.available_codes()[:20]}"
            )

        pairs = np.asarray(self._mat[mat_key])
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"{mat_key} must be Nx2 (row,col) pairs, got shape {pairs.shape}")

        # MATLAB is usually 1-based; keep as int64 for safety
        pairs = pairs.astype(np.int64, copy=False)
        return pairs

    def _prepare(self, fecframe: str, rate: str):
        fecframe = fecframe.strip().lower()
        rate = rate.strip()

        cache_key = (fecframe, rate)
        if cache_key in self._cache:
            return self._cache[cache_key]

        n = _nldpc_from_fecframe(fecframe)
        pairs = self._load_pairs(fecframe, rate)

        # Convert to 0-based
        rows = pairs[:, 0] - 1
        cols = pairs[:, 1] - 1

        if rows.min() < 0 or cols.min() < 0:
            raise ValueError("Found 0 or negative indices in parity table (expected 1-based).")

        m = int(rows.max() + 1)   # number of check equations
        n_from_table = int(cols.max() + 1)

        if n_from_table != n:
            raise ValueError(
                f"Table n={n_from_table} does not match fecframe expected nldpc={n}.\n"
                f"Check you used correct fecframe (normal/short) for this PT_* key."
            )

        k = n - m  # systematic bits length (kldpc)

        # Build row-wise lists for sys and parity parts
        sys_cols = [[] for _ in range(m)]
        par_cols = [[] for _ in range(m)]  # store parity-index (0..m-1), not absolute col

        # Sort pairs by row then col for deterministic behavior
        order = np.lexsort((cols, rows))
        rows_s = rows[order]
        cols_s = cols[order]

        for r, c in zip(rows_s, cols_s):
            if c < k:
                sys_cols[r].append(int(c))
            else:
                par_cols[r].append(int(c - k))  # parity index

        # --- Checks: Hp must be lower-triangular with diag 1s (DVB-S2 property) ---
        # For each row r, must contain parity index r on diagonal
        for r in range(m):
            if r not in par_cols[r]:
                raise RuntimeError(
                    f"Hp diagonal check failed: row {r} does not contain parity index {r}."
                )
            # no entries above diagonal allowed (parity_idx > row)
            if any(pidx > r for pidx in par_cols[r]):
                raise RuntimeError(
                    f"Hp triangular check failed: row {r} contains parity index > row."
                )

        # Convert to numpy arrays for faster XOR loops
        sys_cols_np = [np.array(v, dtype=np.int32) for v in sys_cols]
        # For parity, keep only strictly-lower part (exclude diagonal)
        par_lower_np = []
        for r in range(m):
            v = [p for p in par_cols[r] if p < r]
            par_lower_np.append(np.array(v, dtype=np.int32))

        prepared = {
            "n": n,
            "m": m,
            "k": k,
            "sys_cols": sys_cols_np,
            "par_lower": par_lower_np,
        }
        self._cache[cache_key] = prepared
        return prepared

    def encode(self, u_bits: np.ndarray, fecframe: str, rate: str) -> np.ndarray:
        """
        Input:
          u_bits length must be kldpc = nldpc - m
        Output:
          codeword c length = nldpc = [u | p]
        """
        prep = self._prepare(fecframe, rate)

        n = prep["n"]
        m = prep["m"]
        k = prep["k"]
        sys_cols = prep["sys_cols"]
        par_lower = prep["par_lower"]

        u = _as_1d_bits(u_bits, "u_bits")
        if u.size != k:
            raise ValueError(
                f"LDPC input length must be kldpc={k} for (fecframe={fecframe}, rate={rate}). "
                f"Got {u.size}."
            )

        # syndrome part s[r] = XOR of u over sys positions in row r
        s = np.zeros(m, dtype=np.uint8)
        for r in range(m):
            cols = sys_cols[r]
            if cols.size:
                # XOR reduce
                s[r] = np.bitwise_xor.reduce(u[cols])
            else:
                s[r] = 0

        # forward substitution for parity
        p = np.zeros(m, dtype=np.uint8)
        for r in range(m):
            val = s[r]
            idxs = par_lower[r]
            if idxs.size:
                val ^= np.bitwise_xor.reduce(p[idxs])
            # because diagonal is 1: p[r] = val
            p[r] = val

        c = np.concatenate([u, p])
        if c.size != n:
            raise RuntimeError(f"Internal error: codeword length {c.size} != nldpc {n}")
        return c


# ------------------------------------------------------------
# Convenience wrapper (function-style)
# ------------------------------------------------------------
def ldpc_encode_bits(u_bits: np.ndarray, fecframe: str, rate: str, mat_path: str) -> np.ndarray:
    enc = DVB_LDPC_Encoder(mat_path)
    return enc.encode(u_bits, fecframe, rate)


# ------------------------------------------------------------
# Optional: quick self-test
# ------------------------------------------------------------
def _quick_test(mat_path: str, fecframe="normal", rate="1/2"):
    enc = DVB_LDPC_Encoder(mat_path)
    prep = enc._prepare(fecframe, rate)
    k = prep["k"]
    n = prep["n"]

    u = np.random.randint(0, 2, size=k, dtype=np.uint8)
    c = enc.encode(u, fecframe, rate)

    print("OK")
    print("kldpc =", k, "nldpc =", n, "codeword =", c.size)


if __name__ == "__main__":
    # Example:
    # python ldpc_encoder.py
    MAT = "C:\\Users\\umair\\Videos\\JOB - NASTP\\SATCOM\\Code\\s2xLDPCParityMatrices\\dvbs2xLDPCParityMatrices.mat"
    if os.path.isfile(MAT):
        _quick_test(MAT, "normal", "1/2")
    else:
        print("Place dvbs2xLDPCParityMatrices.mat next to this script or edit MAT path.")
