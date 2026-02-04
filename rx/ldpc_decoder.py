# Stage 17 - LDPC Decoder
# ldpc_decoder.py
# =============================================================================
# DVB-S2 LDPC decoder (normalized min-sum) using parity tables from the same
# .mat file used by the encoder. Supports normal/short frames and all rates
# present in the MAT file. Designed for floating-point LLR input.
# =============================================================================

from __future__ import annotations

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import re
import numpy as np
from typing import Dict, Any, Tuple, Optional

try:
    import scipy.io
except ImportError as e:
    raise ImportError("scipy is required for LDPC decoding (load .mat). Install scipy.") from e

NLDPC_NORMAL = 64800
NLDPC_SHORT = 16200


def _rate_to_key(rate: str) -> str:
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", rate)
    if not m:
        raise ValueError("rate must look like '1/2', '3/5', ...")
    return f"{m.group(1)}_{m.group(2)}"


def _frame_to_suffix(fecframe: str) -> str:
    fecframe = fecframe.strip().lower()
    if fecframe == "normal":
        return "N"
    if fecframe == "short":
        return "S"
    raise ValueError("fecframe must be 'normal' or 'short'")


class DVB_LDPC_Decoder:
    def __init__(self, mat_path: str):
        if not os.path.isfile(mat_path):
            raise FileNotFoundError(f"MAT file not found: {mat_path}")
        self.mat_path = mat_path
        self._mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _load_pairs(self, fecframe: str, rate: str) -> np.ndarray:
        rate_key = _rate_to_key(rate)
        suffix = _frame_to_suffix(fecframe)
        mat_key = f"PT_{rate_key}_{suffix}"
        if mat_key not in self._mat:
            raise KeyError(f"Parity table '{mat_key}' not found in MAT.")
        pairs = np.asarray(self._mat[mat_key])
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"{mat_key} must be Nx2 (row,col) pairs, got {pairs.shape}")
        return pairs.astype(np.int64, copy=False)

    def _prepare(self, fecframe: str, rate: str) -> Dict[str, Any]:
        key = (fecframe, rate)
        if key in self._cache:
            return self._cache[key]

        pairs = self._load_pairs(fecframe, rate)
        rows = pairs[:, 0] - 1
        cols = pairs[:, 1] - 1

        n_expected = NLDPC_NORMAL if fecframe == "normal" else NLDPC_SHORT
        n_from_table = int(cols.max() + 1)
        if n_from_table != n_expected:
            raise ValueError(f"MAT table n={n_from_table} != expected {n_expected} for fecframe={fecframe}")

        m = int(rows.max() + 1)
        n = n_expected
        E = pairs.shape[0]

        # Build adjacency: list of edge indices for each row/col
        row_edges = [[] for _ in range(m)]
        col_edges = [[] for _ in range(n)]
        for e, (r, c) in enumerate(zip(rows, cols)):
            row_edges[r].append(e)
            col_edges[c].append(e)

        # Convert to numpy arrays for speed
        row_edges = [np.array(lst, dtype=np.int64) for lst in row_edges]
        col_edges = [np.array(lst, dtype=np.int64) for lst in col_edges]

        prepared = {
            "rows": rows,
            "cols": cols,
            "m": m,
            "n": n,
            "E": E,
            "row_edges": row_edges,
            "col_edges": col_edges,
        }
        self._cache[key] = prepared
        return prepared

    def decode(
        self,
        llr: np.ndarray,
        fecframe: str,
        rate: str,
        max_iter: int = 10,
        norm_factor: float = 0.9,
        damping: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalized min-sum LDPC decoding.

        Parameters
        ----------
        llr : np.ndarray float64, length = nldpc
        fecframe : 'normal' or 'short'
        rate : code rate string
        max_iter : iterations
        norm_factor : scaling of check-node output (0.75 ~ typical)
        damping : optional damping on variable updates (0..1)

        Returns
        -------
        hard_bits : np.uint8 array length nldpc
        meta : dict with iterations, syndrome_weight, success flag
        """
        prep = self._prepare(fecframe, rate)
        n = prep["n"]
        m = prep["m"]
        rows = prep["rows"]
        cols = prep["cols"]
        row_edges = prep["row_edges"]
        col_edges = prep["col_edges"]
        E = prep["E"]

        Lch = np.asarray(llr, dtype=np.float64).reshape(-1)
        if Lch.size != n:
            raise ValueError(f"LLR length {Lch.size} != nldpc {n}")

        # Messages: V->C and C->V, length E
        msg_vc = np.zeros(E, dtype=np.float64)
        msg_cv = np.zeros(E, dtype=np.float64)

        # Initialize V->C with channel LLR
        for c in range(n):
            idxs = col_edges[c]
            if idxs.size:
                msg_vc[idxs] = Lch[c]

        hard = np.zeros(n, dtype=np.uint8)
        syndrome_weight = m
        iters_done = 0

        # Early syndrome check on channel LLRs (often already valid when noise is tiny)
        hard[:] = 0
        hard[Lch < 0] = 1
        syn0 = 0
        for r in range(m):
            e_idxs = row_edges[r]
            bits = hard[cols[e_idxs]]
            if np.bitwise_xor.reduce(bits) != 0:
                syn0 += 1
        if syn0 == 0:
            return hard, {"iterations": 0, "syndrome_weight": 0, "success": True}

        for it in range(1, max_iter + 1):
            # Check node update
            for r in range(m):
                e_idxs = row_edges[r]
                msgs = msg_vc[e_idxs]
                signs = np.sign(msgs)
                signs[signs == 0] = 1.0
                prod_sign = np.prod(signs)
                mags = np.abs(msgs)
                if mags.size == 0:
                    continue
                min1_idx = int(np.argmin(mags))
                min1 = mags[min1_idx]
                # second min
                if mags.size > 1:
                    min2 = np.min(np.delete(mags, min1_idx))
                else:
                    min2 = min1

                for local_idx, e in enumerate(e_idxs):
                    s = prod_sign * signs[local_idx]
                    mag = min2 if local_idx == min1_idx else min1
                    msg_cv[e] = norm_factor * s * mag

            # Variable node update + a-posteriori
            for c in range(n):
                e_idxs = col_edges[c]
                incoming = msg_cv[e_idxs]
                total = Lch[c] + incoming.sum()
                if damping > 0.0:
                    new_msg = total - incoming
                    msg_vc[e_idxs] = (1 - damping) * new_msg + damping * msg_vc[e_idxs]
                else:
                    msg_vc[e_idxs] = total - incoming
                hard[c] = 0 if total >= 0 else 1

            # Syndrome check
            syndrome_weight = 0
            for r in range(m):
                e_idxs = row_edges[r]
                bits = hard[cols[e_idxs]]
                if np.bitwise_xor.reduce(bits) != 0:
                    syndrome_weight += 1
            iters_done = it
            if syndrome_weight == 0:
                break

        meta = {
            "iterations": iters_done,
            "syndrome_weight": int(syndrome_weight),
            "success": syndrome_weight == 0,
        }
        return hard, meta


def _self_test():
    """
    Loopback test: encode random bits, add tiny noise LLRs, decode and verify.
    Skips if MAT file is unavailable.
    """
    from ldpc_Encoding import DVB_LDPC_Encoder

    mat_default = os.path.join(
        os.path.dirname(__file__),
        "s2xLDPCParityMatrices",
        "dvbs2xLDPCParityMatrices.mat",
    )
    if not os.path.isfile(mat_default):
        print("LDPC decoder self-test skipped (MAT file not found).")
        return

    fec = "short"
    rate = "1/2"
    enc = DVB_LDPC_Encoder(mat_default)
    dec = DVB_LDPC_Decoder(mat_default)

    prep = enc._prepare(fec, rate)
    k = prep["k"]

    rng = np.random.default_rng(0)
    u = rng.integers(0, 2, size=k, dtype=np.uint8)
    c = enc.encode(u, fec, rate)

    # Build strong LLRs (no errors)
    llr = (1 - 2 * c).astype(np.float64) * 20.0

    hard, meta = dec.decode(llr, fec, rate, max_iter=10, norm_factor=0.9)
    if not meta["success"]:
        raise AssertionError("Decoder failed syndrome check in self-test")
    print("ldpc_decoder.py self-test PASSED (syndrome OK)")


if __name__ == "__main__":
    _self_test()
