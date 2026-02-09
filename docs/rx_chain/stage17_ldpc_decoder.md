Stage 17 – LDPC Decoder
-----------------------
Purpose  
Decode DVB‑S2 LDPC codewords via normalized min‑sum using the official parity matrices. EN 302 307‑1 §5.3.2 + Annex tables.

Interface (rx/_05_ldpc_decoder.py)  
`DVB_LDPC_Decoder.decode(llr, fecframe, rate, max_iter=30, norm_factor=0.75, damping=0.0) -> (hard_bits, meta)`

Inputs  
- `llr` : float length n (64800 normal / 16200 short) after de‑interleaver.  
- `fecframe` : "normal" or "short".  
- `rate` : e.g., "1/2", "3/5".  
- `max_iter` : iterations budget.  
- `norm_factor` α : scales check output (0.7–0.9 typical).  
- `damping` β : optional (0–1) for VN updates.  
- MAT path supplied at construction; cached.

Outputs  
- `hard_bits` : uint8 length n, systematic first (k then parity).  
- `meta` : `iterations`, `syndrome_weight`, `success` (True if all checks satisfied).

Prep  
- Load PT_<rate>_<S/N> table → (row,col) 1‑based pairs.  
- Convert to 0‑based, build adjacency lists row_edges, col_edges.  
- Validate Hp lower‑triangular with 1s on diagonal (DVB structure).

Iterative steps (t = 1..max_iter)  
1) **Check node** for each row r:  
   - signs = sign(msg_vc[e]); prod_sign = Π signs  
   - min1, min2 = smallest magnitudes of incoming |msg_vc|  
   - For each edge e: s = prod_sign * signs[e]; mag = min2 if e at min1 else min1; msg_cv[e] = α · s · mag  
2) **Variable node** for each col c:  
   - L_total = L_ch[c] + Σ msg_cv[e in col_edges[c]]  
   - msg_vc[e] = (1−β)(L_total − msg_cv[e]) + β·msg_vc[e]   (damping optional)  
   - hard[c] = 0 if L_total ≥ 0 else 1  
3) **Syndrome**: XOR hard bits per row; break early if all zero.

Early check  
- Before iterations, hard decision on channel LLRs; if syndrome already zero, return iterations=0.

Complexity  
- O(E·iter) time, O(E+n+m) memory; E ≈ 10×n for DVB‑S2 degree profile.

Notes  
- `process_rx_plframe` caches decoder instances by MAT path to avoid repeated 20 MB loads.  
- Normalization factor can be tuned; 0.75 default matches common min‑sum practice.
