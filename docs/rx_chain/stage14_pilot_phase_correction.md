Stage 14 – Pilot-Based Phase Correction
---------------------------------------
Purpose  
Estimate and remove common phase error (CPE) between pilot insertions using the known π/2‑BPSK pilot blocks. EN 302 307‑1 §5.5.3 (note on pilot use).

Interface (rx/_03_pilot_phase_correction.py)  
`apply_pilot_phase_correction(payload_data, pilots, fecframe) -> (corrected, phases, meta)`

Inputs  
- `payload_data` : complex, length S·90 (pilot-free payload).  
- `pilots` : complex array (Npilots, 36), extracted in Stage 13.  
- `fecframe` : "short" or "normal" to derive expected pilot count.

Outputs  
- `corrected` : complex payload after per-interval derotation.  
- `phases` : float radians, length Npilots.  
- `meta` : `chunk_len` (1440 symbols), `pilot_blocks`, `payload_symbols`.

Estimator math  
Pilot symbol ref (all 36 equal): p_ref = (1 + j)/√2.  
For pilot block k:  
  rotated_k = pilots[k] · conj(p_ref)  
  φ_k = arg( mean(rotated_k) )   # ML CPE estimate under AWGN  
Apply correction:  
- For symbols between pilot k and k+1 (16 slots = 1440 symbols): d ← d · exp(−j φ_k).  
- Tail after last pilot uses φ_last.

Edge cases  
- If pilots are disabled upstream, Stage 13 skips pilot extraction and this stage is bypassed.  
- Raises if pilot count ≠ expected for fecframe (protects against wrong TYPE.PILOTS).

Complexity  
- O(N) multiplies; one mean+atan2 per pilot block (22 max for normal frames).
