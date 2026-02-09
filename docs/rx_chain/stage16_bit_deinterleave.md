Stage 16 – Bit De-Interleave
----------------------------
Purpose  
Undo the DVB‑S2 bit interleaver applied before constellation mapping (identity for QPSK). EN 302 307‑1 §5.3.3.

Interface (`common/bit_interleaver.py`)  
- `dvbs2_llr_deinterleave(llr, modulation) -> llr_deint`   (soft metrics)  
- `dvbs2_bit_deinterleave(bits, modulation) -> bits`       (hard bits)

Inputs  
- `llr` or `interleaved_bits` : length must be divisible by m = bits/symbol for the modulation.  
- `modulation` : "QPSK", "8PSK", "16APSK", "32APSK".

Outputs  
- De‑interleaved array same length, reordered to original LDPC bit order.

Permutation (for m>2)  
Given interleaved sequence y of length m·Ns (grouped by bit position), de‑interleaver reconstructs c such that:  
  c[j::m] = y[j*Ns : (j+1)*Ns],  j = 0..m−1  
For QPSK (m=2) the interleaver is identity, so de‑interleave is a no‑op.

Checks  
- Validates modulation support.  
- Length divisibility by m enforced.

Complexity  
- O(N) indexed copy; no arithmetic.

Placement in chain  
- Takes LLRs from Stage 15, outputs LLR order expected by Stage 17 LDPC decoder.
