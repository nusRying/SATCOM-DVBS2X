Stage 12 – PL Descrambler
--------------------------
Goal  
Undo DVB‑S2 physical-layer scrambling Cₙ on payload+pilots while leaving the 90‑symbol PLHEADER unchanged. Standard: EN 302 307‑1 §5.5.4.

Interface (rx/_01_pl_descrambler.py)  
- `pl_descramble_full_plframe(rx_plframe, scrambling_code, plheader_len=90) -> descrambled`

Inputs  
- `rx_plframe` : 1‑D complex array, length = 90 + payload(+pilots).  
- `scrambling_code n` : int in [0, 2¹⁸−2] (TYPE.SCRAMBLING).  
- `plheader_len` : usually 90.

Outputs  
- `descrambled` : complex array, same length. For i < plheader_len: passthrough; for i ≥ plheader_len: multiplied by conj(Cₙ(i−plheader_len)).

Math  
1) m‑sequences (degree 18):  
   - x: poly 1 + x⁷ + x¹⁸, init x₀=1, others 0  
   - y: poly 1 + y⁵ + y⁷ + y¹⁰ + y¹⁸, init all ones  
2) Gold bits: zₙ(i) = x((i+n) mod (2¹⁸−1)) ⊕ y(i).  
3) R sequence: Rₙ(i) = 2·zₙ(i+131072) + zₙ(i) ∈ {0,1,2,3}, defined for i=0…66419.  
4) Scrambler: Cₙ(i) = exp(j·Rₙ(i)·π/2) ∈ {1, j, −1, −j}.  
5) Descramble: s[i] = r[i] · conj(Cₙ(i−plheader_len)) for i ≥ plheader_len.

Guards  
- Validates code range; plheader_len within vector bounds.  
- `_as_complex_1d` ensures flat complex dtype.

Complexity & caching  
- O(N) multiplies; Cₙ generated with lru_cache in `common/pl_scrambler.py`, so repeated calls are cheap.
