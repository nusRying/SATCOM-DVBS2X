Stage 15 – Constellation Demapper (LLRs)
----------------------------------------
Purpose  
Produce soft bit log-likelihood ratios from complex symbols for DVB‑S2 constellations. Uses analytical QPSK LLRs and max‑log for PSK/APSK. Mapping per EN 302 307‑1 figs 9–13.

Interface (rx/_04_constellation_demapper.py)  
`dvbs2_constellation_demapper(symbols, modulation, noise_var=None, esn0_db=None, code_rate=None, apsk_gammas=None) -> (llrs, meta)`

Inputs  
- `symbols` : complex payload after phase correction.  
- `modulation` : "QPSK" | "8PSK" | "16APSK" | "32APSK".  
- Noise: `noise_var` (E|n|²) or `esn0_db` (Es/N0 in dB).  
- APSK radii: `apsk_gammas` or inferred from `code_rate` via ETSI tables.

Outputs  
- `llrs` : float, length m·Ns, m = bits/symbol.  
- `meta` : chosen noise_var, bits_per_symbol, code_rate.

Math  
Noise variance resolution: σ² = 1 / 10^(EsN0/10) when not provided.  
- QPSK (Gray, MSB=I):  
    L_I = (2√2/σ²)·Re(s) ; L_Q = (2√2/σ²)·Im(s)  
    Pack as [L_I0, L_Q0, L_I1, L_Q1, ...].  
- Max‑log (8/16/32):  
    L_k = (1/σ²) [ min_{b=0}|r−s|² − min_{b=1}|r−s|² ]  
    Constellations built from ETSI bit labels; APSK points normalized to unit mean energy using default gammas (or provided).

Bits per symbol & default APSK gammas  
| Mod | m | Default gammas (by rate) |  
|-----|---|---------------------------|  
| QPSK | 2 | n/a |  
| 8PSK | 3 | n/a |  
| 16APSK | 4 | `DEFAULT_GAMMA_16` (e.g., 2/3→3.15, 3/4→2.85, 4/5→2.75, 5/6→2.70, 8/9→2.60, 9/10→2.57) |  
| 32APSK | 5 | `DEFAULT_GAMMA_32` (e.g., 3/4→(2.84,5.27), 4/5→(2.72,4.87), 5/6→(2.64,4.64), 8/9→(2.54,4.33), 9/10→(2.53,4.30)) |

Checks & guards  
- Validates modulation support and noise parameters.  
- For non‑QPSK, length must be divisible by m (interleaver requirement).

Complexity  
- QPSK: O(N).  
- Max‑log: O(N·M·m) with M=8/16/32 → lightweight for frame sizes.
