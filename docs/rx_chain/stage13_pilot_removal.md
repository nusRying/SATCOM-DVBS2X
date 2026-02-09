Stage 13 – Pilot Removal
------------------------
Goal  
Remove pilot blocks (36 symbols) inserted every 16 slots when pilots are enabled; support pilotless mode transparently. EN 302 307‑1 §5.5.3.

Interface (rx/_02_pilot_removal_rx.py)  
- `remove_pilots_from_plframe(rx_plframe, fecframe, plheader_len=90, pilots_on=True) -> payload, pilots, meta`

Inputs  
- `rx_plframe` : descrambled complex PLFRAME.  
- `fecframe` : "short" or "normal".  
- `pilots_on` : bool flag from TYPE.PILOTS.

Outputs  
- `payload_data` : complex, length = S·90 (S slots).  
- `pilots` : (Npilots, 36) complex; empty if pilots_off.  
- `meta` : lengths, slot count, pilot positions, pilots_on.

Structure & lengths  
- Slot length Ls = 90.  
- Slots S: 90 (short) / 360 (normal).  
- Pilot blocks count Np = ⌊(S−1)/16⌋ = 5 (short) / 22 (normal) when pilots_on.  
- Expected payload+pilots length = S·Ls + Np·36.  
- Expected PLFRAME length = 90 + that value.  

Quick reference  
| fecframe | S (slots) | Np (pilots) | payload (sym) | payload+pilots (sym) | full PLFRAME (sym) |  
|----------|-----------|-------------|---------------|-----------------------|--------------------|  
| short    | 90        | 5           | 8 100         | 8 280                 | 8 370              |  
| normal   | 360       | 22          | 32 400        | 33 192                | 33 282             |

Algorithm  
1) Drop PLHEADER (indices 0..89).  
2) Validate length vs expected (different if pilots_off).  
3) For k in 0..Np−1: copy 16·90 data symbols → payload, then next 36 → pilots[k].  
4) Copy any tail after last pilot (pilotless case copies straight through).

Guards  
- Raises on length mismatch (helps catch wrong fecframe or missing pilots).  
- Flattens to 1‑D complex.  
- Metadata includes `pilot_slots_after` (slot indices after which pilots were expected).

Complexity  
- Linear copies; no trig or divisions.
