Stage 19 – BB Deframer
----------------------
Purpose  
Validate BBHEADER, drop padding, and recover the user Data Field (DF) bits. EN 302 307‑1 §5.2.

Interface (rx/_07_bb_deframer.py)  
`deframe_bb(scrambled_kbch, fecframe, rate) -> (df_bits, meta)`

Inputs  
- `scrambled_kbch` : uint8 length Kbch, already BB‑descrambled.  
- `fecframe`, `rate` : determine Kbch via `get_kbch`.

Outputs  
- `df_bits` : length = DFL from BBHEADER.  
- `meta` : MATYPE1, MATYPE2, UPL, DFL, SYNC, SYNCD, CRC, padding_bits.

Steps  
1) Length check: must equal Kbch.  
2) Split: header = first 80 bits; remainder = df_with_pad.  
3) CRC‑8: compute over first 72 header bits, compare to received 8 bits (poly per §5.2.3).  
4) Parse fields (MSB‑first):  
   - MATYPE1 (8): stream type / TS/GSE flags.  
   - MATYPE2 (8): ISSY/NP indicators.  
   - UPL (16): user packet length (bytes).  
   - DFL (16): data field length (bits).  
   - SYNC (8) & SYNCD (16): MPEG‑TS sync byte and offset.  
5) df_bits = df_with_pad[0:DFL]; padding_bits = len(df_with_pad) − DFL.

Guards  
- Raises on CRC mismatch, DFL overflow, or length errors.  
- Validates bits are 0/1 and flattens to 1‑D.

Output semantics  
- df_bits can be TS packets (when MATYPE indicates MPEG‑TS) or GSE data; padding stripped according to DFL only.
