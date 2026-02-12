Stage 11 - PLHEADER Decode
--------------------------
Goal  
Recover PLHEADER bits (pi/2-BPSK), verify the 26-bit Start Of Frame (SOF), and decode the 64-bit PLSC to obtain MODCOD, FECFRAME type, and pilots flag.

Interface (rx/_00_pl_header_decoder.py)  
- `decode_plheader(plheader_syms, sof_max_errors=4) -> dict`
- `demap_plheader_bits(plheader_syms) -> bits90`
- `decode_plsc(plsc_scrambled_bits) -> dict`

Key outputs  
- `modcod`, `modulation`, `code_rate` (Table 12 values)
- `fecframe` ("normal"/"short"), `pilots` (bool)
- `sof_errors`, `rm_distance` (Hamming to nearest RM(1,5) codeword), `type_lsb_mismatches`

Notes  
- Header symbols are assumed phase-coherent (front-end carrier loop must be locked).
- Scrambling sequence index `n` is not signalled; receiver still takes `scrambling_code` (default 0) for payload descrambling.
- Set `sof_max_errors` to tighten/loosen SOF acceptance depending on SNR.
