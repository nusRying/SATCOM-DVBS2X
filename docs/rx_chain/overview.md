RX Chain Overview (Stages 11–19)
---------------------------------
Scope  
- DVB‑S2/S2X baseband receive chain after front‑end timing/FEC sync. Implements EN 302 307‑1 clauses 5.2‑5.5 to recover BBFRAME/DF bits.

Execution order (with file references)  
11. PLHEADER decode (`rx/_00_pl_header_decoder.py`) – demap π/2‑BPSK header, check SOF, decode PLSC → MODCOD/FECFRAME/pilots.  
12. PL descramble (`rx/_01_pl_descrambler.py`) – undo complex Gold scrambler Cₙ after the 90‑sym PLHEADER.  
13. Pilot removal (`rx/_02_pilot_removal_rx.py`) – drop 36‑sym pilot blocks every 16 slots when pilots are present.  
14. Pilot phase correction (`rx/_03_pilot_phase_correction.py`) – estimate CPE per pilot block and derotate payload.  
15. Constellation demap (`rx/_04_constellation_demapper.py`) – soft LLRs for QPSK/8PSK/16APSK/32APSK.  
16. Bit de‑interleave (`common/bit_interleaver.py`) – inverse of TX interleaver (QPSK is identity).  
17. LDPC decode (`rx/_05_ldpc_decoder.py`) – normalized min‑sum with tables from `config/ldpc_matrices/*.mat`.  
18. BCH check/strip (`rx/_06_bch_decoding.py`) – detect‑only outer code, remove parity.  
19. BB deframe (`rx/_07_bb_deframer.py`) – CRC‑8 on BBHEADER, remove padding using DFL, output DF bits.

Orchestration  
- `rx/receiver_Chain.py::process_rx_plframe(...)` wires the stages and returns a dictionary of all intermediates for debugging/plots.  
- `pilots_on` flag lets the chain run with or without pilots; length checks adjust accordingly.  
- LDPC decoders are cached per MAT path to avoid repeated 20 MB loads.

Typical dimensions (short vs normal FECFRAME)  
- Slots: 90 (short), 360 (normal); slot length = 90 symbols.  
- Pilot blocks: 5 (short), 22 (normal) when pilots enabled.  
- Payload symbols (pre‑pilots): 8 100 (short), 32 400 (normal).  
- Payload with pilots: 8 280 (short), 33 192 (normal).  
- PLFRAME length: 90 (PLHEADER) + payload(+pilots).
