"""
Non-interactive TX→RX loopback for DVB-S2 (QPSK, rate 1/2, short frame).
Generates random DF bits, runs full TX chain (sans pulse shaping), then feeds
resulting PLFRAME symbols into the RX pipeline with LDPC+BCH enabled and asserts
the recovered DF matches the original.
"""

from __future__ import annotations

import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tx._01_BB_Frame import build_bbheader
from tx._02_stream_adaptation import stream_adaptation_rate, get_kbch
from tx._03_bch_encoding import bch_encode_bbframe, BCH_PARAMS
from tx._04_ldpc_Encoding import DVB_LDPC_Encoder
from common.bit_interleaver import dvbs2_bit_interleave
from common.constellation_mapper import dvbs2_constellation_map
from common.pilot_insertion import insert_pilots_into_payload
from tx._05_pl_header import build_plheader, modcod_from_modulation_rate
from common.pl_scrambler import pl_scramble_full_plframe
from rx.receiver_Chain import process_rx_plframe


def tx_generate_frame(
    fecframe: str = "short",
    rate: str = "1/2",
    modulation: str = "QPSK",
    pilots_on: bool = True,
    dfl: int = 6000,
    scrambling_code: int = 0,
    mat_path: str | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Build one PLFRAME (symbols) and return (plframe_symbols, meta).
    """
    rng = np.random.default_rng(0)
    df_bits = rng.integers(0, 2, size=dfl, dtype=np.uint8)

    # BBHEADER (minimal GS, upl=0)
    BBHEADER = build_bbheader(
        matype1=0x00,
        matype2=0x00,
        upl=0,
        dfl=dfl,
        sync=0x00,
        syncd=0x0000,
    )
    BBFRAME = np.concatenate([BBHEADER, df_bits])

    # Stream adaptation
    scrambled = stream_adaptation_rate(BBFRAME, fecframe, rate)

    # BCH
    bch_codeword = bch_encode_bbframe(scrambled, fecframe, rate)

    # LDPC
    mat_path = mat_path or os.path.join(ROOT, "s2xLDPCParityMatrices", "dvbs2xLDPCParityMatrices.mat")
    ldpc_encoder = DVB_LDPC_Encoder(mat_path)
    ldpc_codeword = ldpc_encoder.encode(bch_codeword, fecframe, rate)

    # Bit interleaver (if applicable) and mapping
    interleaved = dvbs2_bit_interleave(ldpc_codeword, modulation)
    payload_syms = dvbs2_constellation_map(interleaved, modulation, code_rate=rate)

    # PLHEADER
    modcod = modcod_from_modulation_rate(modulation, rate)
    _, plh_syms = build_plheader(modcod, fecframe, pilots=pilots_on)

    # Pilot insertion
    payload_with_pilots, _ = insert_pilots_into_payload(payload_syms, pilots_on, fecframe=fecframe)

    # Assemble PLFRAME pre-scramble
    plframe_pre = np.concatenate([plh_syms, payload_with_pilots])

    # Scramble (PLHEADER excluded)
    plframe = pl_scramble_full_plframe(plframe_pre, scrambling_code=scrambling_code, plheader_len=len(plh_syms))

    meta = {
        "df_bits": df_bits,
        "bbframe": BBFRAME,
        "scrambled": scrambled,
        "bch_codeword": bch_codeword,
        "ldpc_codeword": ldpc_codeword,
        "interleaved": interleaved,
        "payload_syms": payload_syms,
        "plh_syms": plh_syms,
        "payload_with_pilots": payload_with_pilots,
    }
    return plframe, meta


def main() -> None:
    fecframe = "short"
    rate = "1/2"
    modulation = "QPSK"
    pilots_on = True
    scrambling_code = 0

    plframe, meta_tx = tx_generate_frame(
        fecframe=fecframe,
        rate=rate,
        modulation=modulation,
        pilots_on=pilots_on,
        scrambling_code=scrambling_code,
    )

    # Receiver process
    out = process_rx_plframe(
        plframe,
        fecframe=fecframe,
        scrambling_code=scrambling_code,
        modulation=modulation,
        noise_var=1e-6,  # tiny noise
        rate=rate,
        decode_ldpc=True,
        ldpc_max_iter=20,
    )

    df_rx = out["df_bits"]
    df_tx = meta_tx["df_bits"]

    assert df_rx is not None, "Receiver did not produce DF bits"
    if df_rx.size != df_tx.size or not np.array_equal(df_rx, df_tx):
        raise AssertionError("Loopback failed: DF bits mismatch")

    print("Loopback PASSED")
    print(f"DF bits: {df_tx.size}, Errors corrected (LDPC): {out['ldpc_meta']['syndrome_weight']}")


if __name__ == "__main__":
    main()
