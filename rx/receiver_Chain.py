    
"""
RX Chain: Steps 12-19 - Orchestrator

DVB-S2 receiver chain (partial): PL descramble -> pilot removal -> pilot-based
common phase correction. Downstream stages (demapper, deinterleave, LDPC/BCH)
are orchestrated here (stages 12..19).
"""

from __future__ import annotations

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from typing import Dict, Any

from rx.pl_descrambler import pl_descramble_full_plframe
from rx.pilot_removal_rx import remove_pilots_from_plframe
from rx.pilot_phase_correction import apply_pilot_phase_correction
from rx.constellation_demapper import dvbs2_constellation_demapper
from common.bit_interleaver import dvbs2_llr_deinterleave
from rx.ldpc_decoder import DVB_LDPC_Decoder
from rx.bch_decoding import bch_check_and_strip
from tx.stream_adaptation import stream_deadaptation_rate, BCH_PARAMS
from rx.bb_deframer import deframe_bb


def process_rx_plframe(
    rx_plframe: np.ndarray,
    fecframe: str,
    scrambling_code: int = 0,
    modulation: str = "QPSK",
    rate: str = "1/2",
    noise_var: float | None = None,
    esn0_db: float | None = None,
    ldpc_mat_path: str | None = None,
    ldpc_max_iter: int = 30,
    ldpc_norm: float = 0.75,
    decode_ldpc: bool = True,
) -> Dict[str, Any]:
    """
    Run the current RX stages on a full PLFRAME:
      1) PL descrambling (excludes PLHEADER)
      2) Pilot removal (returns data-only payload + pilots)
      3) Pilot-aided common phase correction on payload
      4) Constellation soft-demap to bit LLRs
      5) Bit deinterleave of LLRs (mirrors TX interleaver)
      6) LDPC decode (normalized min-sum)
      7) BCH check/strip
      8) BB descramble (padding kept)

    Returns a dict with intermediate artifacts for downstream blocks.
    """
    # Step 1: descramble everything after PLHEADER
    descrambled = pl_descramble_full_plframe(rx_plframe, scrambling_code=scrambling_code)

    # Step 2: strip pilots, keep data-only payload + pilots
    payload_data, pilots, pilot_meta = remove_pilots_from_plframe(descrambled, fecframe=fecframe)

    # Step 3: pilot-aided phase correction on payload
    corrected_payload, phases, phase_meta = apply_pilot_phase_correction(
        payload_data,
        pilots,
        fecframe=fecframe,
    )

    # Step 4: demap symbols -> LLRs
    llrs, demap_meta = dvbs2_constellation_demapper(
        corrected_payload,
        modulation=modulation,
        code_rate=rate,
        noise_var=noise_var,
        esn0_db=esn0_db,
    )

    # Step 5: deinterleave LLRs (inverse of TX bit interleaver)
    llrs_deint = dvbs2_llr_deinterleave(llrs, modulation)

    ldpc_bits = None
    ldpc_meta = None
    bch_payload = None
    bch_meta = None
    bbframe_padded = None
    df_bits = None
    df_meta = None

    if decode_ldpc:
        # Step 6: LDPC decode
        if ldpc_mat_path is None:
            ldpc_mat_path = os.path.join(ROOT, "config", "ldpc_matrices", "dvbs2xLDPCParityMatrices.mat")
        mat_path = ldpc_mat_path
        ldpc_dec = DVB_LDPC_Decoder(mat_path)
        ldpc_bits, ldpc_meta = ldpc_dec.decode(
            llrs_deint,
            fecframe=fecframe,
            rate=rate,
            max_iter=ldpc_max_iter,
            norm_factor=ldpc_norm,
        )

        # Step 7: BCH check/strip
        Nbch = BCH_PARAMS[(fecframe, rate)][1]
        ldpc_sys = ldpc_bits[:Nbch]
        bch_payload, bch_meta = bch_check_and_strip(ldpc_sys, fecframe, rate=rate)

        # Step 8: BB descramble (padding still present)
        bbframe_padded = stream_deadaptation_rate(bch_payload, fecframe, rate=rate)

        # Step 9: Deframe to recover DF bits (drop padding)
        df_bits, df_meta = deframe_bb(bbframe_padded, fecframe, rate)

    return {
        "payload_corrected": corrected_payload,
        "payload_raw": payload_data,
        "pilots": pilots,
        "pilot_meta": pilot_meta,
        "phase_estimates": phases,
        "phase_meta": phase_meta,
        "llrs_interleaved": llrs,
        "llrs_deinterleaved": llrs_deint,
        "demap_meta": demap_meta,
        "ldpc_bits": ldpc_bits,
        "ldpc_meta": ldpc_meta,
        "bch_payload": bch_payload,
        "bch_meta": bch_meta,
        "bbframe_padded": bbframe_padded,
        "df_bits": df_bits,
        "df_meta": df_meta,
        "rate": rate,
        "descrambled": descrambled,
    }


def _self_test() -> None:
    """
    Quick loopback using TX pilot insertion to validate removal + correction.
    """
    from common.pilot_insertion import insert_pilots_into_payload, PILOT_SYMBOLS_36
    from common.pl_scrambler import pl_scramble_full_plframe
    from tx.pl_header import build_plheader, modcod_from_modulation_rate

    rng = np.random.default_rng(1)
    fec = "short"
    modulation = "QPSK"
    rate = "1/2"

    # Build dummy PLHEADER (length 90 symbols)
    modcod = modcod_from_modulation_rate(modulation, rate)
    _, plh_syms = build_plheader(modcod, fec, pilots=True)

    # Generate random payload symbols (pretend mapped)
    nslots = 90
    payload_syms = (rng.standard_normal(nslots * 90) + 1j * rng.standard_normal(nslots * 90)).astype(np.complex128)

    payload_with_pilots, _ = insert_pilots_into_payload(payload_syms, pilots_on=True, fecframe=fec)
    plframe_pre = np.concatenate([plh_syms, payload_with_pilots])

    # Scramble as TX would do
    scr_code = 5
    tx = pl_scramble_full_plframe(plframe_pre, scrambling_code=scr_code, plheader_len=len(plh_syms))

    # Impose a constant phase error to simulate channel
    phi = 0.33
    rx = tx * np.exp(1j * phi)

    out = process_rx_plframe(
        rx,
        fecframe=fec,
        scrambling_code=scr_code,
        modulation=modulation,
        noise_var=0.01,  # arbitrary small noise variance for test
        rate=rate,
        ldpc_max_iter=3,  # keep test fast
        decode_ldpc=False,  # skip heavy FEC in quick self-test
    )

    # Payload should be recovered (phase corrected) to original symbols
    assert np.allclose(out["payload_corrected"], payload_syms, atol=1e-9)
    # Pilot estimates should be close to imposed phase
    assert np.allclose(out["phase_estimates"], phi, atol=1e-3)
    # Pilots should match known symbols after descramble
    assert np.allclose(out["pilots"][0], PILOT_SYMBOLS_36 * np.exp(1j * phi), atol=1e-9)
    # LLR signs should match the transmitted bits (hard decisions) length check
    hard = (out["llrs_deinterleaved"] < 0).astype(np.uint8)
    assert hard.size == payload_syms.size * 2

    print("receiver_Chain.py self-test PASSED")


if __name__ == "__main__":
    _self_test()
