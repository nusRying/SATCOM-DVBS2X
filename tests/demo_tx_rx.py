"""
Complete transmitter→receiver demo in one script.
Config: QPSK, rate 1/2, short frame, pilots on, scrambling_code=0.
Outputs:
  - df_tx (original bits)
  - df_rx (recovered bits)
  - status messages
"""

from __future__ import annotations

import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tx._01_BB_Frame import build_bbheader
from tx._02_stream_adaptation import stream_adaptation_rate
from tx._03_bch_encoding import bch_encode_bbframe
from tx._04_ldpc_Encoding import DVB_LDPC_Encoder
from common.bit_interleaver import dvbs2_bit_interleave
from common.constellation_mapper import dvbs2_constellation_map
from common.pilot_insertion import insert_pilots_into_payload
from tx._05_pl_header import build_plheader, modcod_from_modulation_rate
from common.pl_scrambler import pl_scramble_full_plframe
from rx.receiver_Chain import process_rx_plframe


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save_bits(path: str, bits: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join("1" if int(b) else "0" for b in bits.reshape(-1)))


def _save_symbols(path: str, syms: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in syms.reshape(-1):
            f.write(f"{s.real:+.6f} {s.imag:+.6f}\n")


def build_tx_frame(
    dfl: int = 6000,
    fecframe: str = "short",
    rate: str = "1/2",
    modulation: str = "QPSK",
    pilots_on: bool = True,
    scrambling_code: int = 0,
    mat_path: str | None = None,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(42)
    df_bits = rng.integers(0, 2, size=dfl, dtype=np.uint8)

    BBHEADER = build_bbheader(
        matype1=0x00, matype2=0x00, upl=0, dfl=dfl, sync=0x00, syncd=0x0000
    )
    BBFRAME = np.concatenate([BBHEADER, df_bits])

    scrambled = stream_adaptation_rate(BBFRAME, fecframe, rate)
    bch_codeword = bch_encode_bbframe(scrambled, fecframe, rate)

    mat_path = mat_path or os.path.join(ROOT, "s2xLDPCParityMatrices", "dvbs2xLDPCParityMatrices.mat")
    ldpc_encoder = DVB_LDPC_Encoder(mat_path)
    ldpc_codeword = ldpc_encoder.encode(bch_codeword, fecframe, rate)

    interleaved = dvbs2_bit_interleave(ldpc_codeword, modulation)
    payload_syms = dvbs2_constellation_map(interleaved, modulation, code_rate=rate)

    modcod = modcod_from_modulation_rate(modulation, rate)
    _, plh_syms = build_plheader(modcod, fecframe, pilots=pilots_on)

    payload_with_pilots, _ = insert_pilots_into_payload(payload_syms, pilots_on, fecframe=fecframe)
    plframe_pre = np.concatenate([plh_syms, payload_with_pilots])
    plframe = pl_scramble_full_plframe(plframe_pre, scrambling_code=scrambling_code, plheader_len=len(plh_syms))

    meta = {
        "df_bits": df_bits,
        "BBFRAME": BBFRAME,
        "scrambled": scrambled,
        "bch_codeword": bch_codeword,
        "ldpc_codeword": ldpc_codeword,
        "interleaved": interleaved,
        "payload_syms": payload_syms,
        "plh_syms": plh_syms,
    }
    return plframe, meta


def main() -> None:
    fecframe = "short"
    rate = "1/2"
    modulation = "QPSK"
    pilots_on = True
    scrambling_code = 0

    out_dir = _ensure_dir(os.path.join(ROOT, "demo_output"))

    plframe, tx_meta = build_tx_frame(
        fecframe=fecframe,
        rate=rate,
        modulation=modulation,
        pilots_on=pilots_on,
        scrambling_code=scrambling_code,
    )

    # Save TX intermediates
    _save_bits(os.path.join(out_dir, "df_bits.txt"), tx_meta["df_bits"])
    _save_bits(os.path.join(out_dir, "BBFRAME.txt"), tx_meta["BBFRAME"])
    _save_bits(os.path.join(out_dir, "scrambled.txt"), tx_meta["scrambled"])
    _save_bits(os.path.join(out_dir, "bch_codeword.txt"), tx_meta["bch_codeword"])
    _save_bits(os.path.join(out_dir, "ldpc_codeword.txt"), tx_meta["ldpc_codeword"])
    _save_bits(os.path.join(out_dir, "interleaved_bits.txt"), tx_meta["interleaved"])
    _save_symbols(os.path.join(out_dir, "payload_symbols.txt"), tx_meta["payload_syms"])
    _save_symbols(os.path.join(out_dir, "plh_symbols.txt"), tx_meta["plh_syms"])
    _save_symbols(os.path.join(out_dir, "plframe_symbols.txt"), plframe)

    rx_out = process_rx_plframe(
        plframe,
        fecframe=fecframe,
        scrambling_code=scrambling_code,
        modulation=modulation,
        rate=rate,
        noise_var=1e-6,
        decode_ldpc=True,
        ldpc_max_iter=20,
    )

    df_tx = tx_meta["df_bits"]
    df_rx = rx_out["df_bits"]
    if df_rx is None or df_rx.size != df_tx.size:
        raise AssertionError("Receiver did not produce DF of expected length")
    if not np.array_equal(df_tx, df_rx):
        errs = int(np.sum(df_tx != df_rx))
        raise AssertionError(f"DF mismatch: {errs} bit errors")

    # Save RX intermediates
    _save_symbols(os.path.join(out_dir, "payload_corrected.txt"), rx_out["payload_corrected"])
    _save_symbols(os.path.join(out_dir, "pilots_extracted.txt"), rx_out["pilots"].reshape(-1))
    np.savetxt(os.path.join(out_dir, "phase_estimates.txt"), rx_out["phase_estimates"])
    np.savetxt(os.path.join(out_dir, "llrs_interleaved.txt"), rx_out["llrs_interleaved"])
    np.savetxt(os.path.join(out_dir, "llrs_deinterleaved.txt"), rx_out["llrs_deinterleaved"])
    if rx_out["ldpc_bits"] is not None:
        _save_bits(os.path.join(out_dir, "ldpc_bits_rx.txt"), rx_out["ldpc_bits"])
    if rx_out["bch_payload"] is not None:
        _save_bits(os.path.join(out_dir, "bch_payload_rx.txt"), rx_out["bch_payload"])
    if rx_out["bbframe_padded"] is not None:
        _save_bits(os.path.join(out_dir, "bbframe_padded_rx.txt"), rx_out["bbframe_padded"])
    if rx_out["df_bits"] is not None:
        _save_bits(os.path.join(out_dir, "df_bits_rx.txt"), rx_out["df_bits"])

    # Report
    report_path = os.path.join(out_dir, "demo_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TX->RX Demo Report\n")
        f.write("==================\n")
        f.write(f"Modulation : {modulation}\n")
        f.write(f"Rate       : {rate}\n")
        f.write(f"FECFRAME   : {fecframe}\n")
        f.write(f"Pilots     : {pilots_on}\n")
        f.write(f"Scrambling : {scrambling_code}\n")
        f.write("\nLengths:\n")
        f.write(f"DF bits             : {df_tx.size}\n")
        f.write(f"BBFRAME             : {tx_meta['BBFRAME'].size}\n")
        f.write(f"BCH codeword        : {tx_meta['bch_codeword'].size}\n")
        f.write(f"LDPC codeword       : {tx_meta['ldpc_codeword'].size}\n")
        f.write(f"Interleaved bits    : {tx_meta['interleaved'].size}\n")
        f.write(f"Payload symbols     : {tx_meta['payload_syms'].size}\n")
        f.write(f"PLFRAME symbols     : {plframe.size}\n")
        f.write("\nRX stats:\n")
        f.write(f"LDPC iterations     : {rx_out['ldpc_meta']['iterations']}\n")
        f.write(f"LDPC syndrome wgt   : {rx_out['ldpc_meta']['syndrome_weight']}\n")
        f.write(f"BCH corrected?      : {rx_out['bch_meta']['corrected'] if rx_out['bch_meta'] else 'n/a'}\n")
        f.write(f"Recovered DF bits   : {df_rx.size if df_rx is not None else 0}\n")
        f.write("\nResult: PASS (DF match)\n")

    print("TX->RX demo PASSED")
    print(f"DF bits: {df_tx.size}, LDPC iterations: {rx_out['ldpc_meta']['iterations']}")
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
