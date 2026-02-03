"""
End-to-end DVB-S2X-like (S2 baseline) TX→RX using the same input data path as tx/run_dvbs2.py.
Non-interactive, single-frame demo with real bits from GS_data/umair_gs_bits.csv.
Config: QPSK, rate 1/2, short frame, pilots on, scrambling_code=0, DFL capped to fit Kbch.
Outputs saved to dvbs2x_output/.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import json
import shutil
from datetime import datetime

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# TX imports
from tx.BB_Frame import build_bbheader, load_bits_csv, resolve_input_path
from tx.stream_adaptation import stream_adaptation_rate, get_kbch
from tx.bch_encoding import bch_encode_bbframe
from tx.ldpc_Encoding import DVB_LDPC_Encoder
from common.bit_interleaver import dvbs2_bit_interleave
from common.constellation_mapper import dvbs2_constellation_map
from common.pilot_insertion import insert_pilots_into_payload
from tx.pl_header import build_plheader, modcod_from_modulation_rate
from common.pl_scrambler import pl_scramble_full_plframe

# RX imports
from rx.receiver_Chain import process_rx_plframe


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_bits(path: str, bits: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join("1" if int(b) else "0" for b in bits.reshape(-1)))


def save_symbols(path: str, syms: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in syms.reshape(-1):
            f.write(f"{s.real:+.6f} {s.imag:+.6f}\n")

def save_json(path: str, obj: dict) -> None:
    def _convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        return x
    obj_conv = {k: _convert(v) for k, v in obj.items()} if isinstance(obj, dict) else _convert(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj_conv, f, indent=2, sort_keys=True)

def _prompt(msg: str, default: str | None = None) -> str:
    try:
        if default is None:
            return input(msg).strip()
        val = input(f"{msg} [{default}]: ").strip()
        return val if val else default
    except EOFError:
        # Non-interactive: fall back to default if provided
        if default is not None:
            return default
        raise


def run_end_to_end() -> None:
    # Interactive inputs (similar to tx/run_dvbs2.py)
    default_csv = os.path.join(ROOT, "GS_data", "umair_gs_bits.csv")
    default_mat = os.path.join(ROOT, "s2xLDPCParityMatrices", "dvbs2xLDPCParityMatrices.mat")
    bits_csv_path = _prompt("Enter bits CSV path", default_csv)
    mat_path = _prompt("Enter LDPC MAT path", default_mat)
    stream_type = _prompt("Enter stream type (TS/GS)", "GS").upper()
    fecframe = _prompt("Enter FECFRAME (normal/short)", "short").lower()
    rate = _prompt("Enter code rate (e.g., 1/2)", "1/2")
    modulation = _prompt("Enter modulation (QPSK/8PSK/16APSK/32APSK)", "QPSK").upper()
    pilots_on = _prompt("Enable pilots (on/off)", "on").lower() in {"on", "yes", "y", "1"}
    scrambling_code = int(_prompt("Enter PL scrambling code (0..262142)", "0"))

    # Load input bits
    bits_path = resolve_input_path(bits_csv_path)
    in_bits = load_bits_csv(bits_path)

    # DFL selection
    Kbch = get_kbch(fecframe, rate)
    dfl_max = Kbch - 80
    dfl_in = int(_prompt(f"Enter DFL (<= {min(dfl_max, in_bits.size)} and <= available bits)", str(min(6000, dfl_max, in_bits.size))))
    dfl = min(dfl_in, dfl_max, in_bits.size)

    out_dir = ensure_dir(os.path.join(ROOT, "dvbs2x_output"))
    reports_dir = ensure_dir(os.path.join(ROOT, "dvbs2x_reports"))

    df_bits = in_bits[:dfl]

    # BBHEADER (GS continuous: UPL=0)
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
    ldpc_encoder = DVB_LDPC_Encoder(mat_path)
    ldpc_codeword = ldpc_encoder.encode(bch_codeword, fecframe, rate)

    # Interleave + map
    interleaved = dvbs2_bit_interleave(ldpc_codeword, modulation)
    payload_syms = dvbs2_constellation_map(interleaved, modulation, code_rate=rate)

    # PLHEADER
    modcod = modcod_from_modulation_rate(modulation, rate)
    _, plh_syms = build_plheader(modcod, fecframe, pilots=pilots_on)

    # Pilots
    payload_with_pilots, _ = insert_pilots_into_payload(payload_syms, pilots_on, fecframe=fecframe)

    # PLFRAME pre-scramble
    plframe_pre = np.concatenate([plh_syms, payload_with_pilots])
    plframe = pl_scramble_full_plframe(plframe_pre, scrambling_code=scrambling_code, plheader_len=len(plh_syms))

    # Save TX artifacts
    save_bits(os.path.join(out_dir, "df_bits.txt"), df_bits)
    save_bits(os.path.join(out_dir, "BBFRAME.txt"), BBFRAME)
    save_bits(os.path.join(out_dir, "scrambled.txt"), scrambled)
    save_bits(os.path.join(out_dir, "bch_codeword.txt"), bch_codeword)
    save_bits(os.path.join(out_dir, "ldpc_codeword.txt"), ldpc_codeword)
    save_bits(os.path.join(out_dir, "interleaved_bits.txt"), interleaved)
    save_symbols(os.path.join(out_dir, "payload_symbols.txt"), payload_syms)
    save_symbols(os.path.join(out_dir, "plh_symbols.txt"), plh_syms)
    save_symbols(os.path.join(out_dir, "plframe_symbols.txt"), plframe)
    tx_meta_paths = {
        "df_bits": "df_bits.txt",
        "BBFRAME": "BBFRAME.txt",
        "scrambled": "scrambled.txt",
        "bch_codeword": "bch_codeword.txt",
        "ldpc_codeword": "ldpc_codeword.txt",
        "interleaved_bits": "interleaved_bits.txt",
        "payload_symbols": "payload_symbols.txt",
        "plh_symbols": "plh_symbols.txt",
        "plframe_symbols": "plframe_symbols.txt",
    }

    # RX
    rx_out = process_rx_plframe(
        plframe,
        fecframe=fecframe,
        scrambling_code=scrambling_code,
        modulation=modulation,
        rate=rate,
        noise_var=1e-6,
        decode_ldpc=True,
        ldpc_max_iter=30,
    )

    df_rx = rx_out["df_bits"]
    if df_rx is None or df_rx.size != df_bits.size:
        raise AssertionError("Receiver failed to recover DF bits")
    if not np.array_equal(df_bits, df_rx):
        errs = int(np.sum(df_bits != df_rx))
        raise AssertionError(f"DF mismatch: {errs} errors")

    # Save RX artifacts
    save_symbols(os.path.join(out_dir, "payload_corrected.txt"), rx_out["payload_corrected"])
    save_symbols(os.path.join(out_dir, "pilots_extracted.txt"), rx_out["pilots"].reshape(-1))
    np.savetxt(os.path.join(out_dir, "phase_estimates.txt"), rx_out["phase_estimates"])
    np.savetxt(os.path.join(out_dir, "llrs_interleaved.txt"), rx_out["llrs_interleaved"])
    np.savetxt(os.path.join(out_dir, "llrs_deinterleaved.txt"), rx_out["llrs_deinterleaved"])
    if rx_out["ldpc_bits"] is not None:
        save_bits(os.path.join(out_dir, "ldpc_bits_rx.txt"), rx_out["ldpc_bits"])
    if rx_out["bch_payload"] is not None:
        save_bits(os.path.join(out_dir, "bch_payload_rx.txt"), rx_out["bch_payload"])
    if rx_out["bbframe_padded"] is not None:
        save_bits(os.path.join(out_dir, "bbframe_padded_rx.txt"), rx_out["bbframe_padded"])
    if rx_out["df_bits"] is not None:
        save_bits(os.path.join(out_dir, "df_bits_rx.txt"), rx_out["df_bits"])
    # Save additional raw payload and metadata
    save_symbols(os.path.join(out_dir, "payload_raw.txt"), rx_out["payload_raw"])
    save_json(os.path.join(out_dir, "pilot_meta.json"), rx_out["pilot_meta"])
    save_json(os.path.join(out_dir, "phase_meta.json"), rx_out["phase_meta"])
    save_json(os.path.join(out_dir, "demap_meta.json"), rx_out["demap_meta"])
    save_json(os.path.join(out_dir, "ldpc_meta.json"), rx_out["ldpc_meta"])
    if rx_out["df_meta"] is not None:
        save_json(os.path.join(out_dir, "df_meta.json"), rx_out["df_meta"])

    rx_meta_paths = {
        "payload_corrected": "payload_corrected.txt",
        "payload_raw": "payload_raw.txt",
        "pilots_extracted": "pilots_extracted.txt",
        "phase_estimates": "phase_estimates.txt",
        "llrs_interleaved": "llrs_interleaved.txt",
        "llrs_deinterleaved": "llrs_deinterleaved.txt",
        "ldpc_bits_rx": "ldpc_bits_rx.txt" if rx_out["ldpc_bits"] is not None else None,
        "bch_payload_rx": "bch_payload_rx.txt" if rx_out["bch_payload"] is not None else None,
        "bbframe_padded_rx": "bbframe_padded_rx.txt" if rx_out["bbframe_padded"] is not None else None,
        "df_bits_rx": "df_bits_rx.txt" if rx_out["df_bits"] is not None else None,
        "pilot_meta": "pilot_meta.json",
        "phase_meta": "phase_meta.json",
        "demap_meta": "demap_meta.json",
        "ldpc_meta": "ldpc_meta.json",
        "df_meta": "df_meta.json" if rx_out["df_meta"] is not None else None,
    }

    # Report
    def _bits_preview(arr: np.ndarray, max_len: int = 128) -> str:
        bits = "".join("1" if int(b) else "0" for b in arr.reshape(-1))
        if len(bits) > max_len:
            return bits[:max_len] + "..."
        return bits

    def _syms_preview(arr: np.ndarray, count: int = 8) -> str:
        arr = arr.reshape(-1)
        return ", ".join(f"{s.real:+.3f}{s.imag:+.3f}j" for s in arr[:count])

    def _bits_full(arr: np.ndarray) -> str:
        return "".join("1" if int(b) else "0" for b in arr.reshape(-1))

    def _syms_full(arr: np.ndarray) -> str:
        return ", ".join(f"{s.real:+.6f}{s.imag:+.6f}j" for s in arr.reshape(-1))

    def _floats_full(arr: np.ndarray) -> str:
        return " ".join(f"{float(x):+.6f}" for x in arr.reshape(-1))

    report_path = os.path.join(out_dir, "dvbs2x_report.txt")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DVB-S2X END-TO-END REPORT\n")
        f.write(f"Generated on: {now}\n\n")
        f.write("This report documents TX→RX processing from input bits to recovered DF.\n\n")

        f.write("============================================================\n")
        f.write("INPUT DATA\n")
        f.write("============================================================\n")
        f.write(f"Input CSV file : {bits_path}\n")
        f.write(f"Total bits     : {in_bits.size}\n")
        f.write(f"First bits     : {_bits_preview(in_bits)}\n\n")

        f.write("============================================================\n")
        f.write("BBHEADER (ETSI CLAUSE 5.1.6)\n")
        f.write("============================================================\n")
        f.write(f"Length : {BBHEADER.size} bits\n")
        f.write(f"Bits   : {_bits_preview(BBHEADER, 160)}\n\n")
        f.write(f"MATYPE-1 : 0x00\n")
        f.write(f"UPL      : {0 if stream_type=='GS' else 188*8}\n")
        f.write(f"DFL      : {dfl}\n")
        f.write(f"SYNC     : 0x00\n")
        f.write(f"SYNCD    : 0x0000\n\n")

        f.write("============================================================\n")
        f.write("DATA FIELD (MERGER/SLICER)\n")
        f.write("============================================================\n")
        f.write(f"DFL bits : {dfl}\n")
        f.write(f"DF bits  : {_bits_preview(df_bits)}\n\n")

        f.write("============================================================\n")
        f.write("STREAM ADAPTATION + BCH + LDPC\n")
        f.write("============================================================\n")
        f.write(f"Kbch            : {Kbch}\n")
        f.write(f"Nbch            : {bch_codeword.size}\n")
        f.write(f"LDPC length     : {ldpc_codeword.size}\n")
        f.write(f"Scrambled (1st) : {_bits_preview(scrambled)}\n")
        f.write(f"BCH cw (1st)    : {_bits_preview(bch_codeword)}\n")
        f.write(f"LDPC cw (1st)   : {_bits_preview(ldpc_codeword)}\n\n")

        f.write("============================================================\n")
        f.write("CONSTELLATION + PILOTS + PLFRAME\n")
        f.write("============================================================\n")
        f.write(f"Modulation      : {modulation}\n")
        f.write(f"Rate            : {rate}\n")
        f.write(f"Pilots          : {pilots_on}\n")
        f.write(f"Payload symbols : {payload_syms.size}\n")
        f.write(f"First payload   : {_syms_preview(payload_syms)}\n")
        f.write(f"PLHEADER symbols: {plh_syms.size}\n")
        f.write(f"PLFRAME symbols : {plframe.size}\n")
        f.write(f"First PLFRAME   : {_syms_preview(plframe)}\n\n")

        f.write("============================================================\n")
        f.write("RECEIVER PIPELINE\n")
        f.write("============================================================\n")
        f.write(f"LDPC iterations : {rx_out['ldpc_meta']['iterations']}\n")
        f.write(f"LDPC syndrome   : {rx_out['ldpc_meta']['syndrome_weight']}\n")
        f.write(f"BCH corrected?  : {rx_out['bch_meta']['corrected'] if rx_out['bch_meta'] else 'n/a'}\n")
        f.write(f"Recovered DF    : {df_rx.size} bits\n")
        f.write(f"DF match        : {'YES' if np.array_equal(df_rx, df_bits) else 'NO'}\n\n")

        f.write("============================================================\n")
        f.write("RECEIVER INTERMEDIATE DATA (FULL)\n")
        f.write("============================================================\n")
        f.write("Payload (raw, post-pilot removal):\n")
        f.write(_syms_full(rx_out["payload_raw"]) + "\n\n")
        f.write("Payload (phase-corrected):\n")
        f.write(_syms_full(rx_out["payload_corrected"]) + "\n\n")
        f.write("Pilots (extracted):\n")
        f.write(_syms_full(rx_out["pilots"]) + "\n\n")
        f.write("Phase estimates (rad):\n")
        f.write(_floats_full(rx_out["phase_estimates"]) + "\n\n")
        f.write("LLRs (interleaved):\n")
        f.write(_floats_full(rx_out["llrs_interleaved"]) + "\n\n")
        f.write("LLRs (deinterleaved):\n")
        f.write(_floats_full(rx_out["llrs_deinterleaved"]) + "\n\n")
        if rx_out["ldpc_bits"] is not None:
            f.write("LDPC bits (post decode):\n")
            f.write(_bits_full(rx_out["ldpc_bits"]) + "\n\n")
        if rx_out["bch_payload"] is not None:
            f.write("BCH payload (Kbch):\n")
            f.write(_bits_full(rx_out["bch_payload"]) + "\n\n")
        if rx_out["bbframe_padded"] is not None:
            f.write("BBFRAME after descramble (Kbch):\n")
            f.write(_bits_full(rx_out["bbframe_padded"]) + "\n\n")
        if rx_out["df_bits"] is not None:
            f.write("Recovered DF bits (full):\n")
            f.write(_bits_full(rx_out["df_bits"]) + "\n\n")

        f.write("============================================================\n")
        f.write("FILES\n")
        f.write("============================================================\n")
        f.write("TX artifacts:\n")
        for k, v in tx_meta_paths.items():
            f.write(f"  {k:18s}: {v}\n")
        f.write("\nRX artifacts:\n")
        for k, v in rx_meta_paths.items():
            if v:
                f.write(f"  {k:18s}: {v}\n")

    # Copy report to central reports folder with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(report_path, os.path.join(reports_dir, f"dvbs2x_report_{ts}.txt"))

    print("DVB-S2X TX->RX completed successfully.")
    print(f"Artifacts in: {out_dir}")
    print(f"Report archived in: {reports_dir}")


if __name__ == "__main__":
    run_end_to_end()
