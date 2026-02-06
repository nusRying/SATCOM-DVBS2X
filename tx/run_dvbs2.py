print("\n========== DVB-S2 TRANSMITTER RUN ==========\n")

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np

from tx._01_BB_Frame import (
    dvbs2_bbframe_generator_from_bits_csv,
    build_bbheader,
    PacketizedCrc8Stream,
    compute_syncd_packetized,
    load_bits_csv,
    resolve_input_path
)

from tx._02_stream_adaptation import (
    get_kbch,
    pad_bbframe_rate,
    save_bbframe_to_file_rate,
    stream_adaptation_rate
)

from tx.bbframe_report import BBFrameReport
from tx._03_bch_encoding import BCH_PARAMS, bch_encode_bbframe
from tx._04_ldpc_Encoding import DVB_LDPC_Encoder

from common.bit_interleaver import (
    dvbs2_bit_interleave,
    dvbs2_bit_deinterleave
)

from common.constellation_mapper import dvbs2_constellation_map
from tx._05_pl_header import modcod_from_modulation_rate, build_plheader
from common.pl_scrambler import pl_scramble_full_plframe
from common.pilot_insertion import insert_pilots_into_payload
from tx.plot_qpsk_fft import plot_qpsk_fft, plot_dvbs2_qpsk_spectrum
from tx._06_bb_filter import dvbs2_bb_filter

BITS_CSV_PATH = os.path.join(ROOT, "data", "GS_data", "umair_gs_bits.csv")
MAX_FRAMES    = 3
REPORT_FILE   = "dvbs2_full_report.txt"
MAT_PATH      = os.path.join(ROOT, "config", "ldpc_matrices", "dvbs2xLDPCParityMatrices.mat")


def run_dvbs2_transmitter():
    report = BBFrameReport(REPORT_FILE)

    stream_type = input("Enter stream type (TS or GS): ").strip().upper()
    fecframe    = input("Enter FECFRAME type (normal/short): ").strip().lower()
    rate        = input("Enter code rate (e.g., 1/2, 3/5, 2/3, 3/4, 5/6, 8/9, 9/10): ").strip()
    modulation  = input("Enter modulation (QPSK, 8PSK, 16APSK, 32APSK): ").strip().upper()
    pilots_in   = input("Enable pilots (pilots_on/pilots_off): ").strip().lower()
    alpha       = float(input("Enter roll-off alpha (0.35 / 0.25 / 0.20): ").strip())
    scr_in      = input("Enter PL scrambling code (0..262142, default 0): ").strip()
    scrambling_code = int(scr_in) if scr_in else 0
    DFL         = int(input("Enter DFL: "))
    sps_in      = input("Enter RRC samples-per-symbol (default 4): ").strip()
    span_in     = input("Enter RRC span in symbols (default 10): ").strip()
    rrc_sps     = int(sps_in) if sps_in else 4
    rrc_span    = int(span_in) if span_in else 10
    upc_in      = input("Enable RF upconversion (yes/no, default no): ").strip().lower()
    upconvert   = upc_in in {"yes", "y", "1", "true", "on"}
    if upconvert:
        Rs = float(input("Enter symbol rate in Hz (e.g., 1e6): ").strip())
        fc = float(input("Enter carrier frequency in Hz (must be < fs/2): ").strip())
        phase_in = input("Enter carrier phase (radians, default 0): ").strip()
        phase0 = float(phase_in) if phase_in else 0.0
        if Rs <= 0:
            raise ValueError("Symbol rate must be > 0.")
    else:
        Rs = 0.0
        fc = 0.0
        phase0 = 0.0

    pilots_on = pilots_in in {"pilots_on", "on", "yes", "y", "1", "true"}
    modcod = modcod_from_modulation_rate(modulation, rate)

    if stream_type == "TS":
        UPL  = 188 * 8
        SYNC = 0x47
    else:
        UPL  = int(input("Enter UPL in bits (0 for continuous GS): "))
        SYNC = int(input("Enter SYNC byte in hex (e.g., 47): "), 16)

    Kbch = get_kbch(fecframe, rate)
    if not (0 <= DFL <= Kbch - 80):
        raise ValueError(f"DFL must satisfy 0 ≤ DFL ≤ {Kbch - 80}")

    csv_path = resolve_input_path(BITS_CSV_PATH)
    in_bits = load_bits_csv(csv_path)
    report.log_input_data(csv_path, in_bits)

    if UPL == 0:
        mode_stream = in_bits
        stream_ptr = 0
        packetized = False
        df_global_pos = 0
    else:
        crc_stream = PacketizedCrc8Stream(in_bits, UPL)
        packetized = True
        df_global_pos = 0

    frames = 0

    def write_bits_single_line(path: str, bits: np.ndarray):
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join("1" if int(b) else "0" for b in bits.reshape(-1)))

    def write_symbols(path: str, syms: np.ndarray):
        with open(path, "w", encoding="utf-8") as f:
            for s in syms.reshape(-1):
                f.write(f"{s.real:+.6f} {s.imag:+.6f}\n")

    def write_bb_samples(path: str, samps: np.ndarray):
        with open(path, "w", encoding="utf-8") as f:
            for s in samps.reshape(-1):
                f.write(f"{s.real:+.6f} {s.imag:+.6f}\n")

    def write_rf_samples(path: str, samps: np.ndarray):
        with open(path, "w", encoding="utf-8") as f:
            for s in samps.reshape(-1):
                f.write(f"{s:+.6f}\n")

    def iq_upconvert(bb: np.ndarray, fs: float, fc: float, phase0: float = 0.0) -> np.ndarray:
        if fc <= 0 or fc >= fs / 2:
            raise ValueError("fc must be in (0, fs/2).")
        n = np.arange(len(bb), dtype=np.float64)
        t = n / fs
        lo = np.exp(1j * (2.0 * np.pi * fc * t + phase0))
        rf = np.real(bb * lo)
        return rf.astype(np.float64)

    while frames < MAX_FRAMES:

        # -----------------------------
        # BBFRAME build
        # -----------------------------
        if DFL == 0:
            DF = np.array([], dtype=np.uint8)
        elif packetized:
            DF = crc_stream.read_bits(DFL)
        else:
            DF = mode_stream[stream_ptr:stream_ptr + DFL]
            stream_ptr += DFL

        SYNCD = compute_syncd_packetized(df_global_pos, DFL, UPL) if packetized else 0
        df_global_pos += DFL if packetized else 0

        BBHEADER = build_bbheader(
            matype1=0x00,
            matype2=0x00,
            upl=UPL,
            dfl=DFL,
            sync=SYNC,
            syncd=SYNCD
        )

        BBFRAME = np.concatenate([BBHEADER, DF])
        frames += 1

        report.log_bbheader(BBHEADER, 0x00, UPL, DFL, SYNC, SYNCD)
        report.log_merger_slicer(DF, DFL)
        report.log_bbframe(BBFRAME)

        # -----------------------------
        # Stream adaptation (pad + BB scrambling)
        # -----------------------------
        padded_bbframe = pad_bbframe_rate(BBFRAME, fecframe, rate)
        adapted = stream_adaptation_rate(BBFRAME, fecframe, rate)

        # -----------------------------
        # BCH
        # -----------------------------
        Kbch, Nbch, t = BCH_PARAMS[(fecframe, rate)]
        bch_codeword = bch_encode_bbframe(adapted, fecframe, rate)
        report.log_bch_encoding(adapted, bch_codeword, fecframe, rate, Kbch, Nbch, t)

        # -----------------------------
        # LDPC
        # -----------------------------
        ldpc_encoder = DVB_LDPC_Encoder(MAT_PATH)
        ldpc_codeword = ldpc_encoder.encode(bch_codeword, fecframe, rate)

        report.section("LDPC ENCODING")
        report.bits("LDPC codeword", ldpc_codeword)

        # -----------------------------
        # Bit interleaver
        # -----------------------------
        interleaved = dvbs2_bit_interleave(ldpc_codeword, modulation)

        report.section("BIT INTERLEAVING (ETSI 5.3.3)")
        report.bits("Interleaved bits", interleaved)

        # -----------------------------
        # Constellation mapping (payload)
        # -----------------------------
        payload_symbols = dvbs2_constellation_map(
            interleaved,
            modulation,
            code_rate=rate
        )

        report.section("CONSTELLATION MAPPING (ETSI 5.4)")
        report.write(f"Modulation               : {modulation}")
        report.write(f"MODCOD (ETSI)            : {modcod}")
        report.write(f"Payload symbols          : {len(payload_symbols)}")
        report.write("First payload symbols    : " + ", ".join(
            f"{s.real:+.3f}{s.imag:+.3f}j" for s in payload_symbols[:8]
        ))

        # -----------------------------
        # PLHEADER
        # -----------------------------
        plh_bits, plh_syms = build_plheader(modcod, fecframe, pilots_on)

        # -----------------------------
        # PILOT INSERTION (MUST BE BEFORE PL SCRAMBLING)
        # -----------------------------
        payload_with_pilots, pilot_meta = insert_pilots_into_payload(
            payload_symbols,
            pilots_on,
            fecframe=fecframe
        )

        report.section("PILOT INSERTION (ETSI 5.5.3)")
        report.write(f"Pilots enabled           : {pilot_meta['pilots_on']}")
        if pilot_meta["pilots_on"]:
            report.write(f"S (payload slots)        : {pilot_meta['S_slots']}")
            report.write(f"Pilot blocks inserted    : {pilot_meta['pilot_blocks']}")
            report.write(f"Pilot symbol (pre-scr)   : {pilot_meta['pilot_symbol'].real:+.6f}{pilot_meta['pilot_symbol'].imag:+.6f}j")
            report.write(f"Payload syms in/out      : {pilot_meta['payload_symbols_in']} → {pilot_meta['payload_symbols_out']}")
        else:
            report.write("Pilot blocks inserted    : 0")

        # Build PLFRAME BEFORE scrambling: PLHEADER + (payload with pilots)
        plframe_pre_scramble = np.concatenate([plh_syms, payload_with_pilots])

        report.section("PLFRAME (PRE-SCRAMBLE)")
        report.write(f"PLHEADER symbols         : {len(plh_syms)}")
        report.write(f"PLFRAME symbols (pre)    : {len(plframe_pre_scramble)}")
        report.write("First PLFRAME symbols (pre): " + ", ".join(
            f"{s.real:+.3f}{s.imag:+.3f}j" for s in plframe_pre_scramble[:8]
        ))

        # -----------------------------
        # PL SCRAMBLING (exclude PLHEADER)
        # -----------------------------
        symbols = pl_scramble_full_plframe(
            plframe_pre_scramble,
            scrambling_code,
            plheader_len=len(plh_syms)
        )

        report.section("PL SCRAMBLING (ETSI 5.5.4)")
        report.write(f"Scrambling code          : {scrambling_code}")
        report.write(f"PLFRAME symbols (post)   : {len(symbols)}")
        report.write("First PLFRAME symbols (post): " + ", ".join(
            f"{s.real:+.3f}{s.imag:+.3f}j" for s in symbols[:8]
        ))

        # -----------------------------
        # Baseband RRC filter (after PL scrambling)
        # -----------------------------
        bb_samples, rrc_taps = dvbs2_bb_filter(
            symbols,
            alpha=alpha,
            sps=rrc_sps,
            span=rrc_span
        )
        rf_samples = None
        if upconvert:
            fs = Rs * rrc_sps
            rf_samples = iq_upconvert(bb_samples, fs=fs, fc=fc, phase0=phase0)
            report.section("RF UPCONVERSION")
            report.write(f"Symbol rate (Rs)         : {Rs}")
            report.write(f"Sample rate (fs)         : {fs}")
            report.write(f"Carrier frequency (fc)   : {fc}")
            report.write(f"Carrier phase (rad)      : {phase0}")
            report.write("First RF samples         : " + ", ".join(
                f"{s:+.6f}" for s in rf_samples[:8]
            ))
        plot_fs_bb = (Rs * rrc_sps) if upconvert else 1.0
        plot_fs_sym = Rs if upconvert else 1.0
        plot_qpsk_fft(
            bb_samples,
            fs=plot_fs_bb,
            title="DVB-S2 Baseband Spectrum (RRC)"
        )
        if modulation == "QPSK":
            plot_qpsk_fft(
                symbols,
                fs=plot_fs_sym,
                title="DVB-S2 QPSK PLFRAME Spectrum"
            )
            plot_dvbs2_qpsk_spectrum(
                symbols,
                alpha=alpha,
                sps=rrc_sps,
                span=rrc_span
            )

        # -----------------------------
        # Sanity check: interleaver round-trip
        # -----------------------------
        recovered = dvbs2_bit_deinterleave(interleaved, modulation)
        report.section("INTERLEAVER SANITY CHECK")
        if not np.array_equal(recovered, ldpc_codeword):
            report.write("WARNING: Interleaver round-trip mismatch")
        else:
            report.write("Interleaver round-trip: OK")

        # -----------------------------
        # Save outputs
        # -----------------------------
        write_bits_single_line(f"ldpc_{frames}.txt", ldpc_codeword)
        write_bits_single_line(f"interleaved_{frames}.txt", interleaved)
        write_symbols(f"symbols_{frames}.txt", symbols)
        write_bb_samples(f"bb_samples_{frames}.txt", bb_samples)
        if rf_samples is not None:
            write_rf_samples(f"rf_samples_{frames}.txt", rf_samples)

        save_bbframe_to_file_rate(
            padded_bbframe,
            adapted,
            f"bbframe_{frames}.txt",
            fecframe,
            rate,
            output_mode="both_single_line"
        )

        print(f"Frame {frames}: OK")

    report.close()
    print(f"\nReport written to: {REPORT_FILE}")


if __name__ == "__main__":
    run_dvbs2_transmitter()
