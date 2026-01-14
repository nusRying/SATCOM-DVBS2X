print("\n========== DVB-S2 TRANSMITTER RUN ==========\n")

import numpy as np
import os

# ------------------------------------------------------------
# Import modules
# ------------------------------------------------------------

from BB_Frame import (
    dvbs2_bbframe_generator_from_bits_csv,
    build_bbheader,
    PacketizedCrc8Stream,
    compute_syncd_packetized,
    load_bits_csv,
    resolve_input_path
)

from stream_adaptation import (
    get_kbch,
    pad_bbframe_rate,
    save_bbframe_to_file_rate,
    stream_adaptation_rate
)

from bbframe_report import BBFrameReport
from bch_encoding import BCH_PARAMS, bch_encode_bbframe
from ldpc_Encoding import DVB_LDPC_Encoder

from bit_interleaver import (
    dvbs2_bit_interleave,
    dvbs2_bit_deinterleave
)

from constellation_mapper import dvbs2_constellation_map
from pl_header import modcod_from_modulation_rate, build_plheader
from pl_scrambler import pl_scramble_full_plframe

# ------------------------------------------------------------
# User configuration
# ------------------------------------------------------------

BITS_CSV_PATH = r"C:\Users\umair\Videos\JOB - NASTP\SATCOM\Code\GS_data\umair_gs_bits.csv"
MAX_FRAMES    = 3
REPORT_FILE   = "dvbs2_full_report.txt"
MAT_PATH      = r"C:\Users\umair\Videos\JOB - NASTP\SATCOM\Code\s2xLDPCParityMatrices\dvbs2xLDPCParityMatrices.mat"

# ------------------------------------------------------------
# MAIN RUN
# ------------------------------------------------------------

def run_dvbs2_transmitter():
    report = BBFrameReport(REPORT_FILE)

    # -----------------------------
    # User inputs
    # -----------------------------
    stream_type = input("Enter stream type (TS or GS): ").strip().upper()
    fecframe    = input("Enter FECFRAME type (normal/short): ").strip().lower()
    rate        = input("Enter code rate (e.g., 1/2, 3/5, 2/3, 3/4, 5/6, 8/9, 9/10): ").strip()
    modulation  = input("Enter modulation (QPSK, 8PSK, 16APSK, 32APSK): ").strip().upper()
    pilots_in   = input("Enable pilots (pilots_on/pilots_off): ").strip().lower()
    alpha       = float(input("Enter roll-off alpha (0.35 / 0.25 / 0.20): ").strip())
    scr_in      = input("Enter PL scrambling code (0..262142, default 0): ").strip()
    scrambling_code = int(scr_in) if scr_in else 0
    DFL         = int(input("Enter DFL: "))

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

    # -----------------------------
    # Load input bits
    # -----------------------------
    csv_path = resolve_input_path(BITS_CSV_PATH)
    in_bits = load_bits_csv(csv_path)
    report.log_input_data(csv_path, in_bits)

    # -----------------------------
    # Mode adaptation setup
    # -----------------------------
    if UPL == 0:
        mode_stream = in_bits
        stream_ptr = 0
        packetized = False
    else:
        crc_stream = PacketizedCrc8Stream(in_bits, UPL)
        packetized = True
        df_global_pos = 0

    frames = 0

    def write_bits_single_line(path: str, bits: np.ndarray):
        with open(path, "w") as f:
            f.write("".join("1" if b else "0" for b in bits))

    def write_symbols(path: str, syms: np.ndarray):
        with open(path, "w") as f:
            for s in syms:
                f.write(f"{s.real:+.6f} {s.imag:+.6f}\n")

    # -----------------------------
    # FRAME LOOP
    # -----------------------------
    while frames < MAX_FRAMES:

        # ----- DATA FIELD -----
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
        # STREAM ADAPTATION
        # -----------------------------
        padded_bbframe = pad_bbframe_rate(BBFRAME, fecframe, rate)
        adapted = stream_adaptation_rate(BBFRAME, fecframe, rate)

        # -----------------------------
        # BCH ENCODING
        # -----------------------------
        Kbch, Nbch, t = BCH_PARAMS[(fecframe, rate)]
        bch_codeword = bch_encode_bbframe(adapted, fecframe, rate)
        report.log_bch_encoding(adapted, bch_codeword, fecframe, rate, Kbch, Nbch, t)

        # -----------------------------
        # LDPC ENCODING
        # -----------------------------
        ldpc_encoder = DVB_LDPC_Encoder(MAT_PATH)
        ldpc_codeword = ldpc_encoder.encode(bch_codeword, fecframe, rate)

        report.section("LDPC ENCODING")
        report.bits("LDPC codeword", ldpc_codeword)

        # -----------------------------
        # BIT INTERLEAVING (ETSI 5.3.3)
        # -----------------------------
        interleaved = dvbs2_bit_interleave(ldpc_codeword, modulation)

        report.section("BIT INTERLEAVING (ETSI 5.3.3)")
        report.bits("Interleaved bits", interleaved)

        # -----------------------------
        # CONSTELLATION MAPPING (ETSI 5.4)
        # -----------------------------
        payload_symbols = dvbs2_constellation_map(
            interleaved,
            modulation,
            code_rate=rate
        )

        plheader_bits, plheader_symbols = build_plheader(modcod, fecframe, pilots_on)
        plframe_symbols = np.concatenate([plheader_symbols, payload_symbols])
        report.section("PL SCRAMBLING (INTERMEDIATE)")
        report.write(f"PLHEADER symbols         : {len(plheader_symbols)}")
        report.write(f"PLFRAME symbols (pre)    : {len(plframe_symbols)}")
        report.write("First PLFRAME symbols (pre): " + ", ".join(
            f"{s.real:+.3f}{s.imag:+.3f}j" for s in plframe_symbols[:8]
        ))
        symbols = pl_scramble_full_plframe(
            plframe_symbols,
            scrambling_code,
            plheader_len=plheader_symbols.size
        )
        report.write(f"PLFRAME symbols (post)   : {len(symbols)}")
        report.write("First PLFRAME symbols (post): " + ", ".join(
            f"{s.real:+.3f}{s.imag:+.3f}j" for s in symbols[:8]
        ))

    report.section("CONSTELLATION MAPPING (ETSI 5.4)")
    report.write(f"Modulation               : {modulation}")
    report.write(f"MODCOD (ETSI)            : {modcod}")
    report.write(f"Total payload symbols    : {len(payload_symbols)}")
    report.write("First payload symbols    : " + ", ".join(
        f"{s.real:+.3f}{s.imag:+.3f}j" for s in payload_symbols[:8]
    ))

    report.section("PLHEADER / PL SCRAMBLING")
    report.write(f"Scrambling code          : {scrambling_code}")
    report.write(f"PLHEADER symbols         : {len(plheader_symbols)}")
    report.write(f"PLFRAME symbols          : {len(symbols)}")
    report.write("First PLFRAME symbols    : " + ", ".join(
        f"{s.real:+.3f}{s.imag:+.3f}j" for s in symbols[:8]
    ))

    # -----------------------------
    # SANITY CHECK
    # -----------------------------
    recovered = dvbs2_bit_deinterleave(interleaved, modulation)
    if not np.array_equal(recovered, ldpc_codeword):
        report.write("WARNING: Interleaver round-trip mismatch")
    else:
        report.write("Interleaver round-trip: OK")

    # -----------------------------
    # SAVE OUTPUT FILES
    # -----------------------------
    write_bits_single_line(f"ldpc_{frames}.txt", ldpc_codeword)
    write_bits_single_line(f"interleaved_{frames}.txt", interleaved)
    write_symbols(f"symbols_{frames}.txt", symbols)

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

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

if __name__ == "__main__":
    run_dvbs2_transmitter()
