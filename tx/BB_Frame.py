# Stage 02 - BB Frame - BB header & frame build
print("\n========== SCRIPT START ==========\n")

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import csv
from tx.bbframe_report import BBFrameReport
from tx.stream_adaptation import get_kbch, stream_adaptation_rate


# ============================================================
#  DVB-S2 CONSTANTS (ETSI EN 302 307 V1.3.1)
# ============================================================

CRC8_POLY = 0xD5  # g(x)=x^8+x^7+x^6+x^4+x^2+1  (clause 5.1.4)

KBCH_NORMAL = 64800
KBCH_SHORT  = 16200

SYNC_TS = 0x47  # MPEG-TS sync byte


# ============================================================
#  Path helper (robust, best practice)
# ============================================================

def resolve_input_path(path: str) -> str:
    """
    Resolve input path robustly:
    - Absolute path: use directly
    - Relative path: resolve relative to this script directory
    - Validate existence
    """
    if os.path.isabs(path):
        resolved = path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        resolved = os.path.join(script_dir, path)

    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"Input CSV not found:\n"
            f"  Provided: {path}\n"
            f"  Resolved: {resolved}\n"
            f"Tip: Put the CSV next to this script or pass an absolute path."
        )
    return resolved


# ============================================================
#  CRC-8 (ETSI clause 5.1.4 and 5.1.6)
# ============================================================

def dvbs2_crc8(bitstream_bits: np.ndarray) -> int:
    crc = 0
    for bit in bitstream_bits:
        msb = (crc >> 7) & 1
        xor_in = msb ^ int(bit)
        crc = ((crc << 1) & 0xFF)
        if xor_in:
            crc ^= CRC8_POLY
    return crc


# ============================================================
#  MATYPE-1 builder (Table 3, clause 5.1.6)
#  MATYPE-1 bits: [TS/GS(2), SIS/MIS(1), CCM/ACM(1), ISSYI(1), NPD(1), RO(2)]
# ============================================================

def build_matype1(ts_gs_bits: int, sis: int, ccm: int, issyi: int, npd: int, ro_bits: int) -> int:
    v = 0
    v |= (ts_gs_bits & 0b11) << 6
    v |= (sis & 0b1) << 5
    v |= (ccm & 0b1) << 4
    v |= (issyi & 0b1) << 3
    v |= (npd & 0b1) << 2
    v |= (ro_bits & 0b11)
    return v

def ro_to_bits(alpha: float) -> int:
    if abs(alpha - 0.35) < 1e-9:
        return 0b00
    if abs(alpha - 0.25) < 1e-9:
        return 0b01
    if abs(alpha - 0.20) < 1e-9:
        return 0b10
    raise ValueError("RO alpha must be one of: 0.35, 0.25, 0.20")


# ============================================================
#  BBHEADER builder (clause 5.1.6)
#  10 bytes total = 80 bits:
#   MATYPE-1 (8)
#   MATYPE-2 (8)
#   UPL (16)
#   DFL (16)
#   SYNC (8)
#   SYNCD (16)
#   CRC-8 (8) computed over first 9 bytes (72 bits)
# ============================================================

def build_bbheader(matype1: int, matype2: int, upl: int, dfl: int, sync: int, syncd: int) -> np.ndarray:
    def bits(value: int, width: int):
        return [int(b) for b in format(value & ((1 << width) - 1), f"0{width}b")]

    header72 = []
    header72 += bits(matype1, 8)
    header72 += bits(matype2, 8)
    header72 += bits(upl, 16)
    header72 += bits(dfl, 16)
    header72 += bits(sync, 8)
    header72 += bits(syncd, 16)

    if len(header72) != 72:
        raise RuntimeError(f"BBHEADER first 9 bytes must be 72 bits, got {len(header72)}")

    crc = dvbs2_crc8(np.array(header72, dtype=np.uint8))
    full = header72 + bits(crc, 8)

    if len(full) != 80:
        raise RuntimeError(f"BBHEADER must be 80 bits, got {len(full)}")

    return np.array(full, dtype=np.uint8)


# ============================================================
#  CSV loader (single column, one bit per row)
# ============================================================

def load_bits_csv(path: str) -> np.ndarray:
    """
    Load a single-column CSV containing 0/1 bits into a NumPy array.
    Accepts files where each row is '0' or '1' (optionally with spaces).
    """
    bits = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            s = row[0].strip()
            if s == "":
                continue
            b = int(s)
            if b not in (0, 1):
                raise ValueError(f"CSV must contain only 0/1 bits. Found: {b}")
            bits.append(b)

    if len(bits) == 0:
        raise ValueError("CSV appears empty or contains no valid bits.")

    return np.array(bits, dtype=np.uint8)


# ============================================================
#  CRC-8 ENCODER OUTPUT STREAM FOR PACKETIZED INPUTS (clause 5.1.4)
# ============================================================

class PacketizedCrc8Stream:
    """
    Sequential-read view of the MODE ADAPTER output stream after CRC-8 encoding.
    Input must contain original sync bytes at the start of each UP.
    For packetized TS: UPL = 1504 bits (188 bytes).
    """

    def __init__(self, in_bits: np.ndarray, upl_bits: int):
        if upl_bits <= 0:
            raise ValueError("PacketizedCrc8Stream requires UPL > 0")
        if upl_bits < 8:
            raise ValueError("UPL must be >= 8 bits for packetized streams")

        self.in_bits = in_bits
        self.upl = upl_bits
        self.ptr = 0
        self.prev_crc = None
        self.global_out_pos = 0
        self._current_up_out = None

    def _read_exact(self, n: int) -> np.ndarray:
        end = self.ptr + n
        if end > len(self.in_bits):
            chunk = self.in_bits[self.ptr:]
            pad = np.zeros(end - len(self.in_bits), dtype=np.uint8)
            self.ptr = len(self.in_bits)
            return np.concatenate([chunk, pad])
        chunk = np.array(self.in_bits[self.ptr:end], dtype=np.uint8, copy=False)
        self.ptr = end
        return chunk

    def read_bits(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.uint8)
        out_i = 0

        while out_i < n:
            pos_in_up = self.global_out_pos % self.upl
            remaining_in_up = self.upl - pos_in_up
            take = min(n - out_i, remaining_in_up)

            if pos_in_up == 0:
                up_in = self._read_exact(self.upl)
                up_sync_in = up_in[:8]
                up_payload = up_in[8:]

                if self.prev_crc is None:
                    up_sync_out = up_sync_in
                else:
                    up_sync_out = np.array(
                        [int(b) for b in format(self.prev_crc, "08b")],
                        dtype=np.uint8
                    )

                self.prev_crc = dvbs2_crc8(up_payload)
                self._current_up_out = np.concatenate([up_sync_out, up_payload])

            out[out_i:out_i + take] = self._current_up_out[pos_in_up:pos_in_up + take]
            out_i += take
            self.global_out_pos += take

        return out


# ============================================================
#  SYNCD computation (clauses 5.1.5 / 5.1.6)
# ============================================================

def compute_syncd_packetized(df_start_global_pos: int, dfl: int, upl: int) -> int:
    mod = df_start_global_pos % upl
    dist = 0 if mod == 0 else (upl - mod)
    if dist >= dfl:
        return 0xFFFF
    return dist


# ============================================================
#  MAIN ETSI-COMPLIANT PIPELINE USING CSV
# ============================================================

def dvbs2_bbframe_generator_from_bits_csv(bits_csv_path: str, max_frames: int = 10, report_filename: str = "bbframe_report.txt") -> None:
    # Initialize report logger
    report = BBFrameReport(filename=report_filename)

    # -----------------------------
    # USER INPUTS
    # -----------------------------
    stream_type = input("Enter stream type (TS or GS): ").strip().upper()
    fecframe    = input("Enter FECFRAME type (normal/short): ").strip().lower()
    rate        = input("Enter code rate (e.g., 1/2, 3/5, 2/3, 3/4, 5/6, 8/9, 9/10): ").strip()
    alpha       = float(input("Enter roll-off alpha (0.35 / 0.25 / 0.20): ").strip())

    Kbch = get_kbch(fecframe, rate)
    DFL = int(input(f"Enter DFL (0..{Kbch-80}): "))
    if not (0 <= DFL <= Kbch - 80):
        raise ValueError("DFL must satisfy 0 <= DFL <= Kbch-80 (clause 5.1.5).")

    if stream_type == "TS":
        UPL = 188 * 8
        SYNC = SYNC_TS
        ts_gs_bits = 0b11  # TS
    elif stream_type == "GS":
        UPL = int(input("Enter UPL in bits (0 for continuous GS): ").strip())
        if not (0 <= UPL <= 65535):
            raise ValueError("UPL must be in range 0..65535 (clause 5.1.6).")
        ts_gs_bits = 0b01 if UPL == 0 else 0b00
        SYNC = int(input("Enter SYNC byte in hex (e.g., 47 or 00): ").strip(), 16) & 0xFF
    else:
        raise ValueError("stream_type must be TS or GS")

    sis = 1
    ccm = 1
    issyi = 0
    if stream_type == "TS":
        npd = int(input("Null-packet deletion active? (0/1): ").strip())
    else:
        npd = 0

    ro_bits = ro_to_bits(alpha)
    MATYPE1 = build_matype1(ts_gs_bits, sis, ccm, issyi, npd, ro_bits)
    MATYPE2 = 0x00

    # -----------------------------
    # LOAD INPUT BITS FROM CSV (robust path)
    # -----------------------------
    csv_path = resolve_input_path(bits_csv_path)
    in_bits = load_bits_csv(csv_path)
    print(f"Loaded {len(in_bits)} bits from: {csv_path}")

    # Log input data
    report.log_input_data(csv_path, in_bits)

    # Log configuration
    report.log_configuration(
        stream_type=stream_type,
        fecframe=fecframe,
        upl=UPL,
        dfl=DFL,
        rolloff=alpha,
        sync=SYNC
    )

    # Optional sanity check for TS
    if stream_type == "TS" and len(in_bits) >= 8:
        first_byte = int("".join(str(x) for x in in_bits[:8]), 2)
        if first_byte != 0x47:
            print(f"Warning: first 8 bits are 0x{first_byte:02X}, expected 0x47 for TS input.")

    # -----------------------------
    # CREATE MODE ADAPTER OUTPUT STREAM VIEW
    # -----------------------------
    if UPL == 0:
        mode_adapter_stream = in_bits
        stream_ptr = 0
        df_global_pos = 0
        packetized = False
    else:
        crc_stream = PacketizedCrc8Stream(in_bits, UPL)
        df_global_pos = 0
        packetized = True
        
        # Log CRC-8 mode adapter behavior for first two packets
        if len(in_bits) >= UPL * 2:
            # Extract first two User Packets from input
            up0_bits = in_bits[:UPL]
            up0_sync = up0_bits[:8]
            up0_payload = up0_bits[8:]
            
            up1_bits = in_bits[UPL:UPL*2]
            up1_payload = up1_bits[8:]
            
            # Compute CRC-8 of first payload
            crc0 = dvbs2_crc8(up0_payload)
            
            report.log_crc_mode_adapter(
                payload0_bits=up0_payload[:min(32, len(up0_payload))],
                crc0=crc0,
                payload1_bits=up1_payload[:min(32, len(up1_payload))]
            )
            # Also log the explicit concatenation view: [sync0][payload0][CRC0][payload1]
            crc0_bits = np.array([int(b) for b in format(crc0, "08b")], dtype=np.uint8)
            try:
                concat_view = np.concatenate([up0_sync, up0_payload, crc0_bits, up1_payload])
            except Exception:
                # Fallback to a safe construction if shapes are surprising
                concat_view = np.hstack((np.asarray(up0_sync, dtype=np.uint8),
                                         np.asarray(up0_payload, dtype=np.uint8),
                                         crc0_bits,
                                         np.asarray(up1_payload, dtype=np.uint8)))

            report.section("MODE ADAPTER CONCATENATED VIEW")
            report.write("[sync0][payload0][CRC0][payload1]")
            report.bits("Concatenated bits (first view)", concat_view, max_len=len(concat_view))

    frames = 0
    while frames < max_frames:
        # -----------------------------
        # READ DFL BITS AS DATA FIELD (Merger/Slicer clause 5.1.5)
        # -----------------------------
        if DFL == 0:
            DF = np.array([], dtype=np.uint8)
        else:
            if not packetized:
                end = stream_ptr + DFL
                if end <= len(mode_adapter_stream):
                    DF = np.array(mode_adapter_stream[stream_ptr:end], dtype=np.uint8, copy=False)
                else:
                    tail = mode_adapter_stream[stream_ptr:]
                    pad = np.zeros(end - len(mode_adapter_stream), dtype=np.uint8)
                    DF = np.concatenate([tail, pad])
                stream_ptr = min(end, len(mode_adapter_stream))
            else:
                DF = crc_stream.read_bits(DFL)

        # Log DATA FIELD (Merger/Slicer)
        if frames == 1:  # Log details only for first frame to avoid redundancy
            report.log_merger_slicer(DF, DFL)

        # -----------------------------
        # SYNCD (clause 5.1.6)
        # -----------------------------
        if packetized:
            SYNCD = compute_syncd_packetized(df_start_global_pos=df_global_pos, dfl=DFL, upl=UPL)
        else:
            SYNCD = 0x0000  # reserved for continuous GS

        df_global_pos += DFL

        # -----------------------------
        # BUILD BBHEADER (clause 5.1.6)
        # -----------------------------
        BBHEADER = build_bbheader(
            matype1=MATYPE1,
            matype2=MATYPE2,
            upl=UPL,
            dfl=DFL,
            sync=SYNC,
            syncd=SYNCD
        )

        # Log BBHEADER details only for first frame
        if frames == 1:
            report.log_bbheader(
                bbheader_bits=BBHEADER,
                matype1=MATYPE1,
                upl=UPL,
                dfl=DFL,
                sync=SYNC,
                syncd=SYNCD
            )

        BBFRAME = np.concatenate([BBHEADER, DF])
        frames += 1

        # Log BBFRAME for first frame
        if frames == 1:
            report.log_bbframe(BBFRAME)

        # -----------------------------
        # STREAM ADAPTATION (padding + scrambling)
        # -----------------------------
        adapted = stream_adaptation_rate(BBFRAME, fecframe, rate)
        Kbch = get_kbch(fecframe, rate)
        padding_len = Kbch - len(BBFRAME)

        report.section("STREAM ADAPTATION")
        report.write(f"BBFRAME index   : {frames}")
        report.write(f"Padding applied : {padding_len} bits")
        if padding_len > 0:
            padding_bits = np.zeros(padding_len, dtype=np.uint8)
            report.bits("Padding bits (first view)", padding_bits, max_len=64)
        else:
            report.write("Padding bits   : none")
        report.write("Padding + BB Scrambling applied")
        report.write(f"Final BBFRAME length: {len(adapted)} bits (Kbch={Kbch})")
        report.bits("Scrambled BBFRAME (first 64 bits)", adapted, max_len=64)

        print(f"BBFRAME {frames}: {len(BBFRAME)} bits (BBHEADER=80 + DF={DFL}) | SYNCD={SYNCD}")

    # Close report
    report.close()
    print(f"\nReport saved to: {report_filename}")

    return


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    # Put your CSV here. Absolute path works, or use a relative path like:
    # "data/GS_data/umair_gs_bits.csv" if the file is in a folder next to this script.
    bits_csv_path = os.path.join(ROOT, "data", "GS_data", "umair_gs_bits.csv")
    dvbs2_bbframe_generator_from_bits_csv(
        bits_csv_path=bits_csv_path,
        max_frames=10
        
    )
