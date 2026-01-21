# ============================================================
#  BBFRAME REPORT GENERATOR
#  (Pure logging / reporting – NO signal processing)
# ============================================================

import datetime
import numpy as np


# ------------------------------------------------------------
#  Helper functions
# ------------------------------------------------------------

def bits_to_str(bits, max_len=None):
    """
    Convert bit array to string (truncated for readability).
    """
    bits = list(bits)
    shown = bits if max_len is None else bits[:max_len]
    s = "".join(str(int(b)) for b in shown)
    if max_len is not None and len(bits) > max_len:
        s += " ..."
    return s


def byte_to_bitstr(b: int) -> str:
    """
    Convert a byte to MSB-first bit string.
    """
    return format(b & 0xFF, "08b")


# ------------------------------------------------------------
#  Report Logger Class
# ------------------------------------------------------------

class BBFrameReport:
    """
    Human-readable step-by-step report generator for DVB-S2 BBFRAME processing.
    """

    def __init__(self, filename="bbframe_report.txt"):
        self.filename = filename
        self.f = open(filename, "w", encoding="utf-8")

        self._write_header()

    # --------------------------------------------------------
    #  Core write utilities
    # --------------------------------------------------------

    def write(self, text=""):
        self.f.write(text + "\n")

    def section(self, title):
        self.write("")
        self.write("=" * 60)
        self.write(title)
        self.write("=" * 60)

    def bits(self, label, bits, max_len=None):
        self.write(f"{label}:")
        self.write(f"  Length : {len(bits)} bits")
        self.write(f"  Bits   : {bits_to_str(bits, max_len)}")
        self.write("")

    # --------------------------------------------------------
    #  Report sections
    # --------------------------------------------------------

    def _write_header(self):
        self.write("DVB-S2 BASEBAND FRAME PROCESSING REPORT")
        self.write(f"Generated on: {datetime.datetime.now()}")
        self.write("")
        self.write("This report documents step-by-step processing")
        self.write("from input bitstream to BBFRAME output.")
        self.write("")

    def log_input_data(self, csv_path, in_bits):
        self.section("INPUT DATA")
        self.write(f"Input CSV file : {csv_path}")
        self.write(f"Total bits     : {len(in_bits)}")
        self.bits("First input bits", in_bits)

    def log_configuration(
        self,
        stream_type,
        fecframe,
        upl,
        dfl,
        rolloff,
        sync
    ):
        self.section("CONFIGURATION")
        self.write(f"Stream type    : {stream_type}")
        self.write(f"FECFRAME       : {fecframe}")
        self.write(f"UPL            : {upl} bits")
        self.write(f"DFL            : {dfl} bits")
        self.write(f"Roll-off alpha : {rolloff}")
        self.write(f"SYNC byte      : 0x{sync:02X}")
        self.write("")

    def log_crc_mode_adapter(
        self,
        payload0_bits,
        crc0,
        payload1_bits
    ):
        """
        Explicit ETSI-compliant CRC-8 behavior visualization.
        """
        self.section("CRC-8 MODE ADAPTER (ETSI CLAUSE 5.1.4)")

        self.write("CRC-8 is computed over the payload of a User Packet.")
        self.write("The CRC replaces the sync byte of the FOLLOWING packet.")
        self.write("")

        self.write("Resulting stream structure:")
        self.write("[sync0][payload0][CRC0][payload1]")
        self.write("")

        self.write(f"0x47  = {byte_to_bitstr(0x47)}")
        self.write(f"CRC₀  = 0x{crc0:02X} = {byte_to_bitstr(crc0)}")
        self.write("")

        self.bits("payload0", payload0_bits)
        self.bits("payload1 (first bits)", payload1_bits)

    def log_merger_slicer(self, df_bits, dfl):
        self.section("MERGER / SLICER (ETSI CLAUSE 5.1.5)")
        self.write(f"DATA FIELD length (DFL): {dfl} bits")
        self.bits("DATA FIELD (DF)", df_bits)

    def log_bbheader(
        self,
        bbheader_bits,
        matype1,
        upl,
        dfl,
        sync,
        syncd
    ):
        self.section("BBHEADER (ETSI CLAUSE 5.1.6)")
        self.bits("BBHEADER", bbheader_bits)
        self.write(f"MATYPE-1 : 0x{matype1:02X}")
        self.write(f"UPL      : {upl}")
        self.write(f"DFL      : {dfl}")
        self.write(f"SYNC     : 0x{sync:02X}")
        self.write(f"SYNCD    : {syncd}")
        self.write("")

    def log_bbframe(self, bbframe_bits):
        self.section("FINAL BBFRAME")
        self.bits("BBFRAME", bbframe_bits)
        self.write(f"BBFRAME length : {len(bbframe_bits)} bits")

    def log_bch_encoding(
        self,
        bbframe_kbch_bits,
        bch_codeword_bits,
        fecframe,
        rate,
        kbch,
        nbch,
        t
    ):
        self.section("BCH ENCODING (ETSI CLAUSE 5.3.1)")
        self.write(f"FECFRAME : {fecframe}")
        self.write(f"Rate     : {rate}")
        self.write(f"Kbch     : {kbch} bits")
        self.write(f"Nbch     : {nbch} bits")
        self.write(f"t        : {t}")
        self.bits("Input BBFRAME (Kbch)", bbframe_kbch_bits)
        self.bits("BCH codeword (Nbch)", bch_codeword_bits)

    def log_bch_parity(self, parity_bits):
        self.section("BCHFEC PARITY BITS")
        self.bits("Parity bits (Nbch-Kbch)", parity_bits)

    def log_bit_interleaving(self, interleaved_bits, modulation, filename=None):
        """
        Log bit interleaving results.

        Parameters
        ----------
        interleaved_bits : array-like
            The interleaved LDPC codeword bits.
        modulation : str
            Modulation string (QPSK, 8PSK, ...)
        filename : str, optional
            Optional filename where interleaved bits were saved.
        """
        self.section("BIT INTERLEAVING (ETSI CLAUSE 5.3.3)")
        self.write(f"Modulation         : {modulation}")
        if filename:
            self.write(f"Saved to file      : {filename}")
        self.bits("Interleaved bits (first view)", interleaved_bits)

    # --------------------------------------------------------
    #  Finalization
    # --------------------------------------------------------

    def close(self):
        self.write("")
        self.write("END OF REPORT")
        self.f.close()
