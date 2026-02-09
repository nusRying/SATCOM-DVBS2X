# ============================================================
#  TEST SUITE FOR DVB-S2 MERGER/SLICER PIPELINE  (CORRECTED)
# ============================================================
import numpy as np
import builtins
from PL_Frame import (
    dvbs2_crc8,
    convert_to_bits,
    generate_ts_packet,
    generate_gs_packet,
    slice_stream,
    compute_sync_distance,
    build_bbheader,
    generate_dummy_plframe,
    dvbs2_merger_slicer_pipeline
)


# ------------------------------------------------------------
#  CRC TESTS
# ------------------------------------------------------------
def test_crc8_known_pattern():
    bits = np.array([1,0,1,0,1,0,1,0], dtype=np.uint8)
    crc = dvbs2_crc8(bits)
    print("CRC(10101010) =", hex(crc))
    assert crc == 0x1D, "CRC-8 known reference mismatch"


# ------------------------------------------------------------
#  TS PACKET STRUCTURE TESTS
# ------------------------------------------------------------
def test_ts_packet_size():
    bits = convert_to_bits("abc")
    pkt = generate_ts_packet(bits, 0x47)
    assert len(pkt) == 1504, "TS packet must be 1504 bits"


def test_ts_sync_byte():
    pkt = generate_ts_packet(convert_to_bits("hello"), 0x47)
    sync = int("".join(str(b) for b in pkt[:8]), 2)
    assert sync == 0x47, "TS sync byte incorrect"


def test_crc_replaces_next_sync():
    data = convert_to_bits("test message")
    pkt1 = generate_ts_packet(data, 0x47)
    crc_val = dvbs2_crc8(pkt1[8:])
    pkt2 = generate_ts_packet(data, crc_val)

    sync2 = int("".join(str(b) for b in pkt2[:8]), 2)
    assert sync2 == crc_val, "CRC must replace next TS sync byte"


# ------------------------------------------------------------
#  GS PACKET TESTS
# ------------------------------------------------------------
def test_gs_packet_length():
    bits = convert_to_bits("TEST")
    pkt = generate_gs_packet(bits, upl=64, sync_byte=0x47)
    assert len(pkt) == 64, "GS packet must be exactly UPL bits"


# ------------------------------------------------------------
#  SLICER TESTS
# ------------------------------------------------------------
def test_slice_exact():
    bits = np.random.randint(0, 2, 500)
    DFL = 200
    df, leftover = slice_stream(bits, DFL)
    assert len(df) == 200, "Data Field length incorrect"
    assert len(leftover) == 300, "Leftover bits incorrect"


def test_slice_padding():
    bits = np.random.randint(0, 2, 80)
    DFL = 200
    df, leftover = slice_stream(bits, DFL)

    assert len(df) == 200, "Padding did not extend DF to correct length"
    assert np.all(df[80:] == 0), "Padding bits must be zero"
    assert len(leftover) == 0, "No leftover expected when padding"


# ------------------------------------------------------------
#  SYNC DISTANCE TEST
# ------------------------------------------------------------
def test_syncd_detection():
    sync_bits = np.array([0,1,0,0,0,1,1,1], dtype=np.uint8)
    df = np.concatenate([np.random.randint(0,2,40), sync_bits, np.random.randint(0,2,40)])
    d = compute_sync_distance(df)
    assert d == 40, f"SYNCd incorrect (expected 40, got {d})"


# ------------------------------------------------------------
#  BBHEADER TESTS (CORRECTED)
# ------------------------------------------------------------
def test_bbheader_length():
    header = build_bbheader(
        MATYPE1=0x00,
        MATYPE2=0x00,
        UPL=1504,
        DFL=500,
        SYNC=0x47,
        SYNCd=10
    )
    assert len(header) == 80, "BBHEADER must be exactly 80 bits"


def test_bbheader_crc():
    header = build_bbheader(0, 0, 100, 500, 0x47, 20)

    bits_wo_crc = header[:-8]   # 72 bits before CRC
    assert len(bits_wo_crc) == 72, "BBHEADER must contain 72 bits before CRC"

    crc_expected = dvbs2_crc8(bits_wo_crc)
    crc_in_header = int("".join(str(b) for b in header[-8:]), 2)

    assert crc_expected == crc_in_header, "BBHEADER CRC mismatch"


# ------------------------------------------------------------
#  DUMMY PLFRAME TEST
# ------------------------------------------------------------
def test_dummy_plframe():
    DFL = 200
    dummy = generate_dummy_plframe(DFL)
    assert len(dummy) == 80 + DFL, "Dummy PLFRAME must be header + DF"
    assert np.all(dummy[80:] == 0), "Dummy DF must be zeros"


# ------------------------------------------------------------
#  FULL PIPELINE TESTS
# ------------------------------------------------------------
def test_pipeline_ts():
    inputs = iter(["TS", "normal", "500"])

    def fake_input(_):
        return next(inputs)

    real_input = builtins.input
    builtins.input = fake_input

    PLFRAME = dvbs2_merger_slicer_pipeline()

    builtins.input = real_input

    assert len(PLFRAME) == 80 + 500, "TS PLFRAME size incorrect"


def test_pipeline_gs():
    inputs = iter(["GS", "normal", "500", "200"])

    def fake_input(_):
        return next(inputs)

    real_input = builtins.input
    builtins.input = fake_input

    PLFRAME = dvbs2_merger_slicer_pipeline()

    builtins.input = real_input

    assert len(PLFRAME) == 80 + 500, "GS PLFRAME size incorrect"


# ------------------------------------------------------------
#  RUN ALL TESTS
# ------------------------------------------------------------
def run_all_tests():
    print("\n=== Running DVB-S2 Merger/Slicer Test Suite ===\n")

    test_crc8_known_pattern()
    test_ts_packet_size()
    test_ts_sync_byte()
    test_crc_replaces_next_sync()
    test_gs_packet_length()
    test_slice_exact()
    test_slice_padding()
    test_syncd_detection()
    test_bbheader_length()
    test_bbheader_crc()
    test_dummy_plframe()
    test_pipeline_ts()
    test_pipeline_gs()

    print("\n=== ALL TESTS PASSED SUCCESSFULLY ===\n")


if __name__ == "__main__":
    run_all_tests()
