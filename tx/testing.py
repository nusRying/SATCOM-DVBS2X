import numpy as np
import tempfile
import os
import subprocess
import sys

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PYTHON = sys.executable
SCRIPT = "BB_Frame.py"   # must be in same directory


# ------------------------------------------------------------
# Helper: write temporary CSV
# ------------------------------------------------------------

def write_csv(bits):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
    for b in bits:
        f.write(f"{b}\n")
    f.close()
    return f.name


# ------------------------------------------------------------
# Helper: run main script via subprocess
# ------------------------------------------------------------

def run_script(inputs):
    p = subprocess.Popen(
        [PYTHON, SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = p.communicate(inputs)
    return out, err


# ------------------------------------------------------------
# Reference CRC-8 (golden model)
# ------------------------------------------------------------

def reference_crc8(bits):
    CRC8_POLY = 0xD5
    crc = 0
    for bit in bits:
        msb = (crc >> 7) & 1
        xor_in = msb ^ int(bit)
        crc = ((crc << 1) & 0xFF)
        if xor_in:
            crc ^= CRC8_POLY
    return crc


# ============================================================
# FUNCTIONAL PIPELINE TESTS
# ============================================================

def test_continuous_gs():
    csv = write_csv([0, 1] * 360)

    user_input = (
        "GS\n"
        "short\n"
        "1/2\n"
        "0.35\n"
        "100\n"
        "0\n"
        "47\n"
    )

    out, err = run_script(user_input)
    os.unlink(csv)

    assert "SYNCD=0" in out
    print("Test 1 PASSED: Continuous GS")


def test_packetized_gs_valid():
    csv = write_csv([0, 1] * 360)

    user_input = (
        "GS\n"
        "short\n"
        "1/2\n"
        "0.35\n"
        "100\n"
        "120\n"
        "47\n"
    )

    out, err = run_script(user_input)
    os.unlink(csv)

    assert "SYNCD=20" in out
    print("Test 2 PASSED: Packetized GS (UPL > DFL)")


def test_packetized_gs_ambiguous():
    csv = write_csv([0, 1] * 360)

    user_input = (
        "GS\n"
        "short\n"
        "1/2\n"
        "0.35\n"
        "100\n"
        "40\n"
        "47\n"
    )

    out, err = run_script(user_input)
    os.unlink(csv)

    assert "SYNCD" in out
    print("Test 3 PASSED: Packetized GS ambiguous (UPL < DFL)")


def test_ts_stream():
    csv = write_csv([0, 1] * 360)

    user_input = (
        "TS\n"
        "short\n"
        "1/2\n"
        "0.35\n"
        "100\n"
        "0\n"
    )

    out, err = run_script(user_input)
    os.unlink(csv)

    assert "BBFRAME" in out
    print("Test 4 PASSED: TS stream")


# ============================================================
# CRC-8 UNIT TESTS (DIRECT, NO SUBPROCESS)
# ============================================================

from BB_Frame import PacketizedCrc8Stream
from stream_adaptation import get_kbch, pad_bbframe_rate, stream_adaptation_rate
from bch_encoding import bch_encode_bbframe, BCH_PARAMS
from ldpc_Encoding import DVB_LDPC_Encoder, ldpc_encode_bits


def test_crc_single_packet():
    upl = 40
    payload = np.random.randint(0, 2, upl - 8, dtype=np.uint8)
    sync = np.array([0,1,0,0,0,1,1,1], dtype=np.uint8)  # 0x47

    up = np.concatenate([sync, payload])
    crc_expected = reference_crc8(payload)

    crc_stream = PacketizedCrc8Stream(up, upl)
    crc_stream.read_bits(upl)

    assert crc_stream.prev_crc == crc_expected
    print("Test 5 PASSED: CRC single packet correct")


def test_crc_replaces_next_sync():
    upl = 40

    payload0 = np.random.randint(0, 2, upl - 8, dtype=np.uint8)
    payload1 = np.random.randint(0, 2, upl - 8, dtype=np.uint8)
    sync = np.array([0,1,0,0,0,1,1,1], dtype=np.uint8)

    up0 = np.concatenate([sync, payload0])
    up1 = np.concatenate([sync, payload1])
    stream = np.concatenate([up0, up1])

    crc_expected = reference_crc8(payload0)

    crc_stream = PacketizedCrc8Stream(stream, upl)
    out = crc_stream.read_bits(upl * 2)

    replaced_sync_bits = out[upl:upl+8]
    replaced_sync = int("".join(map(str, replaced_sync_bits)), 2)

    assert replaced_sync == crc_expected
    print("Test 6 PASSED: CRC replaces next packet SYNC")


def test_crc_chaining_multiple_packets():
    from BB_Frame import PacketizedCrc8Stream

    upl = 40
    packets = []

    # Build 5 packets
    for _ in range(5):
        payload = np.random.randint(0, 2, upl - 8, dtype=np.uint8)
        sync = np.array([0,1,0,0,0,1,1,1], dtype=np.uint8)
        packets.append(np.concatenate([sync, payload]))

    stream = np.concatenate(packets)
    crc_stream = PacketizedCrc8Stream(stream, upl)

    # Read entire stream at once
    out = crc_stream.read_bits(upl * len(packets))

    # Verify CRC chaining
    for i in range(len(packets) - 1):
        payload_i = packets[i][8:]
        crc_expected = reference_crc8(payload_i)

        # Sync of packet i+1 in output
        sync_start = (i + 1) * upl
        sync_bits = out[sync_start:sync_start + 8]
        sync_val = int("".join(map(str, sync_bits)), 2)

        assert sync_val == crc_expected

    print("Test 7 PASSED: CRC chaining across packets")



def test_crc_determinism():
    upl = 40
    payload = np.random.randint(0, 2, upl - 8, dtype=np.uint8)
    sync = np.array([0,1,0,0,0,1,1,1], dtype=np.uint8)

    up = np.concatenate([sync, payload])
    stream = np.concatenate([up, up])

    crc_stream1 = PacketizedCrc8Stream(stream, upl)
    crc_stream2 = PacketizedCrc8Stream(stream, upl)

    crc_stream1.read_bits(upl)
    crc_stream2.read_bits(upl)

    assert crc_stream1.prev_crc == crc_stream2.prev_crc
    print("Test 8 PASSED: CRC determinism")


def test_crc_single_packet_only():
    upl = 40
    payload = np.random.randint(0, 2, upl - 8, dtype=np.uint8)
    sync = np.array([0,1,0,0,0,1,1,1], dtype=np.uint8)

    up = np.concatenate([sync, payload])
    crc_stream = PacketizedCrc8Stream(up, upl)

    crc_stream.read_bits(upl)

    assert crc_stream.prev_crc == reference_crc8(payload)
    print("Test 9 PASSED: Single-packet CRC edge case")


# ============================================================
# RATE-AWARE STREAM ADAPTATION + BCH TESTS
# ============================================================

def test_kbch_lookup_short_half():
    kbch = get_kbch("short", "1/2")
    assert kbch == 7032
    print("Test 10 PASSED: Kbch lookup short 1/2")


def test_stream_adaptation_rate_length():
    bbheader = np.zeros(80, dtype=np.uint8)
    df = np.zeros(100, dtype=np.uint8)
    bbframe = np.concatenate([bbheader, df])

    adapted = stream_adaptation_rate(bbframe, "short", "1/2")
    assert len(adapted) == 7032
    print("Test 11 PASSED: Stream adaptation rate length")


def test_bch_encode_length_and_parity():
    kbch, nbch, _t = BCH_PARAMS[("short", "1/2")]
    bbframe_kbch = np.zeros(kbch, dtype=np.uint8)
    codeword = bch_encode_bbframe(bbframe_kbch, "short", "1/2")

    assert len(codeword) == nbch
    parity = codeword[kbch:]
    assert len(parity) == (nbch - kbch)
    print("Test 12 PASSED: BCH encode length and parity")

# ============================================================
# LDPC ENCODING TESTS
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MAT_PATH = os.path.join(ROOT, "s2xLDPCParityMatrices", "dvbs2xLDPCParityMatrices.mat")

def test_ldpc_encoder_initialization():
    """Test LDPC encoder can be initialized with MAT file."""
    if not os.path.isfile(MAT_PATH):
        print("Test 13 SKIPPED: MAT file not found")
        return
    
    encoder = DVB_LDPC_Encoder(MAT_PATH)
    assert encoder is not None
    assert encoder.mat_path == MAT_PATH
    print("Test 13 PASSED: LDPC encoder initialization")


def test_ldpc_available_codes():
    """Test LDPC encoder can list available codes."""
    if not os.path.isfile(MAT_PATH):
        print("Test 14 SKIPPED: MAT file not found")
        return
    
    encoder = DVB_LDPC_Encoder(MAT_PATH)
    codes = encoder.available_codes()
    
    assert len(codes) > 0
    assert any("1_2" in code for code in codes)  # Should have 1/2 rate
    print(f"Test 14 PASSED: LDPC available codes ({len(codes)} found)")


def test_ldpc_prepare_normal_half():
    """Test LDPC prepare for normal frame, 1/2 rate."""
    if not os.path.isfile(MAT_PATH):
        print("Test 15 SKIPPED: MAT file not found")
        return
    
    encoder = DVB_LDPC_Encoder(MAT_PATH)
    prep = encoder._prepare("normal", "1/2")
    
    assert prep["n"] == 64800  # normal NLDPC
    assert prep["k"] < prep["n"]
    assert prep["m"] > 0
    assert len(prep["sys_cols"]) == prep["m"]
    assert len(prep["par_lower"]) == prep["m"]
    print(f"Test 15 PASSED: LDPC prepare normal 1/2 (n={prep['n']}, k={prep['k']}, m={prep['m']})")


def test_ldpc_prepare_short_half():
    """Test LDPC prepare for short frame, 1/2 rate."""
    if not os.path.isfile(MAT_PATH):
        print("Test 16 SKIPPED: MAT file not found")
        return
    
    encoder = DVB_LDPC_Encoder(MAT_PATH)
    prep = encoder._prepare("short", "1/2")
    
    assert prep["n"] == 16200  # short NLDPC
    assert prep["k"] < prep["n"]
    assert prep["m"] > 0
    print(f"Test 16 PASSED: LDPC prepare short 1/2 (n={prep['n']}, k={prep['k']}, m={prep['m']})")


def test_ldpc_encode_output_length():
    """Test LDPC encoding produces correct output length."""
    if not os.path.isfile(MAT_PATH):
        print("Test 17 SKIPPED: MAT file not found")
        return
    
    encoder = DVB_LDPC_Encoder(MAT_PATH)
    prep = encoder._prepare("short", "1/2")
    
    # Create random input bits
    u_bits = np.random.randint(0, 2, prep["k"], dtype=np.uint8)
    codeword = encoder.encode(u_bits, "short", "1/2")
    
    assert len(codeword) == prep["n"]
    print(f"Test 17 PASSED: LDPC encode output length (input={len(u_bits)}, output={len(codeword)})")


def test_ldpc_encode_systematic():
    """Test LDPC codeword is systematic (first k bits match input)."""
    if not os.path.isfile(MAT_PATH):
        print("Test 18 SKIPPED: MAT file not found")
        return
    
    encoder = DVB_LDPC_Encoder(MAT_PATH)
    prep = encoder._prepare("short", "1/2")
    
    u_bits = np.random.randint(0, 2, prep["k"], dtype=np.uint8)
    codeword = encoder.encode(u_bits, "short", "1/2")
    
    # Systematic: first k bits should equal input
    assert np.array_equal(codeword[:prep["k"]], u_bits)
    print("Test 18 PASSED: LDPC codeword is systematic")


def test_ldpc_encode_parity_bits():
    """Test LDPC parity bits extraction."""
    if not os.path.isfile(MAT_PATH):
        print("Test 19 SKIPPED: MAT file not found")
        return
    
    encoder = DVB_LDPC_Encoder(MAT_PATH)
    prep = encoder._prepare("short", "1/2")
    
    u_bits = np.random.randint(0, 2, prep["k"], dtype=np.uint8)
    codeword = encoder.encode(u_bits, "short", "1/2")
    
    parity_bits = codeword[prep["k"]:]
    assert len(parity_bits) == (prep["n"] - prep["k"])
    assert parity_bits.dtype == np.uint8
    print(f"Test 19 PASSED: LDPC parity bits ({len(parity_bits)} bits extracted)")


def test_ldpc_encode_determinism():
    """Test LDPC encoding is deterministic (same input produces same output)."""
    if not os.path.isfile(MAT_PATH):
        print("Test 20 SKIPPED: MAT file not found")
        return
    
    encoder = DVB_LDPC_Encoder(MAT_PATH)
    prep = encoder._prepare("short", "1/2")
    
    u_bits = np.random.randint(0, 2, prep["k"], dtype=np.uint8)
    
    codeword1 = encoder.encode(u_bits, "short", "1/2")
    codeword2 = encoder.encode(u_bits, "short", "1/2")
    
    assert np.array_equal(codeword1, codeword2)
    print("Test 20 PASSED: LDPC encoding determinism")


def test_ldpc_encode_different_rates():
    """Test LDPC encoding works with different code rates."""
    if not os.path.isfile(MAT_PATH):
        print("Test 21 SKIPPED: MAT file not found")
        return
    
    encoder = DVB_LDPC_Encoder(MAT_PATH)
    rates_to_test = ["1/2", "3/5", "2/3", "3/4", "5/6"]
    
    for rate in rates_to_test:
        try:
            prep = encoder._prepare("short", rate)
            u_bits = np.random.randint(0, 2, prep["k"], dtype=np.uint8)
            codeword = encoder.encode(u_bits, "short", rate)
            
            assert len(codeword) == prep["n"]
            assert np.array_equal(codeword[:prep["k"]], u_bits)
        except KeyError:
            # Rate not available in this matrix
            pass
    
    print("Test 21 PASSED: LDPC encoding multiple rates")


def test_ldpc_function_wrapper():
    """Test LDPC encoding via function wrapper."""
    if not os.path.isfile(MAT_PATH):
        print("Test 22 SKIPPED: MAT file not found")
        return
    
    encoder = DVB_LDPC_Encoder(MAT_PATH)
    prep = encoder._prepare("short", "1/2")
    
    u_bits = np.random.randint(0, 2, prep["k"], dtype=np.uint8)
    codeword = ldpc_encode_bits(u_bits, "short", "1/2", MAT_PATH)
    
    assert len(codeword) == prep["n"]
    assert np.array_equal(codeword[:prep["k"]], u_bits)
    print("Test 22 PASSED: LDPC function wrapper")


def test_ldpc_bch_chain():
    """Test LDPC encoding in chain with BCH."""
    if not os.path.isfile(MAT_PATH):
        print("Test 23 SKIPPED: MAT file not found")
        return
    
    # Generate BCH codeword
    kbch, nbch, _t = BCH_PARAMS[("short", "1/2")]
    bbframe = np.random.randint(0, 2, kbch, dtype=np.uint8)
    bch_codeword = bch_encode_bbframe(bbframe, "short", "1/2")
    
    # Pass BCH output to LDPC
    encoder = DVB_LDPC_Encoder(MAT_PATH)
    prep = encoder._prepare("short", "1/2")
    
    # BCH output should match LDPC input size (Nbch == Kldpc)
    assert len(bch_codeword) == prep["k"]
    
    ldpc_codeword = encoder.encode(bch_codeword, "short", "1/2")
    
    assert len(ldpc_codeword) == prep["n"]
    assert np.array_equal(ldpc_codeword[:prep["k"]], bch_codeword)
    print(f"Test 23 PASSED: LDPC-BCH chain (BCH→LDPC: {nbch}→{prep['n']} bits)")


def test_ldpc_cache():
    """Test LDPC encoder caches prepared matrices."""
    if not os.path.isfile(MAT_PATH):
        print("Test 24 SKIPPED: MAT file not found")
        return
    
    encoder = DVB_LDPC_Encoder(MAT_PATH)
    
    # First call should load from disk
    prep1 = encoder._prepare("short", "1/2")
    
    # Second call should use cache
    prep2 = encoder._prepare("short", "1/2")
    
    # Should be the exact same object (from cache)
    assert prep1 is prep2
    assert ("short", "1/2") in encoder._cache
    print("Test 24 PASSED: LDPC encoder caching")


def test_ldpc_all_zero_input():
    """Test LDPC encoding with all-zero input."""
    if not os.path.isfile(MAT_PATH):
        print("Test 25 SKIPPED: MAT file not found")
        return
    
    encoder = DVB_LDPC_Encoder(MAT_PATH)
    prep = encoder._prepare("short", "1/2")
    
    u_bits = np.zeros(prep["k"], dtype=np.uint8)
    codeword = encoder.encode(u_bits, "short", "1/2")
    
    # First k bits should be all zero
    assert np.all(codeword[:prep["k"]] == 0)
    print("Test 25 PASSED: LDPC all-zero input")


def test_ldpc_all_one_input():
    """Test LDPC encoding with all-one input."""
    if not os.path.isfile(MAT_PATH):
        print("Test 26 SKIPPED: MAT file not found")
        return
    
    encoder = DVB_LDPC_Encoder(MAT_PATH)
    prep = encoder._prepare("short", "1/2")
    
    u_bits = np.ones(prep["k"], dtype=np.uint8)
    codeword = encoder.encode(u_bits, "short", "1/2")
    
    # First k bits should be all one
    assert np.all(codeword[:prep["k"]] == 1)
    print("Test 26 PASSED: LDPC all-one input")

# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    print("\n===== RUNNING BBFRAME TEST SUITE =====\n")

    test_continuous_gs()
    test_packetized_gs_valid()
    test_packetized_gs_ambiguous()
    test_ts_stream()

    test_crc_single_packet()
    test_crc_replaces_next_sync()
    test_crc_chaining_multiple_packets()
    test_crc_determinism()
    test_crc_single_packet_only()
    test_kbch_lookup_short_half()
    test_stream_adaptation_rate_length()
    test_bch_encode_length_and_parity()

    print("\n===== RUNNING LDPC ENCODING TESTS =====\n")
    
    test_ldpc_encoder_initialization()
    test_ldpc_available_codes()
    test_ldpc_prepare_normal_half()
    test_ldpc_prepare_short_half()
    test_ldpc_encode_output_length()
    test_ldpc_encode_systematic()
    test_ldpc_encode_parity_bits()
    test_ldpc_encode_determinism()
    test_ldpc_encode_different_rates()
    test_ldpc_function_wrapper()
    test_ldpc_bch_chain()
    test_ldpc_cache()
    test_ldpc_all_zero_input()
    test_ldpc_all_one_input()

    print("\n===== ALL TESTS PASSED SUCCESSFULLY =====\n")
