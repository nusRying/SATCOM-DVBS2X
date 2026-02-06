# DVB-S2x Loopback Test - Detailed Report Guide

## Overview

The enhanced `test_tx_rx_loopback.py` script provides comprehensive stage-by-stage reporting for the entire DVB-S2x transmission-reception pipeline. Each frame generates 19 intermediate output files for detailed analysis and debugging.

## Running the Test

### Basic Usage
```bash
cd /home/umair/Videos/DVBS2x/git
conda activate dvbs2
python tests/test_tx_rx_loopback.py
```

### Common Options
```bash
# Test with multiple frames
python tests/test_tx_rx_loopback.py --max-frames 5

# Test at specific SNR
python tests/test_tx_rx_loopback.py --esn0-db 3.0 --max-frames 10

# Test with different modulation/rate
python tests/test_tx_rx_loopback.py --modulation 8PSK --rate 3/4 --max-frames 5

# Disable detailed reporting (faster, minimal output)
python tests/test_tx_rx_loopback.py --no-detailed --max-frames 100

# Change output directory
python tests/test_tx_rx_loopback.py --output-dir /tmp/dvbs2_results
```

## Output Structure

### Console Output Format

Each frame generates formatted console output showing all processing stages:

```
======================================================================
FRAME 1
======================================================================

📤 TRANSMITTER CHAIN (12 stages)
----------------------------------------------------------------------
01. Input Bits         : 720 bits
02. BB Header          : 80 bits
03. BB Frame           : 800 bits (header 80 + data 720)
04. After Scrambling   : 7032 bits
05. BCH Encoded        : 7200 bits
06. LDPC Encoded       : 16200 bits
07. Bit Interleaved    : 16200 bits
08. Constellation Map  : 8100 symbols (QPSK)
    Power (avg)        : 1.0000
09. PL Header          : 90 symbols
10. Payload+Pilots     : 8280 symbols
11. Full PLFRAME       : 8370 symbols (header 90 + payload 8280)
12. PL Scrambled       : 8370 symbols

📡 CHANNEL
----------------------------------------------------------------------
AWGN Channel
  Es/N0 (target)       : 5.0 dB
  Es/N0 (measured)     : 5.03 dB
  Signal Power         : 1.0000
  Noise Power          : 0.316228

📥 RECEIVER CHAIN (8 stages)
----------------------------------------------------------------------
01. Descrambled        : 8370 symbols
02. Pilot Removed      : 8100 symbols
03. Phase Corrected    : 8100 symbols
04. Demapped (LLRs)    : 16200 values
05. Deinterleaved      : 16200 values
06. LDPC Decoded       : 16200 bits
07. BCH Decoded        : 7032 bits
08. Final DF Bits      : 1000 bits

📊 PERFORMANCE ANALYSIS
----------------------------------------------------------------------
Bits Compared          : 720
Bit Errors             : 0/720
BER                    : 0.000000e+00
Frame Success          : ✅ YES
LDPC Iterations        : 3
```

### File System Structure

```
results/
├── loopback/
│   ├── loopback_stats.json          # Summary statistics
│   ├── rx_bits_frame1.txt           # Decoded bits per frame
│   ├── tx_bits_frame1.txt           # Input bits per frame
│   └── intermediate/                # Detailed stage outputs
│       ├── frame1_01_tx_input_bits.txt
│       ├── frame1_03_bbframe.txt
│       ├── frame1_04_adapted.txt
│       ├── frame1_05_bch_codeword.txt
│       ├── frame1_06_ldpc_codeword.txt
│       ├── frame1_07_interleaved.txt
│       ├── frame1_08_constellation_symbols.txt
│       ├── frame1_10_with_pilots.txt
│       ├── frame1_11_plframe_tx.txt
│       ├── frame1_12_plframe_scrambled.txt
│       ├── frame1_13_plframe_rx.txt
│       ├── frame1_01_descrambled.txt
│       ├── frame1_02_payload_raw.txt
│       ├── frame1_03_payload_corrected.txt
│       ├── frame1_04_llrs_interleaved.txt
│       ├── frame1_05_llrs_deinterleaved.txt
│       ├── frame1_06_ldpc_decoded.txt
│       ├── frame1_07_bch_decoded.txt
│       └── frame1_08_df_bits.txt
│       # RX numbering restarts at 01 because it is local to the RX chain
│       └── ... (same for frames 2, 3, ...)
```

## Signal Processing Stages

### Transmitter Chain (12 Stages)

| Stage | Name | Format | Description |
|-------|------|--------|-------------|
| 01 | TX Input Bits | Binary | Original data bits (720 bits) |
| 02 | BB Header | Binary | 80-bit DVB-S2 baseband header |
| 03 | BB Frame | Binary | 800 bits (header + data) |
| 04 | After Scrambling | Binary | 7032 bits after stream adaptation |
| 05 | BCH Codeword | Binary | 7200 bits (BCH t=12) |
| 06 | LDPC Codeword | Binary | 16200 bits (LDPC encoded) |
| 07 | Bit Interleaved | Binary | 16200 bits (DVB-S2 standard interleaver) |
| 08 | Constellation Map | Complex | 8100 QPSK symbols (power ~1.0) |
| 09 | PL Header | Complex | 90 symbols |
| 10 | Payload+Pilots | Complex | 8280 symbols |
| 11 | Full PLFRAME TX | Complex | 8370 symbols |
| 12 | PL Scrambled | Complex | 8370 symbols (after PL scrambling) |

### Channel (Not a stage, but important)

- **Type**: AWGN (Additive White Gaussian Noise)
- **Parameters**: Configurable Es/N0 (0-20 dB typical range)
- **Output**: Complex received signal

### Receiver Chain (8 Stages)

| Stage | Name | Format | Description |
|-------|------|--------|-------------|
| 01 | Descrambled | Complex | After PL descrambling |
| 02 | Pilot Removed | Complex | 8100 symbols (pilots extracted) |
| 03 | Phase Corrected | Complex | Pilot-aided phase correction |
| 04 | Demapped (LLRs) | Real | 16200 LLR values (soft output) |
| 05 | Deinterleaved | Real | 16200 values (inverse bit interleaver) |
| 06 | LDPC Decoded | Binary | 16200 bits (LDPC decoder output) |
| 07 | BCH Decoded | Binary | 7032 bits (BCH error correction) |
| 08 | Final DF Bits | Binary | 1000 bits (user data) |

## Intermediate File Formats

### Binary Files (.txt)
- **Bits**: Single column of 0s and 1s
- **Example**: frame1_01_tx_input_bits.txt
```
0
1
0
0
0
1
...
```

### Complex Symbol Files (.txt)
- **Real and imaginary components**: Space-separated on same line
- **Example**: frame1_07_constellation_symbols.txt
```
+0.70710678 +0.70710678
+0.70710678 -0.70710678
-0.70710678 +0.70710678
...
```

### Real-Valued Files (.txt) - LLRs
- **Log-likelihood ratios**: Single column of floating-point values
- **Example**: frame1_04_llrs_interleaved.txt
```
+5.27924949
+7.84915429
-8.49528728
...
```

## Analysis Workflow

### 1. Verify Signal Dimensions

Check that dimensions match expected values at each stage:

```bash
# Count bits in input
wc -l results/loopback/intermediate/frame1_01_tx_input_bits.txt  # Should be 720

# Count symbols in constellation
wc -l results/loopback/intermediate/frame1_07_constellation_symbols.txt  # Should be 8100

# Count LLRs
wc -l results/loopback/intermediate/frame1_04_llrs_interleaved.txt  # Should be 16200
```

### 2. Analyze Constellation Quality

```python
import numpy as np

# Load QPSK symbols
symbols = np.loadtxt('results/loopback/intermediate/frame1_07_constellation_symbols.txt',
                     dtype=complex)

# Calculate average power
power = np.mean(np.abs(symbols)**2)
print(f"Average Power: {power:.4f}")  # Should be ~1.0

# Plot constellation
import matplotlib.pyplot as plt
plt.scatter(symbols.real, symbols.imag, alpha=0.5)
plt.axis('equal')
plt.grid(True)
plt.xlabel('In-phase')
plt.ylabel('Quadrature')
plt.title('QPSK Constellation')
plt.savefig('qpsk_constellation.png')
```

### 3. Analyze LLR Distribution

```python
import numpy as np
import matplotlib.pyplot as plt

# Load LLRs before deinterleaving
llrs = np.loadtxt('results/loopback/intermediate/frame1_04_llrs_interleaved.txt')

# Statistics
print(f"LLR Mean: {np.mean(llrs):.3f}")
print(f"LLR Std: {np.std(llrs):.3f}")
print(f"LLR Min: {np.min(llrs):.3f}")
print(f"LLR Max: {np.max(llrs):.3f}")

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(llrs, bins=50, alpha=0.7)
plt.xlabel('LLR Value')
plt.ylabel('Frequency')
plt.title(f'LLR Distribution (SNR: N/A)')
plt.grid(True)
plt.savefig('llr_distribution.png')

# Higher LLR magnitude = higher confidence in the bit decision
confident_bits = np.sum(np.abs(llrs) > 5.0)
print(f"High-confidence bits (|LLR| > 5): {confident_bits} / {len(llrs)}")
```

### 4. Track Errors Through Pipeline

```python
import numpy as np

# Load original bits
tx_bits = np.loadtxt('results/loopback/intermediate/frame1_01_tx_input_bits.txt', dtype=int)

# Load decoded bits before BCH (LDPC output)
ldpc_bits = np.loadtxt('results/loopback/intermediate/frame1_06_ldpc_decoded.txt', dtype=int)

# Load final bits (after BCH)
final_bits = np.loadtxt('results/loopback/intermediate/frame1_08_df_bits.txt', dtype=int)

# Check for errors at each stage
print("Error Tracking:")
print(f"TX bits length: {len(tx_bits)}")
print(f"LDPC output length: {len(ldpc_bits)}")
print(f"Final output length: {len(final_bits)}")
print(f"Final BER: {np.sum(tx_bits != final_bits[:len(tx_bits)]) / len(tx_bits):.2e}")
```

## JSON Statistics Format

The `loopback_stats.json` file contains aggregated results:

```json
{
    "timestamp": "2026-02-04T12:08:22.934536",
    "config": {
        "fecframe": "short",
        "rate": "1/2",
        "modulation": "QPSK",
        "pilots_on": true,
        "scrambling_code": 0,
        "esn0_db": 5.0,
        "max_frames": 3,
        "detailed_report": true
    },
    "frames": [
        {
            "frame_num": 1,
            "tx_bits_shape": [720],
            "intermediate_saved": true,
            "rx_bits_shape": [1000],
            "bits_compared": 720,
            "bit_errors": 0,
            "ber": 0.0,
            "frame_success": true
        },
        ...
    ],
    "summary": {
        "total_frames": 3,
        "successful_frames": 3,
        "frame_success_rate": 1.0,
        "total_bit_errors": 0,
        "total_bits_tested": 2160,
        "overall_ber": 0.0
    }
}
```

## Performance Testing

### Quick SNR Sweep

```bash
cd /home/umair/Videos/DVBS2x/git
conda activate dvbs2

# Test across SNR range
for SNR in 0 2 4 6 8; do
    echo "Testing SNR=${SNR} dB..."
    python tests/test_tx_rx_loopback.py --max-frames 2 --esn0-db $SNR 2>&1 | tail -20
    echo ""
done
```

### Comprehensive Analysis

```bash
# Run test with full output capture
python tests/test_tx_rx_loopback.py --max-frames 5 --esn0-db 3 2>&1 | tee loopback_test.log

# Extract key metrics
grep -E "BER|Frame Success|LDPC Iterations" loopback_test.log
```

## Troubleshooting

### Issue: Missing intermediate files
**Solution**: Ensure `--no-detailed` flag is NOT used. Default is detailed reporting enabled.

### Issue: Memory issues with many frames
**Solution**: Reduce `--max-frames` or disable `--no-detailed` to save memory.

### Issue: File format errors when loading
**Solution**: Verify file format:
- Bits: One value per line (0 or 1)
- Complex: Real and imaginary on same line, space-separated
- LLRs: One floating-point value per line

### Issue: Performance unexpectedly poor
**Diagnostic**:
1. Check phase correction stage for phase estimates
2. Verify constellation symbols have appropriate power (~1.0)
3. Check LLR distribution for signs (should be majority positive for correct bits)

## Next Steps

1. **Multi-SNR Analysis**: Run sweep across 0-12 dB range to characterize error floor
2. **Different Modes**: Test with 8PSK, 16APSK, 32APSK modulations and different code rates
3. **Visualization**: Create plots of intermediate signals for debugging
4. **Performance Optimization**: Identify bottlenecks in processing pipeline

## References

- **Standard**: ETSI EN 302 307 V1.3.1 (DVB-S2 specification)
- **LDPC**: Normalized min-sum decoder with configurable iterations
- **BCH**: t=12 error correction capability
- **Modulation**: QPSK/8PSK/16APSK/32APSK with gray mapping
