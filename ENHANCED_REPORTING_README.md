# Enhanced DVB-S2x Loopback Test with Detailed Reporting

## Summary

Your DVB-S2x loopback test now includes **comprehensive stage-by-stage reporting** that captures signals at every step of the transmitter and receiver pipeline. This enables detailed analysis, debugging, and validation of the entire signal processing chain.

## What's New

### 🎯 Key Features

1. **19-Stage Intermediate Output Capture**
   - All signals saved to individual files
   - Multi-format support (bits, complex symbols, LLRs)
   - Per-frame intermediate files organized in `results/loopback/intermediate/`

2. **Enhanced Console Reporting**
   - Emoji-formatted output for clarity
   - Hierarchical stage information
   - Signal dimensions at each stage
   - Power and phase measurements
   - LDPC iteration counts

3. **Automated Analysis Tool**
   - `analyze_intermediates.py` for signal analysis
   - Bit statistics, constellation quality, LLR distribution
   - Per-frame detailed reports

4. **Improved Channel Simulation**
   - Measured vs target SNR reporting
   - Noise power calculations
   - Configurable AWGN channel

## Quick Start

### Basic Test
```bash
cd /home/umair/Videos/DVBS2x/git
conda activate dvbs2
python tests/test_tx_rx_loopback.py
```

### Test with Noise
```bash
python tests/test_tx_rx_loopback.py --max-frames 3 --esn0-db 5
```

### Analyze Intermediate Signals
```bash
python tests/analyze_intermediates.py --frame 1
```

## Output Structure

```
results/loopback/
├── loopback_stats.json                    # Overall statistics
├── tx_bits_frame1.txt                     # Input bits
├── rx_bits_frame1.txt                     # Decoded output bits
└── intermediate/
    ├── frame1_01_tx_input_bits.txt        # TX stage 1
    ├── frame1_02_bbframe.txt              # TX stage 2
    ├── frame1_03_adapted.txt              # TX stage 3
    ...
    ├── frame1_12_scrambled.txt            # TX stage 12 (channel input)
    ├── frame1_12_descrambled.txt          # RX stage 1
    ├── frame1_13_payload_raw.txt          # RX stage 2
    ...
    └── frame1_19_df_bits.txt              # RX stage 8 (final output)
```

## Console Output Example

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
  Es/N0 (measured)     : 5.00 dB
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
LDPC Iterations        : 4
```

## TX Processing Pipeline

| Stage | Input | Output | Processing |
|-------|-------|--------|------------|
| 1 | Bits | 720 bits | Original data |
| 2 | Bits | 800 bits | BB frame header + data |
| 3 | Bits | 7032 bits | Stream adaptation, scrambling |
| 4 | Bits | 7200 bits | BCH encoding (t=12) |
| 5 | Bits | 16200 bits | LDPC encoding |
| 6 | Bits | 16200 bits | DVB-S2 bit interleaving |
| 7 | Bits | 8100 symbols | QPSK constellation mapping |
| 8 | Symbols | 8280 symbols | Pilot insertion |
| 9 | Symbols | 8370 symbols | PL header structure |
| 10 | Symbols | 8370 symbols | PL scrambling |

## RX Processing Pipeline

| Stage | Input | Output | Processing |
|-------|-------|--------|------------|
| 1 | Symbols | 8370 symbols | PL descrambling |
| 2 | Symbols | 8100 symbols | Pilot removal |
| 3 | Symbols | 8100 symbols | Pilot-aided phase correction |
| 4 | Symbols | 16200 LLRs | Soft constellation demapping |
| 5 | LLRs | 16200 LLRs | Bit deinterleaving |
| 6 | LLRs | 16200 bits | LDPC decoding (normalized min-sum) |
| 7 | Bits | 7032 bits | BCH decoding |
| 8 | Bits | 1000 bits | Stream deadaptation |

## Signal Analysis

### Analyze Frame Signals
```bash
python tests/analyze_intermediates.py --frame 1
```

Output includes:
- **Bit Statistics**: Count, zeros, ones, ratio
- **Complex Signals**: Magnitude, power, phase
- **LLRs**: Mean, std, confidence levels
- **Bit Comparison**: TX vs final RX BER

### Manual Analysis Example

```python
import numpy as np

# Load constellation symbols
symbols = np.loadtxt('results/loopback/intermediate/frame1_07_constellation_symbols.txt',
                     dtype=complex)

# Check power
power = np.mean(np.abs(symbols)**2)
print(f"Symbol power: {power:.4f}")  # Should be ~1.0

# Load LLRs
llrs = np.loadtxt('results/loopback/intermediate/frame1_15_llrs_interleaved.txt')
print(f"LLR mean: {np.mean(llrs):.3f}")
print(f"LLR std: {np.std(llrs):.3f}")
```

## Performance Testing

### SNR Performance Sweep
```bash
for SNR in 0 2 4 6 8 10; do
    echo "Testing SNR=$SNR dB"
    python tests/test_tx_rx_loopback.py --max-frames 2 --esn0-db $SNR
done
```

### Different Modulation/Rates
```bash
# 8PSK, rate 3/4
python tests/test_tx_rx_loopback.py --modulation 8PSK --rate 3/4 --max-frames 5

# 16APSK, rate 5/6
python tests/test_tx_rx_loopback.py --modulation 16APSK --rate 5/6 --max-frames 5
```

### Fast Mode (No Intermediate Files)
```bash
python tests/test_tx_rx_loopback.py --max-frames 100 --no-detailed
```

## File Formats

### Bit Files (*.txt)
Binary sequences, one continuous line:
```
01000111010101010100110101000001...
```

### Complex Symbol Files (*.txt)
Real and imaginary components, one pair per line:
```
+0.70710678 +0.70710678
+0.70710678 -0.70710678
-0.70710678 +0.70710678
```

### LLR Files (*.txt)
Log-likelihood ratios, one per line:
```
+5.27924949
+7.84915429
-8.49528728
```

### Statistics JSON (loopback_stats.json)
```json
{
    "timestamp": "2026-02-04T12:08:22",
    "config": {
        "fecframe": "short",
        "rate": "1/2",
        "modulation": "QPSK",
        "esn0_db": 5.0,
        "max_frames": 3
    },
    "frames": [...],
    "summary": {
        "total_frames": 3,
        "successful_frames": 3,
        "frame_success_rate": 1.0,
        "overall_ber": 0.0
    }
}
```

## All Command-Line Options

```bash
python tests/test_tx_rx_loopback.py --help

Options:
  --fecframe {short,normal}     FEC frame type (default: short)
  --rate {1/2,3/4,5/6,...}      Code rate (default: 1/2)
  --modulation {QPSK,8PSK,...}  Modulation (default: QPSK)
  --esn0-db SNR                 Es/N0 in dB (default: 0)
  --max-frames N                Number of frames (default: 1)
  --output-dir PATH             Output directory (default: results/loopback/)
  --no-detailed                 Skip detailed reporting (faster)
  --pilots-on/--pilots-off      Enable/disable pilots (default: on)
  --scrambling-code CODE        PL scrambling code (default: 0)
```

## Documentation

- [DETAILED_REPORT_GUIDE.md](./DETAILED_REPORT_GUIDE.md) - Comprehensive guide with examples
- [STRUCTURE.md](./STRUCTURE.md) - Project architecture
- [PROJECT_TREE.txt](./PROJECT_TREE.txt) - File organization

## Troubleshooting

**Issue**: Missing intermediate files
- **Cause**: `--no-detailed` flag enabled
- **Solution**: Remove the flag or ensure it's not in your command

**Issue**: Memory issues with many frames
- **Solution**: Use `--no-detailed` to disable intermediate file generation

**Issue**: Slow performance
- **Solution**: Use `--no-detailed` flag to skip file I/O

**Issue**: Performance unexpectedly poor
- **Solutions**:
  1. Check phase correction stage in intermediate outputs
  2. Verify constellation symbol power is ~1.0
  3. Analyze LLR distribution for confidence levels

## Next Steps

1. **Characterize Performance**
   - Run SNR sweep (0-12 dB) to measure error floor
   - Generate BER vs SNR curves
   - Compare against DVB-S2 theoretical limits

2. **Test All Modes**
   - Verify different modulations work correctly
   - Test various code rates
   - Validate all mode combinations

3. **Advanced Analysis**
   - Create constellation plots
   - Visualize LLR distributions
   - Track phase error evolution
   - Generate performance reports

4. **Signal Processing Validation**
   - Verify bit interleaver matches DVB-S2 standard
   - Validate LDPC convergence characteristics
   - Check pilot insertion/removal accuracy

## Contact & Support

For detailed technical information, see:
- Standard: ETSI EN 302 307 (DVB-S2)
- Intermediate files: `results/loopback/intermediate/`
- Statistics: `results/loopback/loopback_stats.json`
- Analysis tool: `tests/analyze_intermediates.py`
