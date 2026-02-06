"""
TX→RX Loopback Test

This script:
1. Runs the transmitter (tx/run_dvbs2.py) to generate a full PLFRAME with pilots
2. Optionally adds AWGN noise
3. Feeds the output directly into the receiver chain (rx/receiver_Chain.py)
4. Compares TX input bits with RX decoded output bits
5. Generates performance metrics (BER, packet success rate)
"""

import os
import sys
import numpy as np
import json
from datetime import datetime
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# TX imports
from tx._01_BB_Frame import build_bbheader, load_bits_csv, resolve_input_path
from tx._02_stream_adaptation import stream_adaptation_rate, get_kbch
from tx._03_bch_encoding import bch_encode_bbframe
from tx._04_ldpc_Encoding import DVB_LDPC_Encoder
from common.bit_interleaver import dvbs2_bit_interleave
from common.constellation_mapper import dvbs2_constellation_map
from common.pilot_insertion import insert_pilots_into_payload
from tx._05_pl_header import build_plheader, modcod_from_modulation_rate
from common.pl_scrambler import pl_scramble_full_plframe

# RX imports
from rx.receiver_Chain import process_rx_plframe


def write_bits_single_line(path: str, bits: np.ndarray):
    """Write bits (0/1) as single line separated by spaces."""
    bits_str = " ".join(str(int(b)) for b in bits.flatten())
    with open(path, "w") as f:
        f.write(bits_str + "\n")


def plot_constellation(symbols: np.ndarray, title: str, output_path: str):
    """Plot constellation diagram."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    try:
        if symbols is None or len(symbols) == 0:
            print(f"Warning: No symbols to plot for {title}")
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(symbols.real, symbols.imag, alpha=0.7, s=30, facecolors='C0', edgecolors='k')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('In-phase (I)', fontsize=11)
        ax.set_ylabel('Quadrature (Q)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Set symmetric axis limits around zero for clarity
        md = np.max(np.abs(symbols)) if len(symbols) > 0 else 1.0
        if md == 0:
            md = 1.0
        lim = md * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        # Add power annotation
        power = np.mean(np.abs(symbols)**2)
        ax.text(0.02, 0.98, f'Power: {power:.4f}', transform=ax.transAxes,
                verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not plot constellation: {e}")


def save_intermediate(data, name: str, output_dir: str, frame_num: int, format_type: str = "text"):
    """Save intermediate signal/data at each processing stage."""
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"frame{frame_num}_{name}")
    
    if format_type == "bits":
        bits_str = "".join(str(int(b)) for b in data.flatten())
        with open(fname + ".txt", "w") as f:
            f.write(bits_str)
    elif format_type == "complex":
        # Complex numbers (symbols, signals)
        with open(fname + ".txt", "w") as f:
            for val in data.flatten():
                f.write(f"{val.real:+.8f} {val.imag:+.8f}\n")
    elif format_type == "real":
        # Real numbers (LLRs, etc.)
        with open(fname + ".txt", "w") as f:
            for val in data.flatten():
                f.write(f"{val:+.8f}\n")
    elif format_type == "json":
        with open(fname + ".json", "w") as f:
            json.dump(data, f, indent=2)


def run_tx_rx_loopback(
    fecframe: str = "short",
    rate: str = "1/2",
    modulation: str = "QPSK",
    pilots_on: bool = True,
    scrambling_code: int = 0,
    esn0_db: float | None = None,
    max_frames: int = 1,
    output_dir: str = "loopback_output",
    detailed_report: bool = True,
    dfl: int | None = None,
    upl: int | None = None,
    frameization: str = "repeat",  # repeat | chunk
    pad_policy: str = "pad",  # pad | drop
    interactive: bool = False,
) -> dict:
    """
    Run a complete TX→RX loopback test.
    
    Args:
        fecframe: "normal" or "short"
        rate: Code rate (e.g., "1/2", "3/5", "2/3", "3/4", "5/6", "8/9", "9/10")
        modulation: "QPSK", "8PSK", "16APSK", "32APSK"
        pilots_on: Enable pilot insertion
        scrambling_code: PL scrambling code (0..262142)
        esn0_db: SNR in dB (None for noiseless)
        max_frames: Number of frames to process
        output_dir: Output directory for results
        
    Returns:
        Dictionary with statistics
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get parameters
    Kbch = get_kbch(fecframe, rate)
    # Default DFL/UPL
    DFL_default = min(1000, Kbch - 80)  # Use reasonable data field length
    DFL = DFL_default if dfl is None else int(dfl)
    UPL = 120 if upl is None else int(upl)
    SYNC_BYTE = 0x47

    # Interactive prompts similar to tx/run_dvbs2.py
    if interactive:
        try:
            user_dfl = input(f"Enter DFL (default {DFL}): ").strip()
            if user_dfl:
                DFL = int(user_dfl)
            user_upl = input(f"Enter UPL (default {UPL}): ").strip()
            if user_upl:
                UPL = int(user_upl)
            user_frameization = input(f"Frameization (repeat/chunk) (default {frameization}): ").strip().lower()
            if user_frameization in {"repeat", "chunk"}:
                frameization = user_frameization
            user_pad = input(f"Pad policy for last chunk (pad/drop) (default {pad_policy}): ").strip().lower()
            if user_pad in {"pad", "drop"}:
                pad_policy = user_pad
        except Exception as e:
            print(f"Input error: {e}. Using defaults.")
    
    # Get CSV path and load input bits
    csv_path = resolve_input_path(os.path.join(ROOT, "data", "GS_data", "umair_gs_bits.csv"))
    in_bits = load_bits_csv(csv_path)

    # Prepare frames based on frameization mode
    frames_bits = []
    if frameization == "chunk":
        if DFL <= 0:
            frames_bits = []
        else:
            total = len(in_bits)
            num_full = total // DFL
            for i in range(num_full):
                frames_bits.append(in_bits[i * DFL:(i + 1) * DFL])
            rem = total - num_full * DFL
            if rem > 0:
                if pad_policy == "pad":
                    last = in_bits[num_full * DFL:]
                    padlen = DFL - len(last)
                    last_padded = np.concatenate([last, np.zeros(padlen, dtype=np.uint8)])
                    frames_bits.append(last_padded)
                else:  # drop
                    pass
    else:
        # repeat mode: use the same initial DFL-sized slice for all frames
        base = in_bits[:DFL] if DFL > 0 else np.array([], dtype=np.uint8)
        frames_bits = [base] * max_frames

    # If chunk mode, limit the number of frames to max_frames
    if frameization == "chunk":
        if max_frames is None or max_frames <= 0:
            max_frames = len(frames_bits)
        else:
            max_frames = min(max_frames, len(frames_bits))
    
    # Get LDPC encoder
    mat_path = os.path.join(ROOT, "config", "ldpc_matrices", "dvbs2xLDPCParityMatrices.mat")
    ldpc_encoder = DVB_LDPC_Encoder(mat_path)
    
    # Statistics
    stats = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "fecframe": fecframe,
            "rate": rate,
            "modulation": modulation,
            "pilots_on": pilots_on,
            "scrambling_code": scrambling_code,
            "esn0_db": esn0_db,
            "max_frames": max_frames,
            "detailed_report": detailed_report,
        },
        "frames": []
    }
    
    # Create intermediate output directory if detailed reporting enabled
    intermediate_dir = os.path.join(output_dir, "intermediate") if detailed_report else None
    
    # Process each frame
    for frame_num in range(1, max_frames + 1):
        print(f"\n{'='*70}")
        print(f"FRAME {frame_num}")
        print(f"{'='*70}")
        
        # =====================================================================
        # TX SIDE: Generate PLFRAME
        # =====================================================================
        print("\nTRANSMITTER CHAIN")
        print("-" * 70)
        
        # Get TX input bits (for later comparison)
        # Use prepared frames_bits list (chunk or repeat behavior)
        if len(frames_bits) >= frame_num:
            tx_input_bits = frames_bits[frame_num - 1]
        else:
            # Fallback to default slice
            tx_input_bits = in_bits[:DFL] if DFL > 0 else np.array([], dtype=np.uint8)
        
        if detailed_report:
            # TX chain numbering follows printed stage order
            save_intermediate(tx_input_bits, "01_tx_input_bits", intermediate_dir, frame_num, "bits")
        
        print(f"01. Input Bits         : {len(tx_input_bits)} bits")
        
        # Build BB header
        bbheader_bits = build_bbheader(
            matype1=0x00,
            matype2=0x00,
            upl=UPL,
            dfl=DFL,
            sync=SYNC_BYTE,
            syncd=0
        )
        print(f"02. BB Header          : {len(bbheader_bits)} bits")
        
        # Build BB frame (with padding)
        if len(tx_input_bits) > 0:
            bbframe = np.concatenate([bbheader_bits, tx_input_bits])
        else:
            bbframe = bbheader_bits
        
        if detailed_report:
            save_intermediate(bbframe, "03_bbframe", intermediate_dir, frame_num, "bits")
        
        print(f"03. BB Frame           : {len(bbframe)} bits (header {len(bbheader_bits)} + data {len(tx_input_bits)})")
        
        adapted = stream_adaptation_rate(bbframe, fecframe, rate)
        
        if detailed_report:
            save_intermediate(adapted, "04_adapted", intermediate_dir, frame_num, "bits")
        
        print(f"04. After Scrambling   : {len(adapted)} bits")
        
        # BCH encode
        bch_codeword = bch_encode_bbframe(adapted, fecframe, rate=rate)
        
        if detailed_report:
            save_intermediate(bch_codeword, "05_bch_codeword", intermediate_dir, frame_num, "bits")
        
        print(f"05. BCH Encoded        : {len(bch_codeword)} bits")
        
        # LDPC encode
        ldpc_codeword = ldpc_encoder.encode(bch_codeword, fecframe, rate)
        
        if detailed_report:
            save_intermediate(ldpc_codeword, "06_ldpc_codeword", intermediate_dir, frame_num, "bits")
        
        print(f"06. LDPC Encoded       : {len(ldpc_codeword)} bits")
        
        # Bit interleave
        interleaved = dvbs2_bit_interleave(ldpc_codeword, modulation)
        
        if detailed_report:
            save_intermediate(interleaved, "07_interleaved", intermediate_dir, frame_num, "bits")
        
        print(f"07. Bit Interleaved    : {len(interleaved)} bits")
        
        # Constellation map
        symbols = dvbs2_constellation_map(interleaved, modulation)
        
        if detailed_report:
            save_intermediate(symbols, "08_constellation_symbols", intermediate_dir, frame_num, "complex")
            # Plot constellation
            plot_constellation(
                symbols,
                f"TX Constellation (Frame {frame_num}, {modulation})",
                os.path.join(intermediate_dir, f"frame{frame_num}_07_constellation.png")
            )
        
        print(f"08. Constellation Map  : {len(symbols)} symbols ({modulation})")
        print(f"    Power (avg)        : {np.mean(np.abs(symbols)**2):.4f}")
        
        # Get MODCOD for PL header
        modcod = modcod_from_modulation_rate(modulation, rate)
        
        # Build PL header
        _, plh_syms = build_plheader(modcod, fecframe, pilots=pilots_on)
        
        print(f"09. PL Header          : {len(plh_syms)} symbols")
        
        # Insert pilots
        payload_with_pilots, _ = insert_pilots_into_payload(symbols, pilots_on=pilots_on, fecframe=fecframe)
        
        if detailed_report:
            save_intermediate(payload_with_pilots, "10_with_pilots", intermediate_dir, frame_num, "complex")
        
        print(f"10. Payload+Pilots     : {len(payload_with_pilots)} symbols")
        
        # Concatenate PL header + payload with pilots
        plframe_tx = np.concatenate([plh_syms, payload_with_pilots])
        
        if detailed_report:
            save_intermediate(plframe_tx, "11_plframe_tx", intermediate_dir, frame_num, "complex")
        
        print(f"11. Full PLFRAME       : {len(plframe_tx)} symbols (header {len(plh_syms)} + payload {len(payload_with_pilots)})")
        
        # PL scramble
        plframe_scrambled = pl_scramble_full_plframe(plframe_tx, scrambling_code=scrambling_code, plheader_len=len(plh_syms))
        
        if detailed_report:
            save_intermediate(plframe_scrambled, "12_plframe_scrambled", intermediate_dir, frame_num, "complex")
            # Plot final TX constellation with pilots
            plot_constellation(
                plframe_scrambled,
                f"TX PLFRAME with Pilots (Frame {frame_num}, after scrambling)",
                os.path.join(intermediate_dir, f"frame{frame_num}_10_plframe_tx.png")
            )
        
        print(f"12. PL Scrambled       : {len(plframe_scrambled)} symbols")
        
        # =====================================================================
        # CHANNEL SIMULATION: Add AWGN noise if specified
        # =====================================================================
        print(f"\nCHANNEL")
        print("-" * 70)
        
        if esn0_db is not None:
            signal_power = 1.0
            noise_power = signal_power / (10 ** (esn0_db / 10))
            noise = np.sqrt(noise_power / 2) * (np.random.randn(len(plframe_scrambled)) + 1j * np.random.randn(len(plframe_scrambled)))
            plframe_rx = plframe_scrambled + noise
            snr_measured = 10 * np.log10(np.mean(np.abs(plframe_scrambled)**2) / np.mean(np.abs(noise)**2))
            print(f"AWGN Channel")
            print(f"  Es/N0 (target)       : {esn0_db:.1f} dB")
            print(f"  Es/N0 (measured)     : {snr_measured:.2f} dB")
            print(f"  Signal Power         : {np.mean(np.abs(plframe_scrambled)**2):.4f}")
            print(f"  Noise Power          : {noise_power:.6f}")
        else:
            plframe_rx = plframe_scrambled
            noise_power = 1e-10
            print(f"Noiseless Channel")
        
        if detailed_report:
            # Channel output (post-AWGN)
            save_intermediate(plframe_rx, "13_plframe_rx", intermediate_dir, frame_num, "complex")
            # Plot RX constellation (with noise)
            plot_constellation(
                plframe_rx,
                f"RX PLFRAME (Frame {frame_num}, after channel, SNR={esn0_db if esn0_db is not None else 'inf'} dB)",
                os.path.join(intermediate_dir, f"frame{frame_num}_11_plframe_rx.png")
            )
        
        # =====================================================================
        # RX SIDE: Process received PLFRAME
        # =====================================================================
        print(f"\nRECEIVER CHAIN")
        print("-" * 70)
        
        rx_output = process_rx_plframe(
            plframe_rx,
            fecframe=fecframe,
            scrambling_code=scrambling_code,
            modulation=modulation,
            rate=rate,
            noise_var=noise_power,
            ldpc_mat_path=mat_path,
            ldpc_max_iter=30,
            ldpc_norm=0.75,
            decode_ldpc=True,
        )
        
        # Extract and save intermediate RX outputs
        if detailed_report:
            if rx_output.get("descrambled") is not None:
                save_intermediate(rx_output["descrambled"], "01_descrambled", intermediate_dir, frame_num, "complex")
            if rx_output.get("payload_raw") is not None:
                save_intermediate(rx_output["payload_raw"], "02_payload_raw", intermediate_dir, frame_num, "complex")
            if rx_output.get("payload_corrected") is not None:
                save_intermediate(rx_output["payload_corrected"], "03_payload_corrected", intermediate_dir, frame_num, "complex")
            if rx_output.get("llrs_interleaved") is not None:
                save_intermediate(rx_output["llrs_interleaved"], "04_llrs_interleaved", intermediate_dir, frame_num, "real")
            if rx_output.get("llrs_deinterleaved") is not None:
                save_intermediate(rx_output["llrs_deinterleaved"], "05_llrs_deinterleaved", intermediate_dir, frame_num, "real")
            if rx_output.get("ldpc_bits") is not None:
                save_intermediate(rx_output["ldpc_bits"], "06_ldpc_decoded", intermediate_dir, frame_num, "bits")
            if rx_output.get("bch_payload") is not None:
                save_intermediate(rx_output["bch_payload"], "07_bch_decoded", intermediate_dir, frame_num, "bits")
            if rx_output.get("df_bits") is not None:
                save_intermediate(rx_output["df_bits"], "08_df_bits", intermediate_dir, frame_num, "bits")
        
        print(f"01. Descrambled        : {len(rx_output.get('descrambled', []))} symbols")
        print(f"02. Pilot Removed      : {len(rx_output.get('payload_raw', []))} symbols")
        print(f"03. Phase Corrected    : {len(rx_output.get('payload_corrected', []))} symbols")
        print(f"04. Demapped (LLRs)    : {len(rx_output.get('llrs_interleaved', []))} values")
        print(f"05. Deinterleaved      : {len(rx_output.get('llrs_deinterleaved', []))} values")
        print(f"06. LDPC Decoded       : {len(rx_output.get('ldpc_bits', []))} bits")
        print(f"07. BCH Decoded        : {len(rx_output.get('bch_payload', []))} bits")
        print(f"08. Final DF Bits      : {len(rx_output.get('df_bits', []))} bits")
        
        # =====================================================================
        # EVALUATE RESULTS
        # =====================================================================
        print(f"\nPERFORMANCE ANALYSIS")
        print("-" * 70)
        
        rx_df_bits = rx_output.get("df_bits")
        
        frame_stats = {
            "frame_num": frame_num,
            "tx_bits_shape": list(tx_input_bits.shape),
            "intermediate_saved": detailed_report,
        }
        
        # Compare bits - only compare the actual user data (DFL bits), not padded frame
        # tx_input_bits is DFL (720), rx_df_bits includes padding (1000)
        bits_to_compare = len(tx_input_bits)  # Compare only the DFL bits
        tx_cropped = tx_input_bits[:bits_to_compare]
        rx_cropped = rx_df_bits[:bits_to_compare]  # Extract only first DFL bits from RX

        errors = np.sum(tx_cropped != rx_cropped)
        ber = errors / bits_to_compare if bits_to_compare > 0 else 1.0
        success = errors == 0

        frame_stats.update({
            "rx_bits_shape": list(rx_df_bits.shape),
            "bits_compared": int(bits_to_compare),
            "bit_errors": int(errors),
            "ber": float(ber),
            "frame_success": bool(success),
        })

        print(f"Bits Compared          : {bits_to_compare} (user data only)")
        print(f"Bit Errors             : {errors}/{bits_to_compare}")
        print(f"BER                    : {ber:.6e}")
        print(f"Frame Success          : {'YES' if success else 'NO'}")

        # Add LDPC metrics if available
        if rx_output.get("ldpc_meta") is not None:
            ldpc_meta = rx_output["ldpc_meta"]
            if "iterations" in ldpc_meta:
                print(f"LDPC Iterations        : {ldpc_meta['iterations']}")

        # Add phase estimation metrics
        if rx_output.get("phase_meta") is not None:
            phase_meta = rx_output["phase_meta"]
            if "phase_estimate" in phase_meta:
                print(f"Phase Error Estimate   : {phase_meta.get('phase_estimate', 0):.4f} rad")
        else:
            frame_stats["error"] = "RX decoding failed"
            print("RX Decoding FAILED")
        
        stats["frames"].append(frame_stats)
        
        # Save frame-specific output
        write_bits_single_line(
            os.path.join(output_dir, f"tx_bits_frame{frame_num}.txt"),
            tx_input_bits
        )
        if rx_df_bits is not None:
            write_bits_single_line(
                os.path.join(output_dir, f"rx_bits_frame{frame_num}.txt"),
                rx_df_bits
            )
    
    # =====================================================================
    # SUMMARY STATISTICS
    # =====================================================================
    
    total_frames = len(stats["frames"])
    successful_frames = sum(1 for f in stats["frames"] if f.get("frame_success", False))
    total_errors = sum(f.get("bit_errors", 0) for f in stats["frames"])
    total_bits = sum(f.get("bits_compared", 0) for f in stats["frames"])
    
    stats["summary"] = {
        "total_frames": total_frames,
        "successful_frames": successful_frames,
        "frame_success_rate": successful_frames / total_frames if total_frames > 0 else 0.0,
        "total_bit_errors": int(total_errors),
        "total_bits_tested": int(total_bits),
        "overall_ber": total_errors / total_bits if total_bits > 0 else 1.0,
    }
    
    # Save statistics
    stats_file = os.path.join(output_dir, "loopback_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*60)
    print("LOOPBACK TEST SUMMARY")
    print("="*60)
    print(f"Frames Processed: {total_frames}")
    print(f"Successful Frames: {successful_frames}/{total_frames}")
    print(f"Frame Success Rate: {stats['summary']['frame_success_rate']:.2%}")
    print(f"Total Bit Errors: {total_errors}/{total_bits}")
    print(f"Overall BER: {stats['summary']['overall_ber']:.6e}")
    print(f"Results saved to: {output_dir}/")
    print("="*60)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TX→RX Loopback Test with Detailed Reporting")
    parser.add_argument("--fecframe", default="short", help="normal or short")
    parser.add_argument("--rate", default="1/2", help="Code rate")
    parser.add_argument("--modulation", default="QPSK", help="Modulation scheme")
    parser.add_argument("--no-pilots", action="store_true", help="Disable pilots")
    parser.add_argument("--scrambling-code", type=int, default=0, help="PL scrambling code")
    parser.add_argument("--esn0-db", type=float, default=None, help="Es/N0 in dB (None for noiseless)")
    parser.add_argument("--max-frames", type=int, default=3, help="Number of frames")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: results/loopback)")
    parser.add_argument("--no-detailed", action="store_true", help="Disable detailed stage reporting")
    
    args = parser.parse_args()
    
    # Default output directory
    output_dir = args.output_dir or os.path.join(ROOT, "results", "loopback")
    
    stats = run_tx_rx_loopback(
        fecframe=args.fecframe,
        rate=args.rate,
        modulation=args.modulation,
        pilots_on=not args.no_pilots,
        scrambling_code=args.scrambling_code,
        esn0_db=args.esn0_db,
        max_frames=args.max_frames,
        output_dir=output_dir,
        detailed_report=not args.no_detailed,
    )
