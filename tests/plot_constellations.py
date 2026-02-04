#!/usr/bin/env python3
"""
Plot constellation diagrams from loopback test intermediate files.

This script creates constellation plots for TX and RX symbols to visualize
the signal processing at different stages.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def load_complex_symbols(filepath):
    """Load complex symbols from file (real and imag on same line)."""
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            return np.array([data], dtype=complex)
        return data[:, 0] + 1j * data[:, 1]
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def plot_constellation(symbols, title, output_path, snr_label=None):
    """Create constellation plot with annotations."""
    if symbols is None or len(symbols) == 0:
        print(f"Warning: No symbols to plot for {title}")
        return False
    
    try:
        fig, ax = plt.subplots(figsize=(9, 9))

        # Scatter plot with visible marker styling
        ax.scatter(symbols.real, symbols.imag, alpha=0.7, s=30, facecolors='C0', edgecolors='k')

        # Configure axes
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

        # Set symmetric limits for clarity
        md = np.max(np.abs(symbols)) if len(symbols) > 0 else 1.0
        if md == 0:
            md = 1.0
        lim = md * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        
        # Calculate statistics
        power = np.mean(np.abs(symbols)**2)
        magnitude = np.abs(symbols)
        phase = np.angle(symbols)
        
        # Add annotations
        info_text = f'Power: {power:.4f}\nPoints: {len(symbols)}\nMag (mean): {np.mean(magnitude):.4f}\nPhase (std): {np.std(phase):.4f}'
        if snr_label:
            info_text = f'{snr_label}\n' + info_text
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error plotting {title}: {e}")
        return False


def plot_frame_constellations(intermediate_dir, frame_num, snr=None):
    """Plot all constellation diagrams for a given frame."""
    frame_prefix = f"frame{frame_num}_"
    output_dir = Path(intermediate_dir)
    
    if not output_dir.exists():
        print(f"Error: Intermediate directory not found: {intermediate_dir}")
        return False
    
    print(f"\n{'='*70}")
    print(f"Plotting Constellations - Frame {frame_num}")
    print(f"{'='*70}\n")
    
    # TX Constellation (before pilots)
    tx_symbols_file = output_dir / f"{frame_prefix}07_constellation_symbols.txt"
    if tx_symbols_file.exists():
        symbols = load_complex_symbols(tx_symbols_file)
        plot_constellation(
            symbols,
            f"TX Constellation Map (Frame {frame_num}, before pilots)",
            str(output_dir / f"{frame_prefix}constellation_tx.png")
        )
    
    # TX PLFRAME (with pilots, after scrambling)
    tx_plframe_file = output_dir / f"{frame_prefix}10_plframe_scrambled.txt"
    if tx_plframe_file.exists():
        symbols = load_complex_symbols(tx_plframe_file)
        plot_constellation(
            symbols,
            f"TX PLFRAME (Frame {frame_num}, with pilots & scrambling)",
            str(output_dir / f"{frame_prefix}constellation_tx_plframe.png")
        )
    
    # RX Received Signal (with noise)
    rx_plframe_file = output_dir / f"{frame_prefix}11_plframe_rx.txt"
    if rx_plframe_file.exists():
        symbols = load_complex_symbols(rx_plframe_file)
        snr_label = f"SNR: {snr} dB" if snr else "Received signal"
        plot_constellation(
            symbols,
            f"RX PLFRAME (Frame {frame_num}, after channel)",
            str(output_dir / f"{frame_prefix}constellation_rx.png"),
            snr_label=snr_label
        )
    
    # RX after descrambling
    rx_desc_file = output_dir / f"{frame_prefix}12_descrambled.txt"
    if rx_desc_file.exists():
        symbols = load_complex_symbols(rx_desc_file)
        plot_constellation(
            symbols,
            f"RX Descrambled (Frame {frame_num})",
            str(output_dir / f"{frame_prefix}constellation_rx_descrambled.png"),
            snr_label="After PL descrambling"
        )
    
    # RX after pilot removal
    rx_payload_file = output_dir / f"{frame_prefix}13_payload_raw.txt"
    if rx_payload_file.exists():
        symbols = load_complex_symbols(rx_payload_file)
        plot_constellation(
            symbols,
            f"RX Payload (Frame {frame_num}, after pilot removal)",
            str(output_dir / f"{frame_prefix}constellation_rx_payload.png"),
            snr_label="Pilots removed"
        )
    
    # RX after phase correction
    rx_corrected_file = output_dir / f"{frame_prefix}14_payload_corrected.txt"
    if rx_corrected_file.exists():
        symbols = load_complex_symbols(rx_corrected_file)
        plot_constellation(
            symbols,
            f"RX Phase Corrected (Frame {frame_num})",
            str(output_dir / f"{frame_prefix}constellation_rx_phase_corrected.png"),
            snr_label="After phase correction"
        )
    
    print(f"\n✓ All constellation plots saved to: {output_dir}")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Plot constellation diagrams from DVB-S2x loopback test"
    )
    parser.add_argument(
        "--intermediate-dir",
        default="/home/umair/Videos/DVBS2x/git/results/loopback/intermediate",
        help="Path to intermediate outputs directory"
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=1,
        help="Frame number to plot"
    )
    parser.add_argument(
        "--snr",
        type=float,
        help="SNR in dB (for annotation)"
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Plot all frames found in directory"
    )
    
    args = parser.parse_args()
    
    if args.all_frames:
        # Find all frame directories
        intermediate_dir = Path(args.intermediate_dir)
        frame_nums = set()
        for f in intermediate_dir.glob("frame*_01_tx_input_bits.txt"):
            match = f.name.split('_')[0]  # Extract frame number
            if match.startswith('frame'):
                try:
                    frame_nums.add(int(match[5:]))
                except ValueError:
                    pass
        
        if frame_nums:
            for frame_num in sorted(frame_nums):
                plot_frame_constellations(args.intermediate_dir, frame_num, args.snr)
        else:
            print("No frame files found in directory")
    else:
        plot_frame_constellations(args.intermediate_dir, args.frame, args.snr)


if __name__ == "__main__":
    main()
