#!/usr/bin/env python3
"""
Intermediate Signal Analysis Tool for DVB-S2x Loopback Test

Analyzes intermediate output files from test_tx_rx_loopback.py to provide
detailed signal processing insights at each stage.
"""

import os
import sys
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, Dict, List

class IntermediateAnalyzer:
    """Analyzes intermediate signals from DVB-S2x loopback test."""
    
    def __init__(self, intermediate_dir: str, frame_num: int = 1):
        """
        Initialize analyzer.
        
        Args:
            intermediate_dir: Path to intermediate outputs directory
            frame_num: Frame number to analyze
        """
        self.intermediate_dir = Path(intermediate_dir)
        self.frame_num = frame_num
        self.frame_prefix = f"frame{frame_num}_"
        
        if not self.intermediate_dir.exists():
            raise FileNotFoundError(f"Directory not found: {intermediate_dir}")
    
    def load_bits(self, stage_num: int, stage_name: str) -> np.ndarray:
        """Load binary signal from file."""
        filename = self.intermediate_dir / f"{self.frame_prefix}{stage_num:02d}_{stage_name}.txt"
        if filename.exists():
            with open(filename, 'r') as f:
                content = f.read().replace('\n', '').strip()
            bits = np.array([int(b) for b in content])
            return bits
        raise FileNotFoundError(f"File not found: {filename}")
    
    def load_complex(self, stage_num: int, stage_name: str) -> np.ndarray:
        """Load complex signal from file."""
        filename = self.intermediate_dir / f"{self.frame_prefix}{stage_num:02d}_{stage_name}.txt"
        if filename.exists():
            data = np.loadtxt(filename).flatten()
            if data.ndim == 0:
                data = np.array([data])
            elif data.ndim == 1 and len(data) % 2 == 0:
                # Reshape from flat [real, imag, real, imag, ...] to complex pairs
                data = data.reshape(-1, 2)
            if data.ndim == 2:
                return data[:, 0] + 1j * data[:, 1]
            return data.astype(complex)
        raise FileNotFoundError(f"File not found: {filename}")
    
    def load_real(self, stage_num: int, stage_name: str) -> np.ndarray:
        """Load real-valued signal from file."""
        filename = self.intermediate_dir / f"{self.frame_prefix}{stage_num:02d}_{stage_name}.txt"
        if filename.exists():
            return np.loadtxt(filename).flatten()
        raise FileNotFoundError(f"File not found: {filename}")
    
    def analyze_bits(self, bits: np.ndarray, label: str) -> Dict:
        """Analyze binary signal statistics."""
        return {
            "label": label,
            "length": len(bits),
            "ones": np.sum(bits),
            "zeros": np.sum(bits == 0),
            "one_ratio": float(np.sum(bits) / len(bits))
        }
    
    def analyze_complex(self, signal: np.ndarray, label: str) -> Dict:
        """Analyze complex signal statistics."""
        magnitude = np.abs(signal)
        phase = np.angle(signal)
        
        return {
            "label": label,
            "length": len(signal),
            "avg_magnitude": float(np.mean(magnitude)),
            "std_magnitude": float(np.std(magnitude)),
            "avg_power": float(np.mean(magnitude**2)),
            "peak_power": float(np.max(magnitude**2)),
            "avg_phase": float(np.mean(phase)),
            "phase_std": float(np.std(phase))
        }
    
    def analyze_llr(self, llrs: np.ndarray, label: str) -> Dict:
        """Analyze log-likelihood ratio statistics."""
        abs_llrs = np.abs(llrs)
        
        return {
            "label": label,
            "length": len(llrs),
            "mean": float(np.mean(llrs)),
            "std": float(np.std(llrs)),
            "min": float(np.min(llrs)),
            "max": float(np.max(llrs)),
            "mean_magnitude": float(np.mean(abs_llrs)),
            "high_confidence_count": int(np.sum(abs_llrs > 5.0)),
            "high_confidence_ratio": float(np.sum(abs_llrs > 5.0) / len(llrs))
        }
    
    def analyze_tx_chain(self) -> Dict:
        """Analyze entire transmitter chain."""
        results = {}
        
        try:
            # TX stages - updated numbering based on actual file names
            input_bits = self.load_bits(1, "tx_input_bits")
            results["01_tx_input"] = self.analyze_bits(input_bits, "TX Input Bits")
            
            bb_frame = self.load_bits(3, "bbframe")
            results["03_bb_frame"] = self.analyze_bits(bb_frame, "BB Frame")
            
            adapted = self.load_bits(4, "adapted")
            results["04_adapted"] = self.analyze_bits(adapted, "After Scrambling")
            
            bch = self.load_bits(5, "bch_codeword")
            results["05_bch"] = self.analyze_bits(bch, "BCH Codeword")
            
            ldpc = self.load_bits(6, "ldpc_codeword")
            results["06_ldpc"] = self.analyze_bits(ldpc, "LDPC Codeword")
            
            interleaved = self.load_bits(7, "interleaved")
            results["07_interleaved"] = self.analyze_bits(interleaved, "Bit Interleaved")
            
            # Constellation symbols
            symbols = self.load_complex(8, "constellation_symbols")
            results["08_constellation"] = self.analyze_complex(symbols, "Constellation Symbols")
            
            # PLFRAME
            plframe = self.load_complex(11, "plframe_tx")
            results["11_plframe"] = self.analyze_complex(plframe, "PLFRAME TX")
            
            scrambled = self.load_complex(12, "plframe_scrambled")
            results["12_scrambled"] = self.analyze_complex(scrambled, "PL Scrambled")
            
        except FileNotFoundError as e:
            print(f"Warning: {e}")
        
        return results
    
    def analyze_rx_chain(self) -> Dict:
        """Analyze entire receiver chain."""
        results = {}
        
        try:
            # RX stages - numbering is local to RX chain
            descrambled = self.load_complex(1, "descrambled")
            results["01_descrambled"] = self.analyze_complex(descrambled, "Descrambled")
            
            payload_raw = self.load_complex(2, "payload_raw")
            results["02_payload_raw"] = self.analyze_complex(payload_raw, "Payload Raw")
            
            payload_corrected = self.load_complex(3, "payload_corrected")
            results["03_payload_corrected"] = self.analyze_complex(payload_corrected, "Payload Corrected")
            
            # LLRs
            llrs_int = self.load_real(4, "llrs_interleaved")
            results["04_llrs_int"] = self.analyze_llr(llrs_int, "LLRs Interleaved")
            
            llrs_deint = self.load_real(5, "llrs_deinterleaved")
            results["05_llrs_deint"] = self.analyze_llr(llrs_deint, "LLRs Deinterleaved")
            
            # Decoded outputs
            ldpc_decoded = self.load_bits(6, "ldpc_decoded")
            results["06_ldpc_decoded"] = self.analyze_bits(ldpc_decoded, "LDPC Decoded")
            
            bch_decoded = self.load_bits(7, "bch_decoded")
            results["07_bch_decoded"] = self.analyze_bits(bch_decoded, "BCH Decoded")
            
            final_bits = self.load_bits(8, "df_bits")
            results["08_final_bits"] = self.analyze_bits(final_bits, "Final DF Bits")
            
        except FileNotFoundError as e:
            print(f"Warning: {e}")
        
        return results
    
    def compare_bits(self, bits1: np.ndarray, bits2: np.ndarray, 
                    label1: str = "TX", label2: str = "RX") -> Dict:
        """Compare two bit sequences."""
        min_len = min(len(bits1), len(bits2))
        errors = np.sum(bits1[:min_len] != bits2[:min_len])
        
        return {
            "bits_compared": min_len,
            "bit_errors": int(errors),
            "ber": float(errors / min_len) if min_len > 0 else 0.0,
            "match_ratio": float((min_len - errors) / min_len) if min_len > 0 else 0.0
        }
    
    def print_report(self, verbose: bool = True):
        """Print comprehensive analysis report."""
        print(f"\n{'='*70}")
        print(f"INTERMEDIATE SIGNAL ANALYSIS - Frame {self.frame_num}")
        print(f"{'='*70}\n")
        
        # TX Chain Analysis
        print("📤 TRANSMITTER CHAIN ANALYSIS")
        print("-" * 70)
        tx_results = self.analyze_tx_chain()
        
        for key in sorted(tx_results.keys()):
            stats = tx_results[key]
            label = stats.pop("label")
            print(f"\n{key}: {label}")
            for stat_name, stat_val in stats.items():
                if isinstance(stat_val, float):
                    print(f"  {stat_name:20s}: {stat_val:.6f}")
                else:
                    print(f"  {stat_name:20s}: {stat_val}")
        
        # RX Chain Analysis
        print("\n\n📥 RECEIVER CHAIN ANALYSIS")
        print("-" * 70)
        rx_results = self.analyze_rx_chain()
        
        for key in sorted(rx_results.keys()):
            stats = rx_results[key]
            label = stats.pop("label")
            print(f"\n{key}: {label}")
            for stat_name, stat_val in stats.items():
                if isinstance(stat_val, float):
                    print(f"  {stat_name:20s}: {stat_val:.6f}")
                else:
                    print(f"  {stat_name:20s}: {stat_val}")
        
        # Bit Comparison
        print("\n\n🔄 BIT-LEVEL COMPARISON")
        print("-" * 70)
        try:
            tx_bits = self.load_bits(1, "tx_input_bits")
            final_bits = self.load_bits(8, "df_bits")
            comparison = self.compare_bits(tx_bits, final_bits)
            
            for stat_name, stat_val in comparison.items():
                if isinstance(stat_val, float):
                    print(f"  {stat_name:20s}: {stat_val:.6e}")
                else:
                    print(f"  {stat_name:20s}: {stat_val}")
        except FileNotFoundError:
            print("  Could not load bit files for comparison")
        
        print(f"\n{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze intermediate signals from DVB-S2x loopback test"
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
        help="Frame number to analyze"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = IntermediateAnalyzer(args.intermediate_dir, args.frame)
        analyzer.print_report(verbose=args.verbose)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
