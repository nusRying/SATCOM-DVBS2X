import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn

def plot_qpsk_fft(symbols: np.ndarray, fs: float = 1.0, title="QPSK Spectrum"):
    """
    Plot FFT magnitude of complex QPSK symbols.

    Parameters
    ----------
    symbols : np.ndarray (complex)
        Complex baseband symbols (QPSK)
    fs : float
        Sampling rate (normalized = 1.0 is fine)
    title : str
        Plot title
    """
    symbols = np.asarray(symbols, dtype=np.complex128)

    N = len(symbols)
    if N == 0:
        raise ValueError("Empty symbol array")

    # Remove DC offset (important)
    symbols = symbols - np.mean(symbols)

    # FFT
    fft_vals = np.fft.fftshift(np.fft.fft(symbols))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))

    # Magnitude in dB
    mag_db = 20 * np.log10(np.abs(fft_vals) + 1e-12)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, mag_db)
    plt.grid(True)
    plt.xlabel("Normalized Frequency (cycles/sample)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def rrc_filter(beta: float, span: int, sps: int) -> np.ndarray:
    """
    Root raised cosine (RRC) filter taps.
    """
    if span <= 0 or sps <= 0:
        raise ValueError("span and sps must be positive")
    if beta < 0 or beta > 1:
        raise ValueError("beta must be in [0, 1]")

    n = span * sps
    t = np.arange(-n / 2, n / 2 + 1, dtype=np.float64) / sps
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(ti) == 1 / (4 * beta):
            h[i] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = (
                np.sin(np.pi * ti * (1 - beta)) +
                4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            )
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den

    # Normalize energy
    h /= np.sqrt(np.sum(h * h))
    return h


def plot_dvbs2_qpsk_spectrum(
    symbols: np.ndarray,
    alpha: float = 0.35,
    sps: int = 8,
    span: int = 10,
    title: str = None,
):
    """
    Plot DVB-S2-like QPSK spectrum with RRC pulse shaping.
    """
    symbols = np.asarray(symbols, dtype=np.complex128)
    if symbols.size == 0:
        raise ValueError("Empty symbol array")

    if title is None:
        title = f"DVB-S2 QPSK Spectrum (alpha={alpha})"

    h = rrc_filter(alpha, span=span, sps=sps)
    x = upfirdn(h, symbols, up=sps)

    n = x.size
    X = np.fft.fftshift(np.fft.fft(x))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1 / sps))
    mag_db = 20 * np.log10(np.abs(X) + 1e-12)

    plt.figure(figsize=(10, 5))
    plt.plot(f, mag_db)
    plt.grid(True)
    plt.xlabel("Normalized Frequency (cycles/symbol)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
