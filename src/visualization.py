"""
AudioClean DSP - Signal Visualization Module
Provides reusable Matplotlib plotting routines for time-domain waveforms,
FFT magnitude spectra, STFT spectrograms, filter transfer functions, and comparative displays.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import align_signal_lengths, convert_to_mono


def plot_waveform(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "Time-Domain Audio Waveform",
    time_range: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> plt.Figure:
    """
    Plot the continuous time-domain amplitude waveform of an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal array (mono 1D or stereo 2D).
    sample_rate : int
        Sampling frequency in Hertz (Hz).
    title : str, optional
        Figure title, default is 'Time-Domain Audio Waveform'.
    time_range : Optional[Tuple[float, float]], optional
        Optional time window in seconds [t_start, t_end] to zoom into.
    save_path : Optional[Union[str, Path]], optional
        Destination path to save figure.
    overwrite : bool, optional
        If False and file exists, raises FileExistsError.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate} Hz.")

    num_samples = audio.shape[0]
    t = np.arange(num_samples, dtype=np.float64) / float(sample_rate)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    if audio.ndim == 1:
        ax.plot(t, audio, color="#1f77b4", linewidth=0.8, label="Mono Audio")
    elif audio.ndim == 2:
        for ch in range(audio.shape[1]):
            ax.plot(t, audio[:, ch], linewidth=0.8, alpha=0.85, label=f"Channel {ch + 1}")
    else:
        plt.close(fig)
        raise ValueError(f"Unsupported audio dimensionality: {audio.ndim}D.")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Amplitude (Normalized)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)

    if time_range is not None:
        if time_range[0] < time_range[1]:
            ax.set_xlim(time_range[0], time_range[1])

    if audio.ndim == 2:
        ax.legend(loc="upper right")

    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path, overwrite)

    return fig


def plot_fft_spectrum(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    db_scale: bool = True,
    title: str = "FFT Magnitude Spectrum",
    max_freq: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> plt.Figure:
    """
    Plot the single-sided discrete Fourier magnitude spectrum.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis in physical Hertz.
    magnitude : np.ndarray
        Magnitude data (linear amplitude or decibels).
    db_scale : bool, optional
        If True, vertical axis is displayed in decibels (dB), default is True.
    title : str, optional
        Figure title.
    max_freq : Optional[float], optional
        Maximum frequency in Hz to display on the horizontal axis.
    save_path : Optional[Union[str, Path]], optional
        File destination path.
    overwrite : bool, optional
        If False and file exists, raises FileExistsError.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))

    y_data = magnitude
    if db_scale:
        # If input magnitude is not yet in dB (e.g. positive linear), convert safely
        if np.all(magnitude >= 0.0) and np.max(magnitude) <= 2.0:
            y_data = 20.0 * np.log10(np.maximum(magnitude, 1e-15))
        ax.set_ylabel("Magnitude (dBFS)", fontsize=10)
        ax.set_ylim(bottom=max(-120.0, float(np.min(y_data)) - 5.0), top=max(5.0, float(np.max(y_data)) + 5.0))
    else:
        ax.set_ylabel("Linear Amplitude", fontsize=10)

    ax.plot(frequencies, y_data, color="#d62728", linewidth=1.0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    if max_freq is not None and max_freq > 0:
        ax.set_xlim(0, max_freq)
    else:
        ax.set_xlim(0, float(frequencies[-1]))

    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path, overwrite)

    return fig


def plot_spectrogram(
    times: np.ndarray,
    frequencies: np.ndarray,
    Sxx_db: np.ndarray,
    title: str = "Audio Spectrogram (STFT)",
    max_freq: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> plt.Figure:
    """
    Plot the Short-Time Fourier Transform (STFT) power spectrogram with a calibrated colorbar.

    Parameters
    ----------
    times : np.ndarray
        Time vector in seconds.
    frequencies : np.ndarray
        Frequency vector in Hertz.
    Sxx_db : np.ndarray
        Logarithmic power spectral density in decibels (dB/Hz).
    title : str, optional
        Figure title.
    max_freq : Optional[float], optional
        Maximum frequency in Hz to display.
    save_path : Optional[Union[str, Path]], optional
        Target image file path.
    overwrite : bool, optional
        Whether to overwrite existing destination.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 4.0))

    # Determine frequency limit
    if max_freq is not None and 0 < max_freq < frequencies[-1]:
        freq_mask = frequencies <= max_freq
        f_plot = frequencies[freq_mask]
        s_plot = Sxx_db[freq_mask, :]
    else:
        f_plot = frequencies
        s_plot = Sxx_db

    mesh = ax.pcolormesh(
        times,
        f_plot,
        s_plot,
        shading="gouraud",
        cmap="magma",
    )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Power Spectral Density (dB/Hz)", fontsize=9)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    ax.set_ylim(0, f_plot[-1])

    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path, overwrite)

    return fig


def plot_filter_response(
    frequencies: np.ndarray,
    magnitude_db: np.ndarray,
    title: str = "Digital Filter Magnitude Response",
    cutoff_hz: Optional[Union[float, Sequence[float]]] = None,
    max_freq: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> plt.Figure:
    """
    Plot the logarithmic frequency response curve |H(e^{j omega})| in decibels.

    Parameters
    ----------
    frequencies : np.ndarray
        Physical frequency axis in Hertz.
    magnitude_db : np.ndarray
        Response magnitude in decibels.
    title : str, optional
        Figure title.
    cutoff_hz : Optional[Union[float, Sequence[float]]], optional
        Cutoff frequency or band boundaries to highlight with dashed marker lines.
    max_freq : Optional[float], optional
        Upper frequency boundary to display.
    save_path : Optional[Union[str, Path]], optional
        Target file path.
    overwrite : bool, optional
        Overwrite permission.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(9, 3.5))

    ax.plot(frequencies, magnitude_db, color="#2ca02c", linewidth=1.5, label="Magnitude Response |H(f)|")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Attenuation (dB)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Highlight -3 dB passband edge
    ax.axhline(-3.0, color="orange", linestyle=":", linewidth=1.0, label="-3 dB Cutoff Threshold")

    if cutoff_hz is not None:
        if isinstance(cutoff_hz, (list, tuple, np.ndarray)):
            for idx, fc in enumerate(cutoff_hz):
                ax.axvline(
                    float(fc),
                    color="red",
                    linestyle="--",
                    linewidth=1.0,
                    label=f"Cutoff fc_{idx+1}={fc:.1f} Hz",
                )
        else:
            ax.axvline(
                float(cutoff_hz),
                color="red",
                linestyle="--",
                linewidth=1.0,
                label=f"Cutoff fc={cutoff_hz:.1f} Hz",
            )

    ax.set_ylim(-100.0, 5.0)

    if max_freq is not None and max_freq > 0:
        ax.set_xlim(0, max_freq)
    else:
        ax.set_xlim(0, float(frequencies[-1]))

    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path, overwrite)

    return fig


def plot_phase_response(
    frequencies: np.ndarray,
    phase_radians: np.ndarray,
    title: str = "Digital Filter Phase Response",
    max_freq: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> plt.Figure:
    """
    Plot the continuous unwrapped phase response curve in radians.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis in Hertz.
    phase_radians : np.ndarray
        Unwrapped phase in radians.
    title : str, optional
        Figure title.
    max_freq : Optional[float], optional
        Maximum frequency displayed.
    save_path : Optional[Union[str, Path]], optional
        Destination path.
    overwrite : bool, optional
        Overwrite permission.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(9, 3.5))

    ax.plot(frequencies, phase_radians, color="#9467bd", linewidth=1.2, label="Unwrapped Phase theta(f)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Phase (radians)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    if max_freq is not None and max_freq > 0:
        ax.set_xlim(0, max_freq)
    else:
        ax.set_xlim(0, float(frequencies[-1]))

    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path, overwrite)

    return fig


def plot_signal_comparison(
    original: np.ndarray,
    noisy: np.ndarray,
    filtered: np.ndarray,
    sample_rate: int,
    time_range: Optional[Tuple[float, float]] = None,
    title: str = "Audio Signal Time-Domain Comparison",
    allow_length_mismatch: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> plt.Figure:
    """
    Plot an aligned 3-panel comparative time-domain display comparing:
    1. Clean Ground Truth Audio
    2. Degraded / Noisy Audio
    3. Filtered Output Audio

    Parameters
    ----------
    original : np.ndarray
        Clean reference audio array.
    noisy : np.ndarray
        Degraded noisy audio array.
    filtered : np.ndarray
        Filtered audio array.
    sample_rate : int
        Sampling frequency in Hertz (Hz).
    time_range : Optional[Tuple[float, float]], optional
        Time window [t_start, t_end] in seconds.
    title : str, optional
        Overall figure title.
    allow_length_mismatch : bool, optional
        If False (default), raises ValueError when input lengths differ.
        If True, aligns signals to shortest length via truncation.
    save_path : Optional[Union[str, Path]], optional
        File destination path.
    overwrite : bool, optional
        Overwrite flag.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate} Hz.")

    # Downmix to mono for comparative visualization
    m_orig = convert_to_mono(original)
    m_noisy = convert_to_mono(noisy)
    m_filt = convert_to_mono(filtered)

    if not (len(m_orig) == len(m_noisy) == len(m_filt)):
        if not allow_length_mismatch:
            raise ValueError(
                f"Signal length mismatch: original ({len(m_orig)}), noisy ({len(m_noisy)}), "
                f"filtered ({len(m_filt)}). Set allow_length_mismatch=True to align automatically."
            )
        # Ensure equal sample lengths via truncation alignment
        m_orig, m_noisy = align_signal_lengths(m_orig, m_noisy, mode="truncate")
        m_orig, m_filt = align_signal_lengths(m_orig, m_filt, mode="truncate")
        m_noisy, _ = align_signal_lengths(m_noisy, m_orig, mode="truncate")

    num_samples = len(m_orig)
    t = np.arange(num_samples, dtype=np.float64) / float(sample_rate)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)

    # 1. Clean Reference
    axes[0].plot(t, m_orig, color="#1f77b4", linewidth=0.8)
    axes[0].set_title("Original Clean Audio s[n]", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Amplitude", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_ylim(-1.05, 1.05)

    # 2. Noisy Signal
    axes[1].plot(t, m_noisy, color="#d62728", linewidth=0.8)
    axes[1].set_title("Degraded Noisy Audio x[n] = s[n] + w[n]", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Amplitude", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].set_ylim(-1.05, 1.05)

    # 3. Filtered Signal
    axes[2].plot(t, m_filt, color="#2ca02c", linewidth=0.8)
    axes[2].set_title("Cleaned Filtered Audio y[n]", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Amplitude", fontsize=9)
    axes[2].set_xlabel("Time (seconds)", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].set_ylim(-1.05, 1.05)

    if time_range is not None and time_range[0] < time_range[1]:
        axes[2].set_xlim(time_range[0], time_range[1])

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path, overwrite)

    return fig


def _save_figure(fig: plt.Figure, save_path: Union[str, Path], overwrite: bool) -> None:
    """Internal helper to safely save figure without accidental overwrites."""
    path = Path(save_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Target plot file already exists: {path.resolve()}. Set overwrite=True to replace."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
