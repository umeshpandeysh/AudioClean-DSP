"""
AudioClean DSP - FFT Analysis and Spectral Processing Module
Provides discrete Fourier transform (FFT) magnitude spectrum calculations and
Short-Time Fourier Transform (STFT) spectrogram analysis for audio signals.
"""

from typing import Optional, Tuple
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import spectrogram

from src.preprocessing import convert_to_mono, validate_audio


SUPPORTED_FFT_WINDOWS = ("hann", "hamming", "blackman", "rectangular")


def compute_fft(
    audio: np.ndarray,
    sample_rate: int,
    window: str = "hann",
    n_fft: Optional[int] = None,
    channel: Optional[int] = None,
    allow_truncation: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the calibrated single-sided Fast Fourier Transform (FFT) magnitude spectrum
    of a real-valued discrete audio signal.

    Mathematical Formulation:
    For a discrete-time signal x[n] windowed by w[n] of length N:
        X[k] = sum_{n=0}^{N-1} x[n] w[n] e^{-j 2 pi k n / N}

    The coherent window gain S_1 is:
        S_1 = sum_{n=0}^{N-1} w[n]

    The single-sided magnitude spectrum is calibrated so that an input sinusoid
    A * sin(2 pi f t) yields a peak magnitude equal to its peak amplitude A:
        Magnitude[0] = |X[0]| / S_1                 (DC component)
        Magnitude[k] = 2 * |X[k]| / S_1             (0 < k < Nyquist bin)
        Magnitude[N/2] = |X[N/2]| / S_1             (Nyquist bin, if n_fft is even)

    Logarithmic scaling:
        Magnitude_dB = 20 * log10(max(Magnitude, 1e-15))

    Parameters
    ----------
    audio : np.ndarray
        Discrete audio signal. If 2D stereo and channel is None, downmixed to mono
        via arithmetic averaging.
    sample_rate : int
        Sampling frequency in Hertz (Hz). Must be strictly positive.
    window : str, optional
        Windowing function: 'hann' (default), 'hamming', 'blackman', or 'rectangular' (no window).
    n_fft : Optional[int], optional
        FFT block size. If None, defaults to len(audio).
        If n_fft > len(audio), the signal is zero-padded up to n_fft.
        Note: Zero-padding interpolates the DTFT spectrum but does not improve the physical
        Rayleigh frequency resolution (df_physical = fs / N_samples).
    channel : Optional[int], optional
        If audio is 2D multichannel, specifies which channel index to analyze.
        If None (default), downmixes to mono.
    allow_truncation : bool, optional
        If False (default), raises ValueError when n_fft < len(audio).
        If True, permits slicing the first n_fft samples.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        frequencies_hz : np.ndarray
            Frequency bins in physical Hertz, ranging from 0 to f_s / 2.
        magnitude : np.ndarray
            Calibrated linear peak magnitude spectrum.
        magnitude_db : np.ndarray
            Logarithmic magnitude spectrum in decibels (floored at -300 dB).

    Raises
    ------
    TypeError
        If input types are invalid.
    ValueError
        If sample_rate <= 0, audio is empty/invalid, n_fft <= 0, or n_fft < len(audio) without allow_truncation.
    """
    validate_audio(audio, sample_rate)

    # Standardize to mono 1D float64
    if channel is not None:
        if audio.ndim != 2:
            raise ValueError(f"channel parameter specified ({channel}), but audio is {audio.ndim}D.")
        if channel < 0 or channel >= audio.shape[1]:
            raise ValueError(f"channel {channel} out of bounds for audio with {audio.shape[1]} channels.")
        mono_signal = np.ascontiguousarray(audio[:, channel], dtype=np.float64)
    else:
        mono_signal = convert_to_mono(audio)

    num_samples = len(mono_signal)

    if n_fft is not None:
        if not isinstance(n_fft, (int, np.integer)) or n_fft <= 0:
            raise ValueError(f"n_fft must be a strictly positive integer, got {n_fft}.")
        fft_len = int(n_fft)
        if fft_len < num_samples and not allow_truncation:
            raise ValueError(
                f"Requested n_fft ({fft_len}) is smaller than signal length ({num_samples} samples). "
                f"Truncation will lose signal energy. Set allow_truncation=True to permit explicit truncation."
            )
    else:
        fft_len = num_samples

    clean_window = window.strip().lower() if isinstance(window, str) else "hann"
    if clean_window not in SUPPORTED_FFT_WINDOWS:
        raise ValueError(
            f"Unsupported window: '{window}'. Supported windows are: {SUPPORTED_FFT_WINDOWS}."
        )

    # Handle signal slicing or zero-padding
    if num_samples >= fft_len:
        proc_signal = mono_signal[:fft_len]
        win_len = fft_len
    else:
        # Zero-pad signal to match requested n_fft
        proc_signal = np.zeros(fft_len, dtype=np.float64)
        proc_signal[:num_samples] = mono_signal
        win_len = num_samples

    # Generate tapered window
    if clean_window == "rectangular":
        win = np.ones(win_len, dtype=np.float64)
    elif clean_window == "hann":
        win = np.hanning(win_len).astype(np.float64)
    elif clean_window == "hamming":
        win = np.hamming(win_len).astype(np.float64)
    elif clean_window == "blackman":
        win = np.blackman(win_len).astype(np.float64)
    else:
        win = np.ones(win_len, dtype=np.float64)

    # Coherent gain S_1
    s1 = float(np.sum(win))
    if s1 == 0.0:
        s1 = 1.0

    # Apply window to active samples
    windowed_buffer = np.zeros(fft_len, dtype=np.float64)
    windowed_buffer[:win_len] = proc_signal[:win_len] * win

    # Compute single-sided discrete real Fourier transform
    yf = rfft(windowed_buffer, n=fft_len)
    xf = rfftfreq(fft_len, 1.0 / float(sample_rate)).astype(np.float64)

    # Calibrate single-sided amplitude spectrum
    raw_mag = np.abs(yf) / s1
    magnitude = raw_mag.copy()

    # Double non-DC and non-Nyquist bins to fold negative frequency energy
    if fft_len > 2:
        if fft_len % 2 == 0:
            # Nyquist bin is at index -1 (do not double DC at index 0 or Nyquist at index -1)
            magnitude[1:-1] *= 2.0
        else:
            # Odd length: no exact Nyquist bin
            magnitude[1:] *= 2.0

    # Logarithmic decibel conversion floored at 1e-15 (-300 dB)
    magnitude_db = (20.0 * np.log10(np.maximum(magnitude, 1e-15))).astype(np.float64)

    return xf, magnitude.astype(np.float64), magnitude_db


def compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    nperseg: int = 1024,
    noverlap: Optional[int] = None,
    window: str = "hann",
    scaling: str = "density",
    channel: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform (STFT) power spectrogram of an audio signal.

    Decibel Scaling Convention:
    - For scaling='density': Output is Power Spectral Density (PSD) in dBFS/Hz,
      computed via 10 * log10(max(Sxx, 1e-15)).
    - For scaling='spectrum': Output is Power Spectrum in dBFS,
      computed via 10 * log10(max(Sxx, 1e-15)).
    In both cases, 10 * log10 is used because Sxx represents power (squared amplitude).

    Parameters
    ----------
    audio : np.ndarray
        Audio signal array. Downmixed to mono if stereo and channel is None.
    sample_rate : int
        Sampling frequency in Hertz (Hz).
    nperseg : int, optional
        Length of each Fourier segment in samples, default is 1024.
    noverlap : Optional[int], optional
        Number of overlapping samples between segments, default is nperseg // 2.
    window : str, optional
        Windowing function, default is 'hann'.
    scaling : str, optional
        'density' for Power Spectral Density (V^2/Hz) or 'spectrum' for Power Spectrum (V^2).
        Default is 'density'.
    channel : Optional[int], optional
        Channel index if audio is multichannel. If None, downmixes to mono.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        times : np.ndarray
            Time vector in physical seconds corresponding to the center of each STFT segment.
        frequencies_hz : np.ndarray
            Frequency axis in physical Hertz.
        Sxx_db : np.ndarray
            Logarithmic power spectral density in decibels: 10 * log10(max(Sxx, 1e-15)).

    Raises
    ------
    ValueError
        If nperseg <= 0, noverlap >= nperseg, audio length < nperseg, or invalid channel.
    """
    validate_audio(audio, sample_rate)

    if channel is not None:
        if audio.ndim != 2:
            raise ValueError(f"channel parameter specified ({channel}), but audio is {audio.ndim}D.")
        if channel < 0 or channel >= audio.shape[1]:
            raise ValueError(f"channel {channel} out of bounds for audio with {audio.shape[1]} channels.")
        mono_signal = np.ascontiguousarray(audio[:, channel], dtype=np.float64)
    else:
        mono_signal = convert_to_mono(audio)

    num_samples = len(mono_signal)

    if nperseg <= 0:
        raise ValueError(f"nperseg must be strictly positive, got {nperseg}.")

    if num_samples < nperseg:
        raise ValueError(
            f"Audio length ({num_samples} samples) is too short for spectrogram segment "
            f"length nperseg={nperseg}. Provide at least {nperseg} samples or reduce nperseg."
        )

    if noverlap is None:
        noverlap_val = nperseg // 2
    else:
        if noverlap < 0 or noverlap >= nperseg:
            raise ValueError(
                f"noverlap ({noverlap}) must satisfy 0 <= noverlap < nperseg ({nperseg})."
            )
        noverlap_val = int(noverlap)

    clean_scaling = scaling.strip().lower()
    if clean_scaling not in ("density", "spectrum"):
        raise ValueError(f"Unsupported scaling: '{scaling}'. Must be 'density' or 'spectrum'.")

    f, t, Sxx = spectrogram(
        x=mono_signal,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap_val,
        scaling=clean_scaling,
    )

    # Convert power density/spectrum to decibels (10 * log10)
    Sxx_db = (10.0 * np.log10(np.maximum(Sxx, 1e-15))).astype(np.float64)

    return t.astype(np.float64), f.astype(np.float64), Sxx_db
