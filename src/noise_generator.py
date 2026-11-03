"""
AudioClean DSP - Controlled Noise Generation Engine
Enables calibrated synthesis of acoustic noise, powerline hum, and high-frequency interference
for rigorous digital filter experimentation.
"""

from typing import List, Optional, Tuple, Union
import numpy as np


def add_white_gaussian_noise(
    clean_signal: np.ndarray,
    target_snr_db: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Superimpose zero-mean White Gaussian Noise (WGN) calibrated to a precise target SNR in decibels.

    Mathematical Formulation:
    Signal power P_signal is defined as the mean squared amplitude:
        P_signal = (1 / N) * sum(s[n]^2)

    For a target SNR in decibels:
        SNR_dB = 10 * log10(P_signal / P_noise)

    Rearranging for required noise power P_noise:
        P_noise = P_signal / (10^(SNR_target / 10))

    The required noise standard deviation (RMS) is:
        sigma_noise = sqrt(P_noise)

    Parameters
    ----------
    clean_signal : np.ndarray
        Clean reference audio signal (float array).
    target_snr_db : float
        Target Signal-to-Noise Ratio in decibels (e.g., 0 dB, 10 dB, 20 dB).
    seed : Optional[int], optional
        Random seed for deterministic reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        noisy_signal : np.ndarray
            Degraded signal: s[n] + w[n].
        noise : np.ndarray
            The isolated zero-mean Gaussian noise array w[n].

    Raises
    ------
    ValueError
        If clean_signal has zero power (silent), contains non-finite values, or is empty.
    """
    if not isinstance(clean_signal, np.ndarray):
        raise TypeError("clean_signal must be a NumPy array.")

    if clean_signal.size == 0:
        raise ValueError("Cannot add noise to an empty signal.")

    if not np.all(np.isfinite(clean_signal)):
        raise ValueError("clean_signal contains NaN or infinite values.")

    # Calculate clean signal power
    p_signal = float(np.mean(clean_signal.astype(np.float64)**2))

    if p_signal == 0.0:
        raise ValueError(
            "Cannot calibrate SNR on a silent (zero-power) audio signal. "
            "P_signal is zero, rendering SNR undefined."
        )

    # Compute target noise power and standard deviation
    p_noise = p_signal / (10.0 ** (target_snr_db / 10.0))
    sigma_noise = float(np.sqrt(p_noise))

    # Generate zero-mean Gaussian distribution
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma_noise, size=clean_signal.shape).astype(np.float64)

    noisy_signal = clean_signal.astype(np.float64) + noise
    return noisy_signal, noise


def add_sinusoidal_hum(
    signal: np.ndarray,
    sample_rate: int,
    freq: float = 50.0,
    amplitude: float = 0.1,
    harmonics: Optional[List[int]] = None,
    harmonic_decay: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject sinusoidal electrical interference (e.g., 50 Hz power-line mains hum) with optional harmonics.

    Mathematical Formulation:
    Fundamental hum:
        h_0(t) = A * sin(2 * pi * f_0 * t)
    Harmonic series:
        h(t) = h_0(t) + sum_{k in harmonics} (A * decay^(k-1)) * sin(2 * pi * (k * f_0) * t)

    Where t = n / f_s for discrete sample index n = 0, 1, ..., N-1.

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal array.
    sample_rate : int
        Sampling frequency in Hertz (Hz).
    freq : float, optional
        Fundamental interference frequency in Hz, default is 50.0 Hz.
    amplitude : float, optional
        Peak amplitude of the fundamental component, default is 0.1.
    harmonics : Optional[List[int]], optional
        List of harmonic multipliers (e.g., [2, 3] for 100 Hz and 150 Hz).
    harmonic_decay : float, optional
        Exponential attenuation factor applied per harmonic order, default is 0.5.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        noisy_signal : np.ndarray
            Degraded audio signal with electrical hum.
        hum : np.ndarray
            Isolated sinusoidal hum array matching the input signal shape.

    Raises
    ------
    ValueError
        If frequency or any harmonic equals or exceeds the Nyquist frequency (f_s / 2),
        or if sample_rate <= 0, or amplitude <= 0.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("signal must be a NumPy array.")

    if signal.size == 0:
        raise ValueError("Cannot add hum to an empty signal.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains NaN or infinite values.")

    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate} Hz.")

    if amplitude <= 0.0:
        raise ValueError(f"Amplitude must be strictly positive, got {amplitude}.")

    nyquist = sample_rate / 2.0
    if freq <= 0.0 or freq >= nyquist:
        raise ValueError(
            f"Interference frequency {freq} Hz violates the Nyquist criterion (0 < f < {nyquist} Hz)."
        )

    num_samples = signal.shape[0]
    # Discrete time vector: t = n / f_s
    t = np.arange(num_samples, dtype=np.float64) / float(sample_rate)

    # Fundamental 50 Hz component
    hum = amplitude * np.sin(2.0 * np.pi * freq * t)

    # Add harmonic overtones
    if harmonics is not None:
        for k in harmonics:
            if not isinstance(k, (int, np.integer)) or k < 2:
                raise ValueError(f"Harmonic multiplier must be integer >= 2, got {k}.")
            harmonic_freq = k * freq
            if harmonic_freq >= nyquist:
                raise ValueError(
                    f"Harmonic {k} ({harmonic_freq} Hz) exceeds the Nyquist frequency ({nyquist} Hz)."
                )
            h_amp = amplitude * (harmonic_decay ** (k - 1))
            hum += h_amp * np.sin(2.0 * np.pi * harmonic_freq * t)

    # Match channel dimensionality
    if signal.ndim == 2:
        hum = np.tile(hum[:, np.newaxis], (1, signal.shape[1]))

    noisy_signal = signal.astype(np.float64) + hum
    return noisy_signal, hum


def add_high_frequency_noise(
    signal: np.ndarray,
    sample_rate: int,
    freq: float,
    amplitude: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate controllable high-frequency sinusoidal noise to test low-pass and band-stop filters.

    Mathematical Formulation:
        w_hf[n] = A * sin(2 * pi * f_hf * (n / f_s))

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal array.
    sample_rate : int
        Sampling frequency in Hertz (Hz).
    freq : float
        Interference frequency in Hertz.
    amplitude : float, optional
        Peak amplitude of the tone, default is 0.1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        noisy_signal : np.ndarray
            Degraded audio signal with high-frequency interference.
        hf_noise : np.ndarray
            Isolated high-frequency noise array.

    Raises
    ------
    ValueError
        If freq <= 0 or freq >= Nyquist (f_s / 2).
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("signal must be a NumPy array.")

    if signal.size == 0:
        raise ValueError("Cannot add noise to an empty signal.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains NaN or infinite values.")

    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate} Hz.")

    nyquist = sample_rate / 2.0
    if freq <= 0.0 or freq >= nyquist:
        raise ValueError(
            f"Frequency {freq} Hz must satisfy 0 < freq < Nyquist ({nyquist} Hz)."
        )

    num_samples = signal.shape[0]
    t = np.arange(num_samples, dtype=np.float64) / float(sample_rate)
    hf_noise = amplitude * np.sin(2.0 * np.pi * freq * t)

    if signal.ndim == 2:
        hf_noise = np.tile(hf_noise[:, np.newaxis], (1, signal.shape[1]))

    noisy_signal = signal.astype(np.float64) + hf_noise
    return noisy_signal, hf_noise


def add_noise_combination(
    signal: np.ndarray,
    noise_components: List[np.ndarray],
    prevent_clipping: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Superimpose multiple discrete noise components onto an audio signal.

    Parameters
    ----------
    signal : np.ndarray
        Original clean audio signal.
    noise_components : List[np.ndarray]
        List of noise arrays to add.
    prevent_clipping : bool, optional
        If True, attenuates the combined signal proportionally if peak amplitude exceeds unity (1.0).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        noisy_signal : np.ndarray
            Composite noisy audio signal.
        total_noise : np.ndarray
            Aggregated sum of all noise components.

    Raises
    ------
    ValueError
        If noise_components is empty or arrays have incompatible shapes.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("signal must be a NumPy array.")

    if not noise_components:
        raise ValueError("noise_components list cannot be empty.")

    total_noise = np.zeros_like(signal, dtype=np.float64)

    for i, n_comp in enumerate(noise_components):
        if not isinstance(n_comp, np.ndarray):
            raise TypeError(f"Noise component {i} must be a NumPy array.")
        if n_comp.shape != signal.shape:
            raise ValueError(
                f"Noise component {i} shape {n_comp.shape} does not match signal shape {signal.shape}."
            )
        total_noise += n_comp.astype(np.float64)

    noisy_signal = signal.astype(np.float64) + total_noise

    max_amp = float(np.max(np.abs(noisy_signal)))
    if prevent_clipping and max_amp > 1.0:
        # Scale proportionally to fit within [-1.0, 1.0]
        scaling = 1.0 / max_amp
        noisy_signal *= scaling
        total_noise *= scaling

    return noisy_signal, total_noise
