"""
AudioClean DSP - Preprocessing Module
Provides core mathematical transformations and validation for discrete audio signals.
All DSP operations in AudioClean DSP use 64-bit floating-point (np.float64) representations.
"""

from typing import Tuple
import numpy as np


def validate_audio(data: np.ndarray, sample_rate: int) -> bool:
    """
    Validate that an audio signal satisfies essential physical and numerical constraints.

    Parameters
    ----------
    data : np.ndarray
        Audio signal array.
    sample_rate : int
        Sampling frequency in Hertz (Hz).

    Returns
    -------
    bool
        True if all validation checks pass.

    Raises
    ------
    TypeError
        If data is not a NumPy array or sample_rate is not an integer.
    ValueError
        If data is empty, non-finite (contains NaN or Inf), has invalid dimensions,
        unsupported channel count (> 2), or non-positive sample rate.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Audio data must be a numpy.ndarray, got {type(data).__name__}.")

    if not isinstance(sample_rate, (int, np.integer)):
        raise TypeError(f"Sample rate must be an integer, got {type(sample_rate).__name__}.")

    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be strictly positive, got {sample_rate} Hz.")

    if data.size == 0 or data.shape[0] == 0:
        raise ValueError("Audio data array is empty.")

    if not np.all(np.isfinite(data)):
        raise ValueError("Audio data contains invalid numerical values (NaN or Infinity).")

    if data.ndim == 1:
        # Valid mono signal
        return True
    elif data.ndim == 2:
        channels = data.shape[1]
        if channels not in (1, 2):
            raise ValueError(
                f"Unsupported channel count: {channels}. Only mono (1) and stereo (2) are supported."
            )
        return True
    else:
        raise ValueError(f"Invalid array dimensionality: {data.ndim}D. Expected 1D or 2D array.")


def convert_to_mono(data: np.ndarray) -> np.ndarray:
    """
    Convert an audio signal to single-channel (mono) using arithmetic channel averaging.

    Mathematical Formulation:
    For a two-channel discrete stereo signal [x_L[n], x_R[n]], the downmixed mono signal is:
        x_mono[n] = 0.5 * (x_L[n] + x_R[n])

    This linear combination preserves coherent in-phase acoustic content while preventing
    amplitude inflation.

    Parameters
    ----------
    data : np.ndarray
        Audio signal array, shape (N,) or (N, C).

    Returns
    -------
    np.ndarray
        Mono 1D array of shape (N,) with dtype np.float64.

    Raises
    ------
    ValueError
        If the audio data is empty or has unsupported dimensions.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Audio data must be a NumPy array.")

    if data.size == 0:
        raise ValueError("Cannot convert empty array to mono.")

    if not np.all(np.isfinite(data)):
        raise ValueError("Audio data contains NaN or infinite values.")

    # Convert to float64 for internal precision
    arr = data.astype(np.float64, copy=True)

    if arr.ndim == 1:
        return arr
    elif arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr[:, 0]
        elif arr.shape[1] == 2:
            # Arithmetic mean of Left and Right channels
            return np.mean(arr, axis=1)
        else:
            raise ValueError(f"Unsupported channel count: {arr.shape[1]}. Expected 1 or 2.")
    else:
        raise ValueError(f"Unsupported dimensionality: {arr.ndim}D.")


def normalize_audio(data: np.ndarray, target_peak: float = 1.0) -> np.ndarray:
    """
    Apply peak amplitude normalization to scale an audio signal within [-target_peak, target_peak].

    Mathematical Formulation:
        s_norm[n] = s[n] * (target_peak / max(|s[k]|))

    If the signal is purely silent (all zeros), it is returned unmodified to prevent division by zero.

    Parameters
    ----------
    data : np.ndarray
        Audio signal array.
    target_peak : float, optional
        Target maximum absolute amplitude, default is 1.0. Must be in range (0.0, 1.0].

    Returns
    -------
    np.ndarray
        Peak-normalized signal array with dtype np.float64.

    Raises
    ------
    ValueError
        If target_peak is <= 0 or > 1.0, or if data contains non-finite values.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Audio data must be a NumPy array.")

    if target_peak <= 0.0 or target_peak > 1.0:
        raise ValueError(f"target_peak must be in range (0.0, 1.0], got {target_peak}.")

    if not np.all(np.isfinite(data)):
        raise ValueError("Cannot normalize array containing NaN or infinite values.")

    arr = data.astype(np.float64, copy=True)
    if arr.size == 0:
        return arr

    peak = float(np.max(np.abs(arr)))

    # Prevent division by zero on silent recordings
    if peak == 0.0:
        return arr

    scaling_factor = target_peak / peak
    return arr * scaling_factor


def calculate_duration(num_samples: int, sample_rate: int) -> float:
    """
    Calculate the duration of a discrete-time signal in seconds.

    Mathematical Formulation:
        T = N / f_s

    Parameters
    ----------
    num_samples : int
        Number of discrete samples.
    sample_rate : int
        Sampling frequency in Hertz (Hz).

    Returns
    -------
    float
        Duration in seconds.
    """
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate} Hz.")
    if num_samples < 0:
        raise ValueError(f"Number of samples cannot be negative, got {num_samples}.")

    return float(num_samples / sample_rate)


def align_signal_lengths(
    sig1: np.ndarray, sig2: np.ndarray, mode: str = "truncate"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure two discrete signals have identical sample lengths for objective mathematical comparison.

    Parameters
    ----------
    sig1 : np.ndarray
        First audio signal array.
    sig2 : np.ndarray
        Second audio signal array.
    mode : str, optional
        Alignment strategy:
        - 'truncate' (default): Truncate the longer signal to match the shorter signal.
        - 'pad': Zero-pad the shorter signal to match the longer signal.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two signals having identical length along axis 0.

    Raises
    ------
    ValueError
        If mode is invalid or arrays have incompatible dimensions.
    """
    if not isinstance(sig1, np.ndarray) or not isinstance(sig2, np.ndarray):
        raise TypeError("Both signals must be NumPy arrays.")

    if sig1.ndim != sig2.ndim:
        raise ValueError(f"Signals have mismatched dimensions: {sig1.ndim}D vs {sig2.ndim}D.")

    len1 = sig1.shape[0]
    len2 = sig2.shape[0]

    if len1 == len2:
        return sig1.copy(), sig2.copy()

    if mode == "truncate":
        min_len = min(len1, len2)
        return sig1[:min_len].copy(), sig2[:min_len].copy()
    elif mode == "pad":
        max_len = max(len1, len2)
        if sig1.ndim == 1:
            pad1 = np.zeros(max_len, dtype=sig1.dtype)
            pad2 = np.zeros(max_len, dtype=sig2.dtype)
            pad1[:len1] = sig1
            pad2[:len2] = sig2
            return pad1, pad2
        else:
            pad1 = np.zeros((max_len, sig1.shape[1]), dtype=sig1.dtype)
            pad2 = np.zeros((max_len, sig2.shape[1]), dtype=sig2.dtype)
            pad1[:len1, :] = sig1
            pad2[:len2, :] = sig2
            return pad1, pad2
    else:
        raise ValueError(f"Unknown alignment mode: '{mode}'. Expected 'truncate' or 'pad'.")
