"""
AudioClean DSP - Audio Input and Output Module
Handles robust reading, writing, and inspection of WAV audio files.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union
import numpy as np
import soundfile as sf


def load_audio(file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """
    Load an audio file from disk into a floating-point NumPy array.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the WAV audio file.

    Returns
    -------
    Tuple[np.ndarray, int]
        data : np.ndarray
            Audio samples represented as float64, shape (N,) for mono
            or (N, C) for multi-channel.
        sample_rate : int
            Sampling frequency in Hertz (Hz).

    Raises
    ------
    FileNotFoundError
        If the audio file does not exist.
    ValueError
        If the audio file is empty or contains zero valid samples.
    RuntimeError
        If SoundFile cannot read or decode the file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path.resolve()}")

    try:
        # Read as float64 to maintain high numerical precision for DSP algorithms
        data, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
    except sf.LibsndfileError as err:
        raise RuntimeError(f"Libsndfile failed to decode audio file {path.name}: {err}") from err
    except Exception as err:
        raise RuntimeError(f"Unexpected error while reading {path.name}: {err}") from err

    if data.size == 0 or len(data) == 0:
        raise ValueError(f"Audio file contains zero samples: {path.name}")

    return data, int(sample_rate)


def save_audio(
    file_path: Union[str, Path],
    data: np.ndarray,
    sample_rate: int,
    subtype: str = "PCM_16",
    allow_clipping: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Save floating-point audio data to a WAV file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Target destination path for the WAV file.
    data : np.ndarray
        Floating-point audio data array.
    sample_rate : int
        Sampling frequency in Hertz (Hz).
    subtype : str, optional
        WAV format subtype ('PCM_16', 'PCM_24', 'FLOAT', etc.), default is 'PCM_16'.
    allow_clipping : bool, optional
        If False, values exceeding [-1.0, 1.0] raise a ValueError.
        If True, samples are hard-clipped to [-1.0, 1.0] before writing to integer formats.
    overwrite : bool, optional
        If False and file exists, raises FileExistsError to preserve existing recordings.

    Raises
    ------
    FileExistsError
        If destination file already exists and overwrite is False.
    ValueError
        If data is invalid, empty, or unallowed clipping occurs.
    """
    path = Path(file_path)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Target file already exists: {path.resolve()}. Set overwrite=True to replace."
        )

    if not isinstance(data, np.ndarray):
        raise TypeError("Audio data must be a NumPy array.")

    if data.size == 0:
        raise ValueError("Cannot save empty audio array.")

    if not np.all(np.isfinite(data)):
        raise ValueError("Audio data contains NaN or infinite values.")

    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate} Hz.")

    # Check for amplitude clipping
    max_amp = float(np.max(np.abs(data)))
    processed_data = data.copy()

    if max_amp > 1.0:
        if not allow_clipping:
            raise ValueError(
                f"Audio data peak amplitude ({max_amp:.4f}) exceeds unity [-1.0, 1.0]. "
                "Normalize audio or set allow_clipping=True."
            )
        # Saturate within safe bounds
        processed_data = np.clip(processed_data, -1.0, 1.0)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        sf.write(str(path), processed_data, samplerate=sample_rate, subtype=subtype)
    except Exception as err:
        raise RuntimeError(f"Failed to write audio file {path.name}: {err}") from err


def get_audio_info(data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Extract key time-domain characteristics and structural metadata from an audio signal.

    Parameters
    ----------
    data : np.ndarray
        Audio signal array.
    sample_rate : int
        Sampling frequency in Hertz (Hz).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'num_samples': Total discrete time samples per channel.
        - 'sample_rate': Sampling frequency in Hz.
        - 'duration_seconds': Total signal duration in seconds.
        - 'num_channels': Number of channels (1 = mono, 2 = stereo).
        - 'peak_amplitude': Maximum absolute amplitude.
        - 'rms_amplitude': Root Mean Square amplitude.
        - 'crest_factor_db': Ratio of peak to RMS in decibels (dynamic range metric).

    Raises
    ------
    ValueError
        If data is empty, non-finite, or sample_rate <= 0.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Audio data must be a NumPy array.")

    if data.size == 0:
        raise ValueError("Cannot extract metadata from an empty audio array.")

    if not np.all(np.isfinite(data)):
        raise ValueError("Audio data contains NaN or infinite values.")

    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate} Hz.")

    num_samples = data.shape[0]
    num_channels = 1 if data.ndim == 1 else data.shape[1]
    duration_seconds = float(num_samples / sample_rate)

    peak_amplitude = float(np.max(np.abs(data)))
    rms_amplitude = float(np.sqrt(np.mean(data**2)))

    # Compute crest factor: 20 * log10(peak / RMS)
    if rms_amplitude > 0.0 and peak_amplitude > 0.0:
        crest_factor_db = float(20.0 * np.log10(peak_amplitude / rms_amplitude))
    else:
        crest_factor_db = 0.0

    return {
        "num_samples": num_samples,
        "sample_rate": sample_rate,
        "duration_seconds": duration_seconds,
        "num_channels": num_channels,
        "peak_amplitude": peak_amplitude,
        "rms_amplitude": rms_amplitude,
        "crest_factor_db": crest_factor_db,
    }
