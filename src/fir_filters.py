"""
AudioClean DSP - FIR Digital Filter Engine
Implements Type-I symmetric linear-phase Finite Impulse Response (FIR) digital filters
using the window method (scipy.signal.firwin) with causal and zero-phase filtering.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple, Union
import numpy as np
from scipy.signal import filtfilt, firwin, freqz, lfilter

from src.preprocessing import validate_audio


@dataclass(frozen=True)
class FIRFilterDesign:
    """
    Structured container holding digital FIR filter design specifications and coefficients.

    Attributes
    ----------
    coefficients : np.ndarray
        Discrete-time impulse response / filter tap coefficients b[k], dtype float64.
    sample_rate : int
        Sampling frequency in Hertz (Hz).
    filter_type : str
        Filter topology ('lowpass', 'highpass', 'bandpass', 'bandstop').
    cutoff_hz : Union[float, Tuple[float, float]]
        Effective cutoff frequency or band edges in Hertz.
    window : str
        Window function employed ('hamming', 'hann', 'blackman').
    numtaps : int
        Number of filter impulse response coefficients (taps).
    order : int
        Filter order N = numtaps - 1.
    group_delay_samples : float
        Theoretical linear phase group delay tau_g = (numtaps - 1) / 2 samples.
    group_delay_seconds : float
        Group delay converted to physical time in seconds tau_g / f_s.
    """

    coefficients: np.ndarray
    sample_rate: int
    filter_type: str
    cutoff_hz: Union[float, Tuple[float, float]]
    window: str
    numtaps: int
    order: int
    group_delay_samples: float
    group_delay_seconds: float


SUPPORTED_WINDOWS = ("hamming", "hann", "blackman")
FILTER_TYPE_MAP = {
    "lowpass": "lowpass",
    "low_pass": "lowpass",
    "lp": "lowpass",
    "highpass": "highpass",
    "high_pass": "highpass",
    "hp": "highpass",
    "bandpass": "bandpass",
    "band_pass": "bandpass",
    "bp": "bandpass",
    "bandstop": "bandstop",
    "band_stop": "bandstop",
    "bs": "bandstop",
    "notch": "bandstop",
}


def design_fir_filter(
    sample_rate: int,
    filter_type: str,
    cutoff_hz: Union[float, Sequence[float]],
    numtaps: int,
    window: str = "hamming",
) -> FIRFilterDesign:
    """
    Design a linear-phase Type-I Finite Impulse Response (FIR) digital filter using the window method.

    Parameters
    ----------
    sample_rate : int
        Sampling frequency in Hertz (Hz). Must be strictly positive.
    filter_type : str
        Type of filter: 'lowpass', 'highpass', 'bandpass', or 'bandstop'.
    cutoff_hz : Union[float, Sequence[float]]
        Cutoff frequency in Hz.
        - Single float for 'lowpass' and 'highpass'.
        - Two-element sequence [f_low, f_high] for 'bandpass' and 'bandstop'.
        Note: In window-based FIR filter design (scipy.signal.firwin), the cutoff
        frequency specifies the -6.02 dB (half-amplitude, |H(f)| = 0.5) transition
        midpoint of the ideal sinc response, in contrast to IIR Butterworth filters
        where cutoff specifies the -3.01 dB (half-power, |H(f)| = 1/sqrt(2)) point.
    numtaps : int
        Number of filter coefficients (taps). Must be an ODD integer >= 3.
        Note: Filter order N = numtaps - 1.
    window : str, optional
        Windowing function: 'hamming' (default), 'hann', or 'blackman'.

    Returns
    -------
    FIRFilterDesign
        Immutable dataclass encapsulating the filter coefficients and metadata.

    Raises
    ------
    TypeError
        If input types are invalid.
    ValueError
        If parameters violate physical or numerical constraints (e.g., even taps,
        cutoffs at or above Nyquist, inverted band edges, unsupported filter/window).
    """
    # 1. Validate Sample Rate
    if not isinstance(sample_rate, (int, np.integer)):
        raise TypeError(f"sample_rate must be an integer, got {type(sample_rate).__name__}.")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be strictly positive, got {sample_rate} Hz.")

    nyquist = sample_rate / 2.0

    # 2. Validate Filter Type
    if not isinstance(filter_type, str):
        raise TypeError(f"filter_type must be a string, got {type(filter_type).__name__}.")
    clean_type = filter_type.strip().lower()
    if clean_type not in FILTER_TYPE_MAP:
        raise ValueError(
            f"Unsupported filter_type: '{filter_type}'. "
            f"Supported types are: 'lowpass', 'highpass', 'bandpass', 'bandstop'."
        )
    canonical_type = FILTER_TYPE_MAP[clean_type]

    # 3. Validate Window
    if not isinstance(window, str):
        raise TypeError(f"window must be a string, got {type(window).__name__}.")
    clean_window = window.strip().lower()
    if clean_window not in SUPPORTED_WINDOWS:
        raise ValueError(
            f"Unsupported window: '{window}'. Supported windows are: {SUPPORTED_WINDOWS}."
        )

    # 4. Validate Tap Count (numtaps)
    if not isinstance(numtaps, (int, np.integer)):
        raise TypeError(f"numtaps must be an integer, got {type(numtaps).__name__}.")
    if numtaps < 3:
        raise ValueError(f"numtaps must be at least 3, got {numtaps}.")
    if numtaps % 2 == 0:
        raise ValueError(
            f"numtaps must be an ODD integer to ensure a symmetric Type-I linear-phase FIR filter "
            f"with constant integer group delay. Received even numtaps={numtaps}. "
            f"Filter order N = numtaps - 1."
        )

    # 5. Validate Cutoff Frequencies
    validated_cutoff: Union[float, Tuple[float, float]]
    if canonical_type in ("lowpass", "highpass"):
        if isinstance(cutoff_hz, (list, tuple, np.ndarray)):
            if len(cutoff_hz) != 1:
                raise ValueError(
                    f"filter_type '{canonical_type}' requires exactly one cutoff frequency, "
                    f"got {len(cutoff_hz)} elements."
                )
            fc = float(cutoff_hz[0])
        elif isinstance(cutoff_hz, (int, float, np.floating, np.integer)):
            fc = float(cutoff_hz)
        else:
            raise TypeError(f"Invalid cutoff_hz type: {type(cutoff_hz).__name__}.")

        if not np.isfinite(fc):
            raise ValueError(f"cutoff_hz must be finite, got {fc}.")
        if fc <= 0.0:
            raise ValueError(f"cutoff_hz must be strictly positive (> 0 Hz), got {fc} Hz.")
        if fc >= nyquist:
            raise ValueError(
                f"cutoff_hz ({fc} Hz) must be strictly below the Nyquist frequency "
                f"({nyquist} Hz for sample_rate={sample_rate} Hz)."
            )
        validated_cutoff = fc
        firwin_cutoff = fc

    else:  # bandpass, bandstop
        if not isinstance(cutoff_hz, (list, tuple, np.ndarray)):
            raise TypeError(
                f"filter_type '{canonical_type}' requires a sequence of two cutoff frequencies "
                f"[f_low, f_high], got {type(cutoff_hz).__name__}."
            )
        if len(cutoff_hz) != 2:
            raise ValueError(
                f"filter_type '{canonical_type}' requires exactly two cutoff frequencies, "
                f"got {len(cutoff_hz)} elements: {cutoff_hz}."
            )

        f_low = float(cutoff_hz[0])
        f_high = float(cutoff_hz[1])

        if not (np.isfinite(f_low) and np.isfinite(f_high)):
            raise ValueError(f"Band edges must be finite, got [{f_low}, {f_high}].")
        if f_low <= 0.0:
            raise ValueError(f"Lower cutoff frequency f_low must be > 0 Hz, got {f_low} Hz.")
        if f_high >= nyquist:
            raise ValueError(
                f"Upper cutoff frequency f_high ({f_high} Hz) must be strictly below "
                f"the Nyquist frequency ({nyquist} Hz)."
            )
        if f_low >= f_high:
            raise ValueError(
                f"Lower cutoff f_low ({f_low} Hz) must be strictly less than upper cutoff "
                f"f_high ({f_high} Hz)."
            )

        validated_cutoff = (f_low, f_high)
        firwin_cutoff = [f_low, f_high]

    # 6. Synthesize FIR Coefficients using Window Method
    pass_zero_map = {
        "lowpass": "lowpass",
        "highpass": "highpass",
        "bandpass": "bandpass",
        "bandstop": "bandstop",
    }
    pass_zero_val = pass_zero_map[canonical_type]

    try:
        coefficients = firwin(
            numtaps=numtaps,
            cutoff=firwin_cutoff,
            window=clean_window,
            pass_zero=pass_zero_val,
            fs=sample_rate,
        ).astype(np.float64)
    except Exception as err:
        raise RuntimeError(f"scipy.signal.firwin failed to compute coefficients: {err}") from err

    # 7. Compute Structural Metrics
    order = numtaps - 1
    group_delay_samples = float(order / 2.0)
    group_delay_seconds = float(group_delay_samples / sample_rate)

    return FIRFilterDesign(
        coefficients=coefficients,
        sample_rate=int(sample_rate),
        filter_type=canonical_type,
        cutoff_hz=validated_cutoff,
        window=clean_window,
        numtaps=int(numtaps),
        order=order,
        group_delay_samples=group_delay_samples,
        group_delay_seconds=group_delay_seconds,
    )


def apply_fir_filter(
    audio: np.ndarray,
    filter_design: FIRFilterDesign,
    mode: str = "zero_phase",
) -> np.ndarray:
    """
    Apply a designed FIR filter to single-channel (mono) or multi-channel (stereo) audio.

    Parameters
    ----------
    audio : np.ndarray
        Discrete audio signal array of shape (N,) for mono or (N, C) for stereo.
    filter_design : FIRFilterDesign
        Filter design object containing coefficients and sampling rate.
    mode : str, optional
        Filtering execution mode:
        - 'zero_phase' (default): Noncausal forward-backward filtering via scipy.signal.filtfilt.
          Eliminates phase distortion and produces zero group delay (ideal for offline analysis).
        - 'causal': Real-time direct-form discrete convolution via scipy.signal.lfilter.
          Introduces a constant linear-phase group delay of (numtaps - 1) / 2 samples.

    Returns
    -------
    np.ndarray
        Filtered audio array matching the input shape and dtype float64.

    Raises
    ------
    TypeError
        If audio or filter_design are of incorrect types.
    ValueError
        If audio is invalid, empty, or too short for zero-phase reflection padding,
        or if mode is unrecognized.
    """
    if not isinstance(filter_design, FIRFilterDesign):
        raise TypeError(
            f"filter_design must be an instance of FIRFilterDesign, got {type(filter_design).__name__}."
        )

    # Validate audio array
    validate_audio(audio, filter_design.sample_rate)

    clean_mode = mode.strip().lower()
    if clean_mode not in ("zero_phase", "causal"):
        raise ValueError(
            f"Unsupported filtering mode: '{mode}'. Must be 'zero_phase' or 'causal'."
        )

    data = audio.astype(np.float64, copy=True)
    b = filter_design.coefficients
    a = 1.0  # FIR transfer function denominator polynomial is unity

    if clean_mode == "causal":
        # Causal discrete convolution: y[n] = sum_{k=0}^{M-1} b[k] * x[n-k]
        filtered = lfilter(b, a, data, axis=0)
        return filtered.astype(np.float64)

    else:  # zero_phase
        # Minimum length required for scipy.signal.filtfilt default reflection padding (padlen = 3 * numtaps)
        padlen = 3 * filter_design.numtaps
        num_samples = data.shape[0]

        if num_samples <= padlen:
            raise ValueError(
                f"Input audio length ({num_samples} samples) is too short for zero-phase "
                f"filtering with {filter_design.numtaps} taps. filtfilt reflection padding "
                f"requires signal length > 3 * numtaps ({padlen} samples). "
                f"Please provide a longer signal or switch to mode='causal'."
            )

        # Forward-backward noncausal filtering: zero phase shift, squared magnitude response
        filtered = filtfilt(b, a, data, axis=0)
        return filtered.astype(np.float64)


def get_fir_frequency_response(
    filter_design: FIRFilterDesign,
    worN: Union[int, Sequence[float], np.ndarray] = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the complex discrete-time frequency response H(e^{j omega}) of the FIR filter.

    Parameters
    ----------
    filter_design : FIRFilterDesign
        Filter design object.
    worN : Union[int, Sequence[float], np.ndarray], optional
        Number of discrete frequency points evaluated between 0 and Nyquist (if int),
        or explicit array/sequence of frequencies in Hertz. Default is 2048.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        frequencies_hz : np.ndarray
            Frequency axis in physical Hertz (from 0 to f_s / 2).
        magnitude : np.ndarray
            Linear magnitude response |H(f)|.
        magnitude_db : np.ndarray
            Logarithmic magnitude response 20 * log10(|H(f)|) in decibels (floored at -300 dB).
        phase_radians : np.ndarray
            Unwrapped continuous phase response in radians.

    Raises
    ------
    ValueError
        If worN <= 0 or frequency array contains invalid entries.
    """
    nyquist = filter_design.sample_rate / 2.0
    if isinstance(worN, (int, np.integer)):
        if worN <= 0:
            raise ValueError(f"worN must be a strictly positive integer, got {worN}.")
        w, h = freqz(
            b=filter_design.coefficients,
            a=1.0,
            worN=int(worN),
            include_nyquist=True,
            fs=filter_design.sample_rate,
        )
    else:
        freq_arr = np.asarray(worN, dtype=np.float64)
        if len(freq_arr) == 0:
            raise ValueError("Frequency array worN cannot be empty.")
        if np.any(freq_arr < 0.0) or np.any(freq_arr > nyquist):
            raise ValueError(f"All frequencies in worN must lie within [0, Nyquist ({nyquist} Hz)].")
        w, h = freqz(
            b=filter_design.coefficients,
            a=1.0,
            worN=freq_arr,
            fs=filter_design.sample_rate,
        )

    frequencies_hz = w.astype(np.float64)
    magnitude = np.abs(h).astype(np.float64)

    # Floor magnitude at 1e-15 to avoid log(0) numerical warnings
    magnitude_db = (20.0 * np.log10(np.maximum(magnitude, 1e-15))).astype(np.float64)
    phase_radians = np.unwrap(np.angle(h)).astype(np.float64)

    return frequencies_hz, magnitude, magnitude_db, phase_radians
