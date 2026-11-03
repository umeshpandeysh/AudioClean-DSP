"""
AudioClean DSP - IIR Digital Filter Engine
Implements Infinite Impulse Response (IIR) digital filters (Butterworth lowpass,
highpass, bandpass, bandstop, and configurable notch filters) using numerically
stable Second-Order Sections (SOS) representations.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfiltfilt, sosfreqz, tf2sos

from src.preprocessing import validate_audio


@dataclass(frozen=True)
class IIRFilterDesign:
    """
    Structured immutable container holding IIR digital filter specifications and SOS coefficients.

    Attributes
    ----------
    sos : np.ndarray
        Second-Order Sections coefficient matrix of shape (n_sections, 6), dtype float64.
        Each row represents: [b0, b1, b2, a0, a1, a2].
    sample_rate : int
        Sampling frequency in Hertz (Hz).
    filter_type : str
        Filter topology: 'lowpass', 'highpass', 'bandpass', 'bandstop', or 'notch'.
    cutoff_hz : Optional[Union[float, Tuple[float, float]]]
        Effective cutoff frequency or band edges in Hertz. None for notch filters.
    order : int
        Resulting filter order (total number of poles in discrete z-domain).
        Equal to prototype order N for lowpass/highpass, 2N for bandpass/bandstop, and 2 for notch.
    notch_freq : Optional[float]
        Center rejection frequency in Hz (only for notch filters).
    quality_factor : Optional[float]
        Quality factor Q = f0 / delta_f (only for notch filters).
    design_method : str
        Underlying design methodology: 'butterworth' or 'iirnotch'.
    prototype_order : Optional[int]
        Analog lowpass prototype order N specified during filter design.
    resulting_order : Optional[int]
        Effective digital filter order (total poles). Equal to order.
    """

    sos: np.ndarray
    sample_rate: int
    filter_type: str
    cutoff_hz: Optional[Union[float, Tuple[float, float]]]
    order: int
    notch_freq: Optional[float]
    quality_factor: Optional[float]
    design_method: str
    prototype_order: Optional[int] = None
    resulting_order: Optional[int] = None


def check_sos_stability(sos: np.ndarray, tolerance: float = 1e-7) -> Tuple[bool, float]:
    """
    Verify whether an IIR filter represented in Second-Order Sections (SOS) is BIBO stable.

    A discrete-time linear time-invariant filter is Bounded-Input Bounded-Output (BIBO) stable
    if and only if all poles of its transfer function H(z) lie strictly inside the unit circle
    of the complex z-plane:
        |p_k| < 1.0  for all k = 1, ..., P

    Each row of the SOS matrix has the form: [b0, b1, b2, a0, a1, a2], corresponding to:
        H_k(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
               = (b0*z^2 + b1*z + b2) / (a0*z^2 + a1*z + a2)

    The section poles are the roots of the quadratic polynomial a0*z^2 + a1*z + a2 = 0.

    Parameters
    ----------
    sos : np.ndarray
        Second-Order Sections matrix of shape (n_sections, 6).
    tolerance : float, optional
        Margin below 1.0 for strict interior stability check (default 1e-7).

    Returns
    -------
    Tuple[bool, float]
        (is_stable, max_pole_magnitude)
        - is_stable: True if all pole magnitudes are strictly less than (1.0 - tolerance).
        - max_pole_magnitude: Maximum pole radius max |p_k| across all sections.
    """
    if not isinstance(sos, np.ndarray) or sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError(f"SOS matrix must have shape (n_sections, 6), got {getattr(sos, 'shape', None)}.")

    all_poles = []
    for section_idx in range(sos.shape[0]):
        a = sos[section_idx, 3:6]
        if a[0] == 0.0:
            raise ValueError(f"Invalid SOS section {section_idx}: leading denominator coefficient a0 is zero.")
        poles = np.roots(a)
        all_poles.extend(poles)

    if len(all_poles) == 0:
        return True, 0.0

    pole_mags = np.abs(all_poles)
    max_pole_mag = float(np.max(pole_mags))
    is_stable = bool(max_pole_mag < (1.0 - tolerance))

    return is_stable, max_pole_mag


IIR_TYPE_MAP = {
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
}


def design_iir_filter(
    sample_rate: int,
    filter_type: str,
    cutoff_hz: Union[float, Sequence[float]],
    order: int = 4,
) -> IIRFilterDesign:
    """
    Design an analog-prototype Butterworth Infinite Impulse Response (IIR) digital filter
    in Second-Order Sections (SOS) format.

    Butterworth filters provide a maximally flat magnitude response in the passband
    with no equiripple behavior.

    Parameters
    ----------
    sample_rate : int
        Sampling frequency in Hertz (Hz). Must be strictly positive.
    filter_type : str
        Topology: 'lowpass', 'highpass', 'bandpass', or 'bandstop'.
    cutoff_hz : Union[float, Sequence[float]]
        Cutoff frequency in Hz (-3 dB half-power point).
        - Single float for 'lowpass' and 'highpass'.
        - Two-element sequence [f_low, f_high] for 'bandpass' and 'bandstop'.
    order : int, optional
        Filter order N, default is 4. Must be an integer >= 1.

    Returns
    -------
    IIRFilterDesign
        Immutable dataclass holding SOS matrix and filter metadata.

    Raises
    ------
    TypeError
        If parameter types are invalid.
    ValueError
        If cutoff frequencies violate Nyquist, order is invalid, or band edges are inverted.
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
    if clean_type not in IIR_TYPE_MAP:
        raise ValueError(
            f"Unsupported filter_type: '{filter_type}'. "
            f"Supported types are: 'lowpass', 'highpass', 'bandpass', 'bandstop'."
        )
    canonical_type = IIR_TYPE_MAP[clean_type]

    # 3. Validate Filter Order
    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"order must be an integer, got {type(order).__name__}.")
    if order < 1:
        raise ValueError(f"order must be an integer >= 1, got {order}.")

    # 4. Validate Cutoff Frequencies
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
                f"cutoff_hz ({fc} Hz) must be strictly below Nyquist ({nyquist} Hz for fs={sample_rate} Hz)."
            )
        validated_cutoff = fc
        butter_cutoff: Union[float, Sequence[float]] = fc

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
            raise ValueError(f"Lower cutoff f_low must be > 0 Hz, got {f_low} Hz.")
        if f_high >= nyquist:
            raise ValueError(
                f"Upper cutoff f_high ({f_high} Hz) must be strictly below Nyquist ({nyquist} Hz)."
            )
        if f_low >= f_high:
            raise ValueError(
                f"Lower cutoff f_low ({f_low} Hz) must be strictly less than upper cutoff f_high ({f_high} Hz)."
            )

        validated_cutoff = (f_low, f_high)
        butter_cutoff = [f_low, f_high]

    # 5. Compute Butterworth Second-Order Sections (SOS)
    try:
        sos = butter(
            N=order,
            Wn=butter_cutoff,
            btype=canonical_type,
            output="sos",
            fs=sample_rate,
        ).astype(np.float64)
    except Exception as err:
        raise RuntimeError(f"scipy.signal.butter failed: {err}") from err

    # 6. Verify BIBO Pole Stability
    is_stable, max_pole = check_sos_stability(sos)
    if not is_stable:
        raise ValueError(
            f"Designed Butterworth {canonical_type} filter is unstable: "
            f"maximum pole magnitude {max_pole:.6f} >= 1.0."
        )

    # Effective digital filter order: N for lowpass/highpass, 2N for bandpass/bandstop
    eff_order = 2 * int(order) if canonical_type in ("bandpass", "bandstop") else int(order)

    return IIRFilterDesign(
        sos=sos,
        sample_rate=int(sample_rate),
        filter_type=canonical_type,
        cutoff_hz=validated_cutoff,
        order=eff_order,
        notch_freq=None,
        quality_factor=None,
        design_method="butterworth",
        prototype_order=int(order),
        resulting_order=eff_order,
    )


def design_iir_notch(
    sample_rate: int,
    notch_freq: float = 50.0,
    quality_factor: float = 30.0,
) -> IIRFilterDesign:
    """
    Design a narrow Second-Order Sections IIR Notch (band-reject) filter to eliminate
    single-frequency electrical or tonal interference (e.g., 50 Hz power-line hum).

    The notch filter transfer function features complex-conjugate zeros on the unit circle
    at the notch frequency and corresponding poles placed slightly inside the unit circle:
        H(z) = b0 * (1 - 2*cos(w0)*z^-1 + z^-2) / (1 - 2*r*cos(w0)*z^-1 + r^2*z^-2)

    Where Quality Factor Q = f0 / delta_f (-3 dB rejection bandwidth).

    Parameters
    ----------
    sample_rate : int
        Sampling frequency in Hertz (Hz). Must be strictly positive.
    notch_freq : float, optional
        Center frequency to reject in Hz, default is 50.0 Hz.
    quality_factor : float, optional
        Quality factor Q = f0 / delta_f, default is 30.0. Higher Q yields a narrower notch.

    Returns
    -------
    IIRFilterDesign
        Immutable dataclass holding SOS matrix and notch metadata.

    Raises
    ------
    TypeError
        If parameter types are invalid.
    ValueError
        If notch_freq <= 0, notch_freq >= Nyquist, or quality_factor <= 0.
    """
    if not isinstance(sample_rate, (int, np.integer)):
        raise TypeError(f"sample_rate must be an integer, got {type(sample_rate).__name__}.")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be strictly positive, got {sample_rate} Hz.")

    if not isinstance(notch_freq, (int, float, np.floating, np.integer)):
        raise TypeError(f"notch_freq must be numeric, got {type(notch_freq).__name__}.")
    f0 = float(notch_freq)
    if not np.isfinite(f0):
        raise ValueError(f"notch_freq must be finite, got {f0}.")

    nyquist = sample_rate / 2.0
    if f0 <= 0.0 or f0 >= nyquist:
        raise ValueError(
            f"notch_freq ({f0} Hz) must satisfy 0 < notch_freq < Nyquist ({nyquist} Hz)."
        )

    if not isinstance(quality_factor, (int, float, np.floating, np.integer)):
        raise TypeError(f"quality_factor must be numeric, got {type(quality_factor).__name__}.")
    q = float(quality_factor)
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError(f"quality_factor must be strictly positive (> 0), got {q}.")

    # Design analog-matched notch filter using scipy.signal.iirnotch
    try:
        b, a = iirnotch(w0=f0, Q=q, fs=sample_rate)
        # Convert transfer function [b, a] to Second-Order Sections (SOS) for uniform processing
        sos = tf2sos(b, a).astype(np.float64)
    except Exception as err:
        raise RuntimeError(f"scipy.signal.iirnotch failed: {err}") from err

    # Verify BIBO Pole Stability
    is_stable, max_pole = check_sos_stability(sos)
    if not is_stable:
        raise ValueError(
            f"Designed notch filter is unstable: maximum pole magnitude {max_pole:.6f} >= 1.0."
        )

    return IIRFilterDesign(
        sos=sos,
        sample_rate=int(sample_rate),
        filter_type="notch",
        cutoff_hz=None,
        order=2,  # A 2nd-order notch filter has 2 poles and 2 zeros
        notch_freq=f0,
        quality_factor=q,
        design_method="iirnotch",
        prototype_order=2,
        resulting_order=2,
    )


def apply_iir_filter(
    audio: np.ndarray,
    filter_design: IIRFilterDesign,
    mode: str = "zero_phase",
) -> np.ndarray:
    """
    Apply a designed IIR filter to single-channel (mono) or multi-channel (stereo) audio
    using Second-Order Sections (SOS) implementations.

    Parameters
    ----------
    audio : np.ndarray
        Discrete audio signal array, shape (N,) for mono or (N, C) for stereo.
    filter_design : IIRFilterDesign
        IIR filter design object containing SOS matrix.
    mode : str, optional
        Filtering execution mode:
        - 'zero_phase' (default): Noncausal forward-backward filtering via scipy.signal.sosfiltfilt.
          Completely cancels nonlinear phase distortion and introduces zero group delay.
        - 'causal': Real-time recursive filtering via scipy.signal.sosfilt.
          Introduces frequency-dependent group delay and phase lag.

    Returns
    -------
    np.ndarray
        Filtered audio array matching input shape and dtype float64.

    Raises
    ------
    TypeError
        If audio or filter_design are invalid types.
    ValueError
        If audio violates validation constraints, is too short for zero-phase padding,
        or mode is unrecognized.
    """
    if not isinstance(filter_design, IIRFilterDesign):
        raise TypeError(
            f"filter_design must be an instance of IIRFilterDesign, got {type(filter_design).__name__}."
        )

    # Validate audio array integrity
    validate_audio(audio, filter_design.sample_rate)

    clean_mode = mode.strip().lower()
    if clean_mode not in ("zero_phase", "causal"):
        raise ValueError(
            f"Unsupported filtering mode: '{mode}'. Must be 'zero_phase' or 'causal'."
        )

    data = audio.astype(np.float64, copy=True)
    sos = filter_design.sos

    if clean_mode == "causal":
        # Recursive Second-Order Sections filtering: y[n] = sosfilt(sos, x[n])
        filtered = sosfilt(sos, data, axis=0)
        return filtered.astype(np.float64)

    else:  # zero_phase
        # Match exact default padlen calculation used by scipy.signal.sosfiltfilt:
        # ntaps = 2 * n_sections + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
        # padlen = 3 * ntaps
        # This properly accounts for first-order SOS sections where b2 == 0 or a2 == 0 (e.g. odd-order filters).
        n_sections = sos.shape[0]
        ntaps = 2 * n_sections + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
        padlen = 3 * ntaps
        num_samples = data.shape[0]

        if num_samples <= padlen:
            raise ValueError(
                f"Input audio length ({num_samples} samples) is too short for zero-phase "
                f"IIR filtering. sosfiltfilt reflection padding requires signal length > {padlen} samples. "
                f"Please provide a longer signal or switch to mode='causal'."
            )

        # Forward-backward noncausal filtering
        filtered = sosfiltfilt(sos, data, axis=0)
        return filtered.astype(np.float64)


def get_iir_frequency_response(
    filter_design: IIRFilterDesign,
    worN: Union[int, Sequence[float], np.ndarray] = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the complex discrete-time frequency response H(e^{j omega}) of the IIR SOS filter.

    Parameters
    ----------
    filter_design : IIRFilterDesign
        IIR filter design object.
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
        If worN <= 0.
    """
    nyquist = filter_design.sample_rate / 2.0

    if isinstance(worN, (int, np.integer)):
        if worN <= 0:
            raise ValueError(f"worN must be a strictly positive integer, got {worN}.")
        freq_axis = np.linspace(0.0, nyquist, int(worN), endpoint=True, dtype=np.float64)
    else:
        freq_axis = np.asarray(worN, dtype=np.float64)
        if len(freq_axis) == 0:
            raise ValueError("Frequency array worN cannot be empty.")
        if np.any(freq_axis < 0.0) or np.any(freq_axis > nyquist):
            raise ValueError(f"All frequencies in worN must lie within [0, Nyquist ({nyquist} Hz)].")

    w, h = sosfreqz(
        sos=filter_design.sos,
        worN=freq_axis,
        fs=filter_design.sample_rate,
    )

    frequencies_hz = w.astype(np.float64)
    magnitude = np.abs(h).astype(np.float64)

    # Floor magnitude at 1e-15 (-300 dB) to prevent log10(0) numerical warnings
    magnitude_db = (20.0 * np.log10(np.maximum(magnitude, 1e-15))).astype(np.float64)
    phase_radians = np.unwrap(np.angle(h)).astype(np.float64)

    return frequencies_hz, magnitude, magnitude_db, phase_radians
