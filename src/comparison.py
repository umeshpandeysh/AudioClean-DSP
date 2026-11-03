"""
AudioClean DSP - FIR vs IIR Filter Comparison Engine
Provides scientifically fair, objective comparative performance evaluation between
Finite Impulse Response (FIR) and Infinite Impulse Response (IIR) digital filters
applied to identical discrete audio signals.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from src.fir_filters import FIRFilterDesign, apply_fir_filter, get_fir_frequency_response
from src.iir_filters import IIRFilterDesign, apply_iir_filter, get_iir_frequency_response
from src.metrics import (
    AudioEvaluationResult,
    calculate_mae,
    calculate_rmse,
    calculate_snr,
    evaluate_audio_quality,
    measure_processing_time,
)
from src.preprocessing import validate_audio


@dataclass(frozen=True)
class FilterFrequencyResponseData:
    """
    Frequency response evaluation data across a common frequency grid.

    Attributes
    ----------
    frequencies_hz : np.ndarray
        Frequency axis points in Hertz from 0 to Nyquist.
    fir_magnitude : np.ndarray
        FIR linear magnitude response |H_FIR(f)|.
    fir_magnitude_db : np.ndarray
        FIR magnitude response in decibels (20*log10(|H|)).
    fir_phase_radians : np.ndarray
        FIR continuous unwrapped phase response in radians.
    iir_magnitude : np.ndarray
        IIR linear magnitude response |H_IIR(f)|.
    iir_magnitude_db : np.ndarray
        IIR magnitude response in decibels (20*log10(|H|)).
    iir_phase_radians : np.ndarray
        IIR continuous unwrapped phase response in radians.
    """

    frequencies_hz: np.ndarray
    fir_magnitude: np.ndarray
    fir_magnitude_db: np.ndarray
    fir_phase_radians: np.ndarray
    iir_magnitude: np.ndarray
    iir_magnitude_db: np.ndarray
    iir_phase_radians: np.ndarray

    def as_dict(self) -> Dict[str, Any]:
        """Convert container to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class SingleFilterBenchmark:
    """
    Performance benchmark and design parameters for a single digital filter.

    Attributes
    ----------
    filter_family : str
        Filter category: 'FIR' or 'IIR'.
    filter_type : str
        Topology ('lowpass', 'highpass', 'bandpass', 'bandstop', 'notch').
    mode : str
        Execution mode ('zero_phase' or 'causal').
    order : int
        Filter order (N = taps - 1 for FIR; total pole count for IIR).
    num_taps_or_sections : int
        Number of taps b[k] for FIR, or number of Second-Order Sections (SOS) for IIR.
    group_delay_samples : Optional[float]
        Nominal group delay in samples (exact constant for Type-I FIR, None for nonlinear IIR).
    input_snr_db : float
        SNR of noisy input relative to clean reference in dB over the evaluated interval.
    output_snr_db : float
        SNR of filtered output relative to clean reference in dB over the evaluated interval.
    snr_improvement_db : float
        SNR improvement Delta SNR = output_snr_db - input_snr_db.
    input_rmse : float
        Root Mean Square Error of input signal against clean reference.
    output_rmse : float
        Root Mean Square Error of filtered output against clean reference.
    rmse_reduction_percent : float
        Percentage reduction in RMSE.
    input_mae : float
        Mean Absolute Error of input signal against clean reference.
    output_mae : float
        Mean Absolute Error of filtered output against clean reference.
    median_processing_time_ms : float
        Median execution time in milliseconds measured across repeated trials.
    timing_statistics : Dict[str, Any]
        Complete timing dictionary containing mean, std, min, max, trials, and warmups.
    """

    filter_family: str
    filter_type: str
    mode: str
    order: int
    num_taps_or_sections: int
    group_delay_samples: Optional[float]
    input_snr_db: float
    output_snr_db: float
    snr_improvement_db: float
    input_rmse: float
    output_rmse: float
    rmse_reduction_percent: float
    input_mae: float
    output_mae: float
    median_processing_time_ms: float
    timing_statistics: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        """Convert benchmark to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class FilterComparisonReport:
    """
    Comprehensive structured comparison report evaluating FIR vs IIR digital filters.

    Attributes
    ----------
    sample_rate : int
        Sampling frequency in Hertz (Hz).
    mode : str
        Execution mode ('zero_phase' or 'causal').
    evaluation_interval : Tuple[int, int]
        Sample index range [start_idx, end_idx) in reference signal coordinates over which objective
        metrics were evaluated: (0, N) in zero-phase mode, (tau, N - tau) in causal mode.
    evaluation_sample_count : int
        Number of valid evaluation samples: N in zero-phase mode, N - 2*tau in causal mode.
    fir_benchmark : SingleFilterBenchmark
        Benchmark results for the FIR filter.
    iir_benchmark : SingleFilterBenchmark
        Benchmark results for the IIR filter.
    snr_difference_db : float
        FIR SNR improvement minus IIR SNR improvement (positive favors FIR, negative favors IIR).
    speedup_ratio : Optional[float]
        FIR median latency / IIR median latency. None if either latency is zero/invalid.
    frequency_response : Optional[FilterFrequencyResponseData]
        Frequency response data across a common frequency grid.
    notes : str
        Scientific caveats regarding design differences, phase linearity, and cutoff definitions.
    """

    sample_rate: int
    mode: str
    evaluation_interval: Tuple[int, int]
    evaluation_sample_count: int
    fir_benchmark: SingleFilterBenchmark
    iir_benchmark: SingleFilterBenchmark
    snr_difference_db: float
    speedup_ratio: Optional[float]
    frequency_response: Optional[FilterFrequencyResponseData]
    notes: str

    def as_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary representation."""
        return asdict(self)


CUTOFF_TOLERANCE_HZ: float = 1e-3
CUTOFF_REL_TOL: float = 1e-5


def _is_close(
    val1: Union[int, float, np.floating],
    val2: Union[int, float, np.floating],
    atol: float = CUTOFF_TOLERANCE_HZ,
    rtol: float = CUTOFF_REL_TOL,
) -> bool:
    """Check whether two cutoff frequencies are equal within documented numerical tolerance."""
    return math.isclose(float(val1), float(val2), rel_tol=rtol, abs_tol=atol)


def align_causal_signal(
    signal: np.ndarray,
    delay_samples: int,
) -> np.ndarray:
    """
    Compensate for integer causal filter group delay by shifting backward and padding trailing edge.

    Parameters
    ----------
    signal : np.ndarray
        Delayed audio signal array (1D mono or 2D stereo).
    delay_samples : int
        Number of discrete samples to shift backward. Must be integer >= 0 and strictly less
        than signal length.

    Returns
    -------
    np.ndarray
        Time-aligned signal matching original shape and dtype.

    Raises
    ------
    TypeError
        If signal is not an ndarray, is complex-valued, non-numeric, or boolean;
        or if delay_samples is not an integer (boolean rejected).
    ValueError
        If signal is empty or has dimension other than 1D/2D, or if 2D array has > 2 channels;
        or if delay_samples < 0 or delay_samples >= signal length.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError(f"signal must be a numpy.ndarray, got {type(signal).__name__}.")

    if signal.dtype == bool or not np.issubdtype(signal.dtype, np.number) or np.iscomplexobj(signal):
        raise TypeError(f"signal must be a numeric floating-point or integer array, got {signal.dtype}.")

    if signal.ndim not in (1, 2):
        raise ValueError(f"signal must be a 1D (mono) or 2D (stereo) array, got {signal.ndim}D array.")

    if signal.ndim == 2 and signal.shape[1] not in (1, 2):
        raise ValueError(f"2D audio signal must have 1 or 2 channels, got shape {signal.shape}.")

    if signal.shape[0] == 0:
        raise ValueError("signal cannot be empty.")

    if isinstance(delay_samples, bool) or not isinstance(delay_samples, (int, np.integer)):
        raise TypeError(f"delay_samples must be an integer, got {type(delay_samples).__name__}.")

    if delay_samples < 0:
        raise ValueError(f"delay_samples must be non-negative (>= 0), got {delay_samples}.")

    n = signal.shape[0]
    if delay_samples >= n:
        raise ValueError(f"delay_samples ({delay_samples}) must be strictly less than signal length ({n}).")

    if delay_samples == 0:
        return signal.copy()

    aligned = np.zeros_like(signal)
    if signal.ndim == 1:
        aligned[: n - delay_samples] = signal[delay_samples:]
        aligned[n - delay_samples :] = 0.0
    else:
        aligned[: n - delay_samples, :] = signal[delay_samples:, :]
        aligned[n - delay_samples :, :] = 0.0

    return aligned


def _validate_filter_compatibility(
    fir_design: FIRFilterDesign,
    iir_design: IIRFilterDesign,
    sample_rate: Optional[int] = None,
) -> int:
    """
    Validate that FIR and IIR filter designs have compatible topologies, sampling rates,
    and actual cutoff specifications for a fair scientific comparison.

    Parameters
    ----------
    fir_design : FIRFilterDesign
        Designed FIR filter container.
    iir_design : IIRFilterDesign
        Designed IIR filter container.
    sample_rate : Optional[int]
        Optional expected sampling rate in Hz.

    Returns
    -------
    int
        Validated sampling frequency in Hz.

    Raises
    ------
    TypeError
        If designs are not instances of their respective dataclasses or sample_rate is invalid.
    ValueError
        If sampling rates mismatch, topologies are incompatible, or cutoff frequencies differ.
    """
    if not isinstance(fir_design, FIRFilterDesign):
        raise TypeError(f"fir_design must be an instance of FIRFilterDesign, got {type(fir_design).__name__}.")
    if not isinstance(iir_design, IIRFilterDesign):
        raise TypeError(f"iir_design must be an instance of IIRFilterDesign, got {type(iir_design).__name__}.")

    fs = sample_rate if sample_rate is not None else fir_design.sample_rate
    if not isinstance(fs, (int, np.integer)) or isinstance(fs, bool):
        raise TypeError(f"sample_rate must be an integer, got {type(fs).__name__}.")
    if fs <= 0:
        raise ValueError(f"sample_rate must be strictly positive, got {fs} Hz.")

    if fir_design.sample_rate != iir_design.sample_rate:
        raise ValueError(
            f"Sampling rate mismatch: FIR filter is designed for {fir_design.sample_rate} Hz, "
            f"while IIR filter is designed for {iir_design.sample_rate} Hz."
        )
    if fs != fir_design.sample_rate:
        raise ValueError(
            f"Specified sample_rate ({fs} Hz) does not match filter sample_rate ({fir_design.sample_rate} Hz)."
        )

    # Topology normalization: FIR 'bandstop' vs IIR 'notch' / 'bandstop'
    fir_type = fir_design.filter_type.lower()
    iir_type = iir_design.filter_type.lower()

    # Map aliases
    type_equiv = {
        "lowpass": "lowpass",
        "highpass": "highpass",
        "bandpass": "bandpass",
        "bandstop": "bandstop",
        "notch": "bandstop",
    }
    fir_norm = type_equiv.get(fir_type, fir_type)
    iir_norm = type_equiv.get(iir_type, iir_type)

    if fir_norm != iir_norm:
        raise ValueError(
            f"Incompatible filter topologies: cannot compare FIR '{fir_design.filter_type}' "
            f"with IIR '{iir_design.filter_type}'. Topologies must match for a valid comparison."
        )

    # Validate cutoff configuration compatibility
    if fir_norm in ("lowpass", "highpass"):
        if not isinstance(fir_design.cutoff_hz, (int, float, np.floating)) or isinstance(fir_design.cutoff_hz, bool):
            raise ValueError(f"FIR cutoff_hz must be a single scalar for {fir_norm}, got {fir_design.cutoff_hz}.")
        if (
            iir_design.cutoff_hz is None
            or not isinstance(iir_design.cutoff_hz, (int, float, np.floating))
            or isinstance(iir_design.cutoff_hz, bool)
        ):
            raise ValueError(f"IIR cutoff_hz must be a single scalar for {fir_norm}, got {iir_design.cutoff_hz}.")
        if not _is_close(fir_design.cutoff_hz, iir_design.cutoff_hz):
            raise ValueError(
                f"Cutoff frequency mismatch for {fir_norm}: FIR cutoff is {fir_design.cutoff_hz} Hz, "
                f"while IIR cutoff is {iir_design.cutoff_hz} Hz (tolerance: {CUTOFF_TOLERANCE_HZ} Hz)."
            )

    elif fir_norm == "bandpass":
        if not (isinstance(fir_design.cutoff_hz, (tuple, list, np.ndarray)) and len(fir_design.cutoff_hz) == 2):
            raise ValueError("FIR cutoff_hz must be a 2-element sequence [f_low, f_high] for bandpass.")
        if not (isinstance(iir_design.cutoff_hz, (tuple, list, np.ndarray)) and len(iir_design.cutoff_hz) == 2):
            raise ValueError("IIR cutoff_hz must be a 2-element sequence [f_low, f_high] for bandpass.")
        fir_f1, fir_f2 = fir_design.cutoff_hz[0], fir_design.cutoff_hz[1]
        iir_f1, iir_f2 = iir_design.cutoff_hz[0], iir_design.cutoff_hz[1]
        if not _is_close(fir_f1, iir_f1) or not _is_close(fir_f2, iir_f2):
            raise ValueError(
                f"Bandpass cutoff mismatch: FIR band edges are ({fir_f1}, {fir_f2}) Hz, "
                f"while IIR band edges are ({iir_f1}, {iir_f2}) Hz (tolerance: {CUTOFF_TOLERANCE_HZ} Hz)."
            )

    elif fir_norm == "bandstop":
        # Check if IIR is a notch filter
        is_iir_notch = (iir_type == "notch") or (iir_design.design_method == "iirnotch")
        if is_iir_notch:
            f0 = iir_design.notch_freq
            q = iir_design.quality_factor
            if f0 is None or isinstance(f0, bool) or not isinstance(f0, (int, float, np.floating)) or f0 <= 0:
                raise ValueError(f"IIR notch filter must specify positive notch_freq, got {f0}.")
            if q is None or isinstance(q, bool) or not isinstance(q, (int, float, np.floating)) or q <= 0:
                raise ValueError(f"IIR notch filter must specify positive quality_factor, got {q}.")
            bw = float(f0) / float(q)
            f_low = float(f0) - bw / 2.0
            f_high = float(f0) + bw / 2.0
            if not (isinstance(fir_design.cutoff_hz, (tuple, list, np.ndarray)) and len(fir_design.cutoff_hz) == 2):
                raise ValueError(
                    f"FIR bandstop cutoff_hz must be a 2-element sequence matching notch edges "
                    f"({f_low:.4f}, {f_high:.4f}) Hz, got {fir_design.cutoff_hz}."
                )
            fir_f1, fir_f2 = fir_design.cutoff_hz[0], fir_design.cutoff_hz[1]
            if not _is_close(fir_f1, f_low) or not _is_close(fir_f2, f_high):
                raise ValueError(
                    f"Notch vs FIR bandstop specification mismatch: IIR notch (f0={f0} Hz, Q={q}, "
                    f"BW={bw:.4f} Hz) corresponds to -3 dB band edges ({f_low:.4f}, {f_high:.4f}) Hz, "
                    f"but FIR band edges are ({fir_f1}, {fir_f2}) Hz (tolerance: {CUTOFF_TOLERANCE_HZ} Hz)."
                )
        else:
            # Standard Butterworth bandstop comparison
            if not (isinstance(fir_design.cutoff_hz, (tuple, list, np.ndarray)) and len(fir_design.cutoff_hz) == 2):
                raise ValueError("FIR cutoff_hz must be a 2-element sequence [f_low, f_high] for bandstop.")
            if not (isinstance(iir_design.cutoff_hz, (tuple, list, np.ndarray)) and len(iir_design.cutoff_hz) == 2):
                raise ValueError("IIR cutoff_hz must be a 2-element sequence [f_low, f_high] for bandstop.")
            fir_f1, fir_f2 = fir_design.cutoff_hz[0], fir_design.cutoff_hz[1]
            iir_f1, iir_f2 = iir_design.cutoff_hz[0], iir_design.cutoff_hz[1]
            if not _is_close(fir_f1, iir_f1) or not _is_close(fir_f2, iir_f2):
                raise ValueError(
                    f"Bandstop cutoff mismatch: FIR band edges are ({fir_f1}, {fir_f2}) Hz, "
                    f"while IIR band edges are ({iir_f1}, {iir_f2}) Hz (tolerance: {CUTOFF_TOLERANCE_HZ} Hz)."
                )

    return int(fs)


def compare_fir_and_iir(
    reference: np.ndarray,
    noisy_signal: np.ndarray,
    fir_design: FIRFilterDesign,
    iir_design: IIRFilterDesign,
    sample_rate: Optional[int] = None,
    mode: str = "zero_phase",
    timing_trials: int = 30,
    timing_warmup: int = 5,
    n_freq_points: int = 2048,
) -> Tuple[FilterComparisonReport, np.ndarray, np.ndarray]:
    """
    Conduct an objective, side-by-side comparative analysis of FIR and IIR digital filters
    applied to an identical audio signal under controlled experimental conditions.

    Methodological & Scientific Standards:
    1. Both filters process identical input audio sampled at matching sampling rates.
    2. Causal FIR introduces an exact constant group delay tau = (numtaps - 1) / 2 samples.
       To eliminate transient startup effects and artificial phase-cancellation degradation,
       the reference comparison evaluates the shared steady-state reference interval [tau : N - tau]:
         - Aligned FIR output segment: fir_output[2*tau : N]
         - Corresponding clean reference segment: reference[tau : N - tau]
         - Baseline noisy input segment: noisy_signal[tau : N - tau]
         - IIR output segment: iir_output[tau : N - tau] evaluated over the EXACT same interval
           to ensure a strictly fair, identical time duration and identical clean reference basis.
    3. Causal IIR introduces nonlinear, frequency-dependent group delay that remains an intrinsic
       part of its real-time performance.
    4. In zero-phase mode, forward-backward filtering (`filtfilt` / `sosfiltfilt`) is applied
       over the full signal length [0 : N], resulting in zero group delay for both filters.
    5. Disclosed Design Differences: Cutoff conventions differ: windowed FIR specifies the -6.02 dB
       midpoint, while Butterworth IIR specifies the -3.01 dB half-power point.
    6. Full-length filtered outputs are always preserved and returned to callers.

    Parameters
    ----------
    reference : np.ndarray
        Ground-truth clean audio array (1D mono or 2D stereo).
    noisy_signal : np.ndarray
        Degraded/noisy audio array matching reference shape.
    fir_design : FIRFilterDesign
        Designed FIR filter container.
    iir_design : IIRFilterDesign
        Designed IIR filter container.
    sample_rate : Optional[int], optional
        Sampling frequency in Hz. If None, derived from filter designs.
    mode : str, optional
        Execution mode: 'zero_phase' (default) or 'causal'.
    timing_trials : int, optional
        Number of timed measurement executions. Default is 30.
    timing_warmup : int, optional
        Number of unrecorded warm-up executions. Default is 5.
    n_freq_points : int, optional
        Number of frequency points evaluated from 0 to Nyquist. Default is 2048.

    Returns
    -------
    Tuple[FilterComparisonReport, np.ndarray, np.ndarray]
        report : FilterComparisonReport
            Comprehensive structured performance report with metrics and frequency responses.
        fir_output : np.ndarray
            Full-length filtered signal output from the FIR filter.
        iir_output : np.ndarray
            Full-length filtered signal output from the IIR filter.

    Raises
    ------
    TypeError or ValueError
        If inputs are incompatible, mismatched, or violate DSP invariants.
    """
    # 1. Validate filter compatibility & sampling rate
    fs = _validate_filter_compatibility(fir_design, iir_design, sample_rate)

    clean_mode = mode.strip().lower()
    if clean_mode not in ("zero_phase", "causal"):
        raise ValueError(f"mode must be 'zero_phase' or 'causal', got '{mode}'.")

    # 2. Validate input audio signals
    validate_audio(reference, fs)
    validate_audio(noisy_signal, fs)
    if reference.shape != noisy_signal.shape:
        raise ValueError(
            f"Shape mismatch: reference shape {reference.shape} != "
            f"noisy_signal shape {noisy_signal.shape}."
        )

    num_samples = reference.shape[0]

    # 3. Processing-time benchmark for FIR & full filtering
    fir_timing = measure_processing_time(
        apply_fir_filter,
        noisy_signal,
        fir_design,
        mode=clean_mode,
        num_trials=timing_trials,
        warm_up=timing_warmup,
    )
    fir_output = apply_fir_filter(noisy_signal, fir_design, mode=clean_mode)

    # 4. Processing-time benchmark for IIR & full filtering
    iir_timing = measure_processing_time(
        apply_iir_filter,
        noisy_signal,
        iir_design,
        mode=clean_mode,
        num_trials=timing_trials,
        warm_up=timing_warmup,
    )
    iir_output = apply_iir_filter(noisy_signal, iir_design, mode=clean_mode)

    # 5. Define Evaluation Intervals for Fair Objective Comparison
    tau = int(round(fir_design.group_delay_samples))

    if clean_mode == "causal":
        if num_samples <= 2 * tau:
            raise ValueError(
                f"Signal length ({num_samples} samples) is too short to evaluate causal FIR "
                f"with delay tau = {tau} samples (requires N > 2*tau = {2 * tau} samples)."
            )
        # Shared valid reference interval excluding FIR startup boundary (2*tau samples):
        # eval_ref: reference[tau : num_samples - tau]
        # eval_noisy: noisy_signal[tau : num_samples - tau]
        # eval_fir: fir_output[2*tau : num_samples]
        # eval_iir: iir_output[tau : num_samples - tau]
        eval_ref = reference[tau : num_samples - tau]
        eval_noisy = noisy_signal[tau : num_samples - tau]
        eval_fir = fir_output[2 * tau : num_samples]
        eval_iir = iir_output[tau : num_samples - tau]

        eval_interval = (tau, num_samples - tau)
        eval_count = num_samples - 2 * tau
    else:  # zero_phase
        eval_ref = reference
        eval_noisy = noisy_signal
        eval_fir = fir_output
        eval_iir = iir_output

        eval_interval = (0, num_samples)
        eval_count = num_samples

    # 6. Objective metric evaluations over identical intervals
    fir_metrics: AudioEvaluationResult = evaluate_audio_quality(
        eval_ref, eval_noisy, eval_fir, timing_result=fir_timing
    )
    iir_metrics: AudioEvaluationResult = evaluate_audio_quality(
        eval_ref, eval_noisy, eval_iir, timing_result=iir_timing
    )

    # 7. Frequency Response Evaluation across identical grid
    f_fir, mag_fir, mag_db_fir, phase_fir = get_fir_frequency_response(fir_design, worN=n_freq_points)
    f_iir, mag_iir, mag_db_iir, phase_iir = get_iir_frequency_response(iir_design, worN=n_freq_points)

    if len(f_fir) != n_freq_points:
        raise ValueError(
            f"FIR frequency grid length ({len(f_fir)}) does not match expected length ({n_freq_points})."
        )
    if len(f_iir) != n_freq_points:
        raise ValueError(
            f"IIR frequency grid length ({len(f_iir)}) does not match expected length ({n_freq_points})."
        )

    if not np.all(np.isfinite(f_fir)):
        raise ValueError("FIR frequency response grid contains non-finite (NaN or Inf) values.")
    if not np.all(np.isfinite(f_iir)):
        raise ValueError("IIR frequency response grid contains non-finite (NaN or Inf) values.")

    FREQ_GRID_TOLERANCE_HZ: float = 1e-3
    if not np.allclose(f_fir, f_iir, atol=FREQ_GRID_TOLERANCE_HZ, rtol=1e-5):
        max_diff = float(np.max(np.abs(f_fir - f_iir)))
        raise ValueError(
            f"Frequency response grid mismatch between FIR and IIR: maximum absolute difference is "
            f"{max_diff:.6e} Hz (tolerance: {FREQ_GRID_TOLERANCE_HZ} Hz)."
        )

    freq_response_data = FilterFrequencyResponseData(
        frequencies_hz=f_fir,
        fir_magnitude=mag_fir,
        fir_magnitude_db=mag_db_fir,
        fir_phase_radians=phase_fir,
        iir_magnitude=mag_iir,
        iir_magnitude_db=mag_db_iir,
        iir_phase_radians=phase_iir,
    )

    # 8. SingleFilterBenchmark containers
    fir_bench = SingleFilterBenchmark(
        filter_family="FIR",
        filter_type=fir_design.filter_type,
        mode=clean_mode,
        order=fir_design.order,
        num_taps_or_sections=fir_design.numtaps,
        group_delay_samples=fir_design.group_delay_samples if clean_mode == "causal" else 0.0,
        input_snr_db=fir_metrics.input_snr_db,
        output_snr_db=fir_metrics.output_snr_db,
        snr_improvement_db=fir_metrics.snr_improvement_db,
        input_rmse=fir_metrics.input_rmse,
        output_rmse=fir_metrics.output_rmse,
        rmse_reduction_percent=fir_metrics.rmse_reduction_percent,
        input_mae=fir_metrics.input_mae,
        output_mae=fir_metrics.output_mae,
        median_processing_time_ms=fir_timing["median_ms"],
        timing_statistics=fir_timing,
    )

    iir_bench = SingleFilterBenchmark(
        filter_family="IIR",
        filter_type=iir_design.filter_type,
        mode=clean_mode,
        order=iir_design.order,
        num_taps_or_sections=int(iir_design.sos.shape[0]),
        group_delay_samples=None if clean_mode == "causal" else 0.0,
        input_snr_db=iir_metrics.input_snr_db,
        output_snr_db=iir_metrics.output_snr_db,
        snr_improvement_db=iir_metrics.snr_improvement_db,
        input_rmse=iir_metrics.input_rmse,
        output_rmse=iir_metrics.output_rmse,
        rmse_reduction_percent=iir_metrics.rmse_reduction_percent,
        input_mae=iir_metrics.input_mae,
        output_mae=iir_metrics.output_mae,
        median_processing_time_ms=iir_timing["median_ms"],
        timing_statistics=iir_timing,
    )

    # 9. Comparative delta calculations & Speedup Ratio Handling
    snr_diff = float(fir_bench.snr_improvement_db - iir_bench.snr_improvement_db)

    fir_lat = fir_bench.median_processing_time_ms
    iir_lat = iir_bench.median_processing_time_ms
    if fir_lat <= 0.0 or iir_lat <= 0.0 or not (math.isfinite(fir_lat) and math.isfinite(iir_lat)):
        speedup = None
    else:
        speedup = float(fir_lat / iir_lat)

    if clean_mode == "causal":
        interval_desc = (
            f"Evaluated over shared steady-state reference interval [{tau}, {num_samples - tau}) "
            f"containing {eval_count} samples (N - 2*tau), strictly excluding FIR startup boundary "
            f"(2*tau = {2*tau} samples) and group delay (tau={tau} samples, {tau/fs*1000:.2f} ms). "
            f"Clean reference, noisy baseline, aligned FIR (fir_output[2*tau:N]), and causal IIR "
            f"are all compared over this exact shared reference interval."
        )
    else:
        interval_desc = "Evaluated over full signal duration with zero-phase filtering."

    notes = (
        f"Comparative Analysis ({clean_mode} mode):\n"
        f"- FIR Order {fir_bench.order} ({fir_bench.num_taps_or_sections} taps, {fir_design.window} window) vs "
        f"IIR Order {iir_bench.order} ({iir_bench.num_taps_or_sections} Second-Order Sections).\n"
        f"- Cutoff Conventions: FIR fc defines -6.02 dB midpoint; IIR fc defines -3.01 dB half-power edge.\n"
        f"- Phase Performance: FIR provides strictly linear phase delay in causal mode ({fir_bench.group_delay_samples} samples), "
        f"while IIR introduces frequency-dependent group delay which remains part of its causal response.\n"
        f"- Interval Alignment: {interval_desc}"
    )

    report = FilterComparisonReport(
        sample_rate=fs,
        mode=clean_mode,
        evaluation_interval=eval_interval,
        evaluation_sample_count=eval_count,
        fir_benchmark=fir_bench,
        iir_benchmark=iir_bench,
        snr_difference_db=snr_diff,
        speedup_ratio=speedup,
        frequency_response=freq_response_data,
        notes=notes,
    )

    return report, fir_output, iir_output
