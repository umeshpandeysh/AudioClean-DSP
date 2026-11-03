"""
AudioClean DSP - FIR vs IIR Filter Comparison Engine
Provides scientifically fair, objective comparative performance evaluation between
Finite Impulse Response (FIR) and Infinite Impulse Response (IIR) digital filters
applied to identical discrete audio signals.
"""

from dataclasses import asdict, dataclass
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
        SNR of noisy input relative to clean reference in dB.
    output_snr_db : float
        SNR of filtered output relative to clean reference in dB.
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
    align_causal_delay : bool
        Whether causal FIR linear-phase delay was compensated during reference error comparison.
    fir_benchmark : SingleFilterBenchmark
        Benchmark results for the FIR filter.
    iir_benchmark : SingleFilterBenchmark
        Benchmark results for the IIR filter.
    snr_difference_db : float
        FIR SNR improvement minus IIR SNR improvement (positive favors FIR, negative favors IIR).
    speedup_ratio : float
        FIR median latency / IIR median latency (>1.0 indicates IIR is faster).
    notes : str
        Scientific caveats regarding design differences, phase linearity, and cutoff definitions.
    """

    sample_rate: int
    mode: str
    align_causal_delay: bool
    fir_benchmark: SingleFilterBenchmark
    iir_benchmark: SingleFilterBenchmark
    snr_difference_db: float
    speedup_ratio: float
    notes: str

    def as_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary representation."""
        return asdict(self)


def align_causal_signal(
    signal: np.ndarray,
    delay_samples: int,
) -> np.ndarray:
    """
    Compensate for integer causal filter group delay by shifting backward and padding edge.

    Parameters
    ----------
    signal : np.ndarray
        Delayed audio signal array (1D or 2D).
    delay_samples : int
        Number of discrete samples to shift backward.

    Returns
    -------
    np.ndarray
        Time-aligned signal matching original length.
    """
    if delay_samples <= 0:
        return signal.copy()

    n = signal.shape[0]
    if delay_samples >= n:
        raise ValueError(f"delay_samples ({delay_samples}) must be less than signal length ({n}).")

    if signal.ndim == 1:
        aligned = np.zeros_like(signal)
        aligned[: n - delay_samples] = signal[delay_samples:]
        # zero-pad trailing edge
        aligned[n - delay_samples :] = 0.0
        return aligned
    else:
        aligned = np.zeros_like(signal)
        aligned[: n - delay_samples, :] = signal[delay_samples:, :]
        aligned[n - delay_samples :, :] = 0.0
        return aligned


def compare_fir_and_iir(
    reference: np.ndarray,
    noisy_signal: np.ndarray,
    fir_design: FIRFilterDesign,
    iir_design: IIRFilterDesign,
    sample_rate: Optional[int] = None,
    mode: str = "zero_phase",
    align_causal_fir: bool = True,
    timing_trials: int = 30,
    timing_warmup: int = 5,
) -> Tuple[FilterComparisonReport, np.ndarray, np.ndarray]:
    """
    Conduct an objective, side-by-side comparative analysis of FIR and IIR digital filters
    applied to an identical audio signal under controlled experimental conditions.

    Methodological & Scientific Standards:
    1. Both filters process identical input audio sampled at matching sampling rates.
    2. Causal FIR introduces an exact constant group delay tau = (numtaps - 1) / 2 samples.
       If align_causal_fir is True, the causal FIR output is shifted backward by tau samples
       for reference error comparison, removing artificial phase-cancellation degradation.
    3. Causal IIR introduces nonlinear, frequency-dependent group delay; delay varies with
       frequency and cannot be compensated by a single constant shift.
    4. Zero-phase filtering applies forward-backward passes (`filtfilt` / `sosfiltfilt`),
       achieving zero phase distortion at the expense of squaring the transfer function
       magnitude response (|H(f)|^2) and doubling the effective stopband attenuation in dB.
    5. Cutoff conventions differ: windowed FIR cutoff specifies the -6.02 dB (half-amplitude)
       point, while Butterworth IIR specifies the -3.01 dB (half-power) point.
    6. Neither filter is declared universally superior. The report presents the trade-off
       between phase linearity, order, and computational efficiency per pole.

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
    align_causal_fir : bool, optional
        If True and mode is 'causal', aligns the FIR output by its group delay before
        metric calculation. Default is True.
    timing_trials : int, optional
        Number of timed measurement executions. Default is 30.
    timing_warmup : int, optional
        Number of unrecorded warm-up executions. Default is 5.

    Returns
    -------
    Tuple[FilterComparisonReport, np.ndarray, np.ndarray]
        report : FilterComparisonReport
            Comprehensive structured performance report.
        fir_output : np.ndarray
            Actual filtered signal output from the FIR filter.
        iir_output : np.ndarray
            Actual filtered signal output from the IIR filter.

    Raises
    ------
    TypeError or ValueError
        If inputs are incompatible, mismatched, or violate DSP invariants.
    """
    if not isinstance(fir_design, FIRFilterDesign):
        raise TypeError(f"fir_design must be FIRFilterDesign, got {type(fir_design).__name__}.")
    if not isinstance(iir_design, IIRFilterDesign):
        raise TypeError(f"iir_design must be IIRFilterDesign, got {type(iir_design).__name__}.")

    fs = sample_rate if sample_rate is not None else fir_design.sample_rate
    if fir_design.sample_rate != iir_design.sample_rate:
        raise ValueError(
            f"Sampling rate mismatch: FIR design ({fir_design.sample_rate} Hz) != "
            f"IIR design ({iir_design.sample_rate} Hz)."
        )
    if fs != fir_design.sample_rate:
        raise ValueError(f"Specified sample_rate ({fs}) does not match filter sample_rate ({fir_design.sample_rate}).")

    clean_mode = mode.strip().lower()
    if clean_mode not in ("zero_phase", "causal"):
        raise ValueError(f"mode must be 'zero_phase' or 'causal', got '{mode}'.")

    # Validate audio signals
    validate_audio(reference, fs)
    validate_audio(noisy_signal, fs)
    if reference.shape != noisy_signal.shape:
        raise ValueError(f"Shape mismatch: reference {reference.shape} != noisy_signal {noisy_signal.shape}.")

    # 1. Processing-time benchmark for FIR
    fir_timing = measure_processing_time(
        apply_fir_filter,
        noisy_signal,
        fir_design,
        mode=clean_mode,
        num_trials=timing_trials,
        warm_up=timing_warmup,
    )
    fir_output = apply_fir_filter(noisy_signal, fir_design, mode=clean_mode)

    # 2. Processing-time benchmark for IIR
    iir_timing = measure_processing_time(
        apply_iir_filter,
        noisy_signal,
        iir_design,
        mode=clean_mode,
        num_trials=timing_trials,
        warm_up=timing_warmup,
    )
    iir_output = apply_iir_filter(noisy_signal, iir_design, mode=clean_mode)

    # 3. Reference alignment for causal FIR
    fir_eval = fir_output
    delay_samples = int(round(fir_design.group_delay_samples))
    did_align_fir = False
    if clean_mode == "causal" and align_causal_fir:
        fir_eval = align_causal_signal(fir_output, delay_samples)
        did_align_fir = True

    # 4. Objective metric evaluations
    fir_metrics: AudioEvaluationResult = evaluate_audio_quality(
        reference, noisy_signal, fir_eval, timing_result=fir_timing
    )
    iir_metrics: AudioEvaluationResult = evaluate_audio_quality(
        reference, noisy_signal, iir_output, timing_result=iir_timing
    )

    # 5. Build SingleFilterBenchmark for FIR
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

    # 6. Build SingleFilterBenchmark for IIR
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

    # 7. Comparative delta calculations
    snr_diff = float(fir_bench.snr_improvement_db - iir_bench.snr_improvement_db)
    iir_lat = iir_bench.median_processing_time_ms
    speedup = float(fir_bench.median_processing_time_ms / iir_lat) if iir_lat > 0 else 1.0

    notes = (
        f"Comparative Analysis ({clean_mode} mode):\n"
        f"- FIR Order {fir_bench.order} ({fir_bench.num_taps_or_sections} taps, {fir_design.window} window) vs "
        f"IIR Order {iir_bench.order} ({iir_bench.num_taps_or_sections} Second-Order Sections).\n"
        f"- Cutoff Conventions: FIR fc defines -6.02 dB midpoint; IIR fc defines -3.01 dB half-power edge.\n"
        f"- Phase Performance: FIR provides strictly linear phase delay in causal mode ({fir_bench.group_delay_samples} samples), "
        f"while IIR introduces frequency-dependent group delay.\n"
        f"- Causal FIR Alignment: {'Compensated by ' + str(delay_samples) + ' samples' if did_align_fir else 'Uncompensated'}."
    )

    report = FilterComparisonReport(
        sample_rate=fs,
        mode=clean_mode,
        align_causal_delay=did_align_fir,
        fir_benchmark=fir_bench,
        iir_benchmark=iir_bench,
        snr_difference_db=snr_diff,
        speedup_ratio=speedup,
        notes=notes,
    )

    return report, fir_output, iir_output
