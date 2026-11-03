"""
AudioClean DSP - Quantitative Performance Evaluation Module
Provides reference-based objective evaluation metrics (SNR, RMSE, MAE, SNR Improvement,
RMSE Reduction) and rigorous computation timing for digital audio filtering.
"""

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np


def _validate_signal_pair(
    reference: np.ndarray,
    evaluated_signal: np.ndarray,
    ref_name: str = "reference",
    eval_name: str = "evaluated_signal",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate that two audio signals have compatible shapes, valid numeric types, and finite real values.
    Returns float64 views of the arrays.

    Raises
    ------
    TypeError
        If arrays are not numpy.ndarray or have non-numeric/complex dtypes.
    ValueError
        If arrays are empty, non-finite, complex-valued, shape-mismatched,
        or have unsupported dimensions/channel counts.
    """
    if not isinstance(reference, np.ndarray):
        raise TypeError(f"{ref_name} must be a NumPy ndarray, got {type(reference).__name__}.")
    if not isinstance(evaluated_signal, np.ndarray):
        raise TypeError(f"{eval_name} must be a NumPy ndarray, got {type(evaluated_signal).__name__}.")

    # Reject complex types explicitly
    if np.iscomplexobj(reference):
        raise TypeError(f"{ref_name} contains complex values; audio signals must be real-valued.")
    if np.iscomplexobj(evaluated_signal):
        raise TypeError(f"{eval_name} contains complex values; audio signals must be real-valued.")

    # Reject boolean arrays
    if reference.dtype == bool:
        raise TypeError(f"{ref_name} has boolean dtype; expected numeric audio samples.")
    if evaluated_signal.dtype == bool:
        raise TypeError(f"{eval_name} has boolean dtype; expected numeric audio samples.")

    # Validate numeric dtype
    if not np.issubdtype(reference.dtype, np.number):
        raise TypeError(f"{ref_name} has non-numeric dtype '{reference.dtype}'. Expected numeric data.")
    if not np.issubdtype(evaluated_signal.dtype, np.number):
        raise TypeError(f"{eval_name} has non-numeric dtype '{evaluated_signal.dtype}'. Expected numeric data.")

    if reference.size == 0:
        raise ValueError(f"{ref_name} array is empty.")
    if evaluated_signal.size == 0:
        raise ValueError(f"{eval_name} array is empty.")

    if not np.all(np.isfinite(reference)):
        raise ValueError(f"{ref_name} contains non-finite values (NaN or Inf).")
    if not np.all(np.isfinite(evaluated_signal)):
        raise ValueError(f"{eval_name} contains non-finite values (NaN or Inf).")

    if reference.shape != evaluated_signal.shape:
        raise ValueError(
            f"Shape mismatch: {ref_name} shape {reference.shape} != "
            f"{eval_name} shape {evaluated_signal.shape}."
        )

    if reference.ndim not in (1, 2):
        raise ValueError(f"Unsupported signal dimensionality: {reference.ndim}D. Expected 1D or 2D.")

    if reference.ndim == 2 and reference.shape[1] not in (1, 2):
        raise ValueError(
            f"Unsupported channel count: {reference.shape[1]}. Only mono (1) and stereo (2) supported."
        )

    try:
        ref_f64 = reference.astype(np.float64, copy=False)
        eval_f64 = evaluated_signal.astype(np.float64, copy=False)
    except Exception as err:
        raise TypeError(f"Could not convert signal data to float64: {err}") from err

    return ref_f64, eval_f64


def calculate_snr(
    reference: np.ndarray,
    evaluated_signal: np.ndarray,
    per_channel: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Calculate the Signal-to-Noise Ratio (SNR) in decibels (dB) against a clean reference.

    Mathematical Formulation:
    For a known discrete-time clean reference s_ref[n] and evaluated signal s_eval[n]:
        e[n] = s_eval[n] - s_ref[n]                  (residual error / noise component)
        P_signal = (1 / M) * sum(s_ref[n]^2)          (reference signal power)
        P_noise  = (1 / M) * sum(e[n]^2)              (noise / error power)

    Signal-to-Noise Ratio in decibels:
        SNR = 10 * log10(P_signal / P_noise)

    Multichannel Handling:
    - Overall SNR: Computed by aggregating total signal power and total error power across
      all samples and channels (energy pooling). This avoids misleading arithmetic averages
      of logarithmic dB quantities.
    - Per-channel SNR: When `per_channel=True`, also computes individual channel SNRs.

    Edge Cases:
    - P_noise == 0.0: Perfect reconstruction (evaluated signal equals reference). Returns float('inf').
    - P_signal == 0.0: Reference is purely silent (zero energy).
      - For single-channel / overall SNR: Raises ValueError as SNR is undefined.
      - For per-channel calculation: Channels with zero signal energy receive float('-inf') if noise > 0,
        or float('nan') if also noise == 0. If ALL reference channels have zero energy, ValueError is raised.

    Parameters
    ----------
    reference : np.ndarray
        Ground-truth clean audio reference signal array (mono 1D or stereo 2D).
    evaluated_signal : np.ndarray
        Signal to evaluate (e.g. noisy input or filtered output) of identical shape.
    per_channel : bool, optional
        If True and signal is 2D, returns a tuple (overall_snr_db, channel_snrs_array).
        Default is False.

    Returns
    -------
    Union[float, Tuple[float, np.ndarray]]
        Overall SNR in dB (or tuple of overall SNR and per-channel SNRs if per_channel is True).

    Raises
    ------
    TypeError
        If inputs are not numeric real-valued NumPy arrays.
    ValueError
        If inputs are empty, non-finite, shape-mismatched, or if reference signal power is zero.
    """
    ref, eval_sig = _validate_signal_pair(reference, evaluated_signal)

    # 1. Total Signal Power and Total Error Power
    p_signal_total = float(np.mean(ref ** 2))
    if p_signal_total == 0.0:
        raise ValueError("Reference signal has zero power (silent); SNR is mathematically undefined.")

    error = eval_sig - ref
    p_noise_total = float(np.mean(error ** 2))

    if p_noise_total == 0.0:
        overall_snr = float("inf")
    else:
        overall_snr = float(10.0 * np.log10(p_signal_total / p_noise_total))

    if not per_channel or ref.ndim == 1:
        return overall_snr

    # 2. Per-Channel Calculation for 2D Signals
    n_channels = ref.shape[1]
    ch_snrs = np.zeros(n_channels, dtype=np.float64)
    for c in range(n_channels):
        p_sig_c = float(np.mean(ref[:, c] ** 2))
        p_noise_c = float(np.mean(error[:, c] ** 2))
        if p_sig_c == 0.0 and p_noise_c == 0.0:
            ch_snrs[c] = float("nan")
        elif p_sig_c == 0.0:
            ch_snrs[c] = float("-inf")
        elif p_noise_c == 0.0:
            ch_snrs[c] = float("inf")
        else:
            ch_snrs[c] = float(10.0 * np.log10(p_sig_c / p_noise_c))

    return overall_snr, ch_snrs


def calculate_rmse(
    reference: np.ndarray,
    evaluated_signal: np.ndarray,
    per_channel: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Calculate the Root Mean Square Error (RMSE) between an evaluated signal and clean reference.

    Mathematical Formulation:
        RMSE = sqrt( (1 / M) * sum((reference - evaluated_signal)^2) )

    Parameters
    ----------
    reference : np.ndarray
        Clean reference signal array.
    evaluated_signal : np.ndarray
        Evaluated signal array with matching shape.
    per_channel : bool, optional
        If True and signal is 2D, returns (overall_rmse, channel_rmses_array). Default False.

    Returns
    -------
    Union[float, Tuple[float, np.ndarray]]
        Overall RMSE value (or tuple with per-channel values).
    """
    ref, eval_sig = _validate_signal_pair(reference, evaluated_signal)
    error = eval_sig - ref
    overall_rmse = float(np.sqrt(np.mean(error ** 2)))

    if not per_channel or ref.ndim == 1:
        return overall_rmse

    n_channels = ref.shape[1]
    ch_rmses = np.zeros(n_channels, dtype=np.float64)
    for c in range(n_channels):
        ch_rmses[c] = float(np.sqrt(np.mean(error[:, c] ** 2)))

    return overall_rmse, ch_rmses


def calculate_mae(
    reference: np.ndarray,
    evaluated_signal: np.ndarray,
    per_channel: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Calculate the Mean Absolute Error (MAE) between an evaluated signal and clean reference.

    Mathematical Formulation:
        MAE = (1 / M) * sum(|reference - evaluated_signal|)

    Parameters
    ----------
    reference : np.ndarray
        Clean reference signal array.
    evaluated_signal : np.ndarray
        Evaluated signal array with matching shape.
    per_channel : bool, optional
        If True and signal is 2D, returns (overall_mae, channel_maes_array). Default False.

    Returns
    -------
    Union[float, Tuple[float, np.ndarray]]
        Overall MAE value (or tuple with per-channel values).
    """
    ref, eval_sig = _validate_signal_pair(reference, evaluated_signal)
    error = np.abs(eval_sig - ref)
    overall_mae = float(np.mean(error))

    if not per_channel or ref.ndim == 1:
        return overall_mae

    n_channels = ref.shape[1]
    ch_maes = np.zeros(n_channels, dtype=np.float64)
    for c in range(n_channels):
        ch_maes[c] = float(np.mean(error[:, c]))

    return overall_mae, ch_maes


def measure_processing_time(
    func: Callable[..., Any],
    *args: Any,
    num_trials: int = 30,
    warm_up: int = 5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Measure filtering computation time in isolation using warm-up runs and repeated trials.

    Parameters
    ----------
    func : Callable[..., Any]
        The filtering function or callable to benchmark.
    *args : Any
        Positional arguments forwarded to func.
    num_trials : int, optional
        Number of timed measurement executions. Default is 30. Must be integer >= 1 (bool rejected).
    warm_up : int, optional
        Number of un-timed preliminary warm-up executions. Default is 5. Must be integer >= 0 (bool rejected).
    **kwargs : Any
        Keyword arguments forwarded to func.

    Returns
    -------
    Dict[str, Any]
        Dictionary with benchmark summary statistics:
        - 'median_seconds': float
        - 'median_ms': float
        - 'mean_ms': float
        - 'std_ms': float
        - 'min_ms': float
        - 'max_ms': float
        - 'num_trials': int
        - 'warm_up': int

    Raises
    ------
    TypeError
        If func is not callable or if num_trials / warm_up are booleans or invalid types.
    ValueError
        If num_trials < 1 or warm_up < 0.
    """
    if not callable(func):
        raise TypeError(f"Target func must be callable, got {type(func).__name__}.")

    # Reject boolean values explicitly (since bool is a subclass of int in Python)
    if isinstance(num_trials, bool):
        raise TypeError("num_trials cannot be a boolean value.")
    if isinstance(warm_up, bool):
        raise TypeError("warm_up cannot be a boolean value.")

    if not isinstance(num_trials, (int, np.integer)):
        raise TypeError(f"num_trials must be an integer, got {type(num_trials).__name__}.")
    if not isinstance(warm_up, (int, np.integer)):
        raise TypeError(f"warm_up must be an integer, got {type(warm_up).__name__}.")

    if num_trials < 1:
        raise ValueError(f"num_trials must be an integer >= 1, got {num_trials}.")
    if warm_up < 0:
        raise ValueError(f"warm_up must be an integer >= 0, got {warm_up}.")

    # 1. Warm-up iterations (cache warming, JIT/interpreter stabilization)
    for _ in range(warm_up):
        func(*args, **kwargs)

    # 2. Timed measurement trials
    trial_durations = np.zeros(num_trials, dtype=np.float64)
    for i in range(num_trials):
        t_start = time.perf_counter()
        func(*args, **kwargs)
        t_end = time.perf_counter()
        trial_durations[i] = t_end - t_start

    trial_ms = trial_durations * 1000.0
    median_sec = float(np.median(trial_durations))
    median_ms = float(np.median(trial_ms))
    mean_ms = float(np.mean(trial_ms))
    std_ms = float(np.std(trial_ms))
    min_ms = float(np.min(trial_ms))
    max_ms = float(np.max(trial_ms))

    return {
        "median_seconds": median_sec,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "num_trials": int(num_trials),
        "warm_up": int(warm_up),
    }


@dataclass(frozen=True)
class AudioEvaluationResult:
    """
    Structured immutable container holding reference-based audio restoration metrics.

    Attributes
    ----------
    input_snr_db : float
        Signal-to-Noise Ratio of degraded/noisy signal relative to clean reference in dB.
    output_snr_db : float
        Signal-to-Noise Ratio of filtered/restored signal relative to clean reference in dB.
    snr_improvement_db : float
        Difference between output SNR and input SNR: Delta SNR = output_snr_db - input_snr_db.
        If both input and output are infinite, snr_improvement_db is 0.0 dB.
    input_rmse : float
        Root Mean Square Error of noisy input relative to clean reference.
    output_rmse : float
        Root Mean Square Error of filtered output relative to clean reference.
    rmse_reduction_percent : float
        Percentage reduction in RMSE: ((input_rmse - output_rmse) / input_rmse) * 100.
        If input_rmse == 0.0, rmse_reduction_percent is safely defined as 0.0.
    input_mae : float
        Mean Absolute Error of noisy input relative to clean reference.
    output_mae : float
        Mean Absolute Error of filtered output relative to clean reference.
    processing_time : Optional[Dict[str, Any]]
        Optional execution timing dictionary measured via measure_processing_time.
    """

    input_snr_db: float
    output_snr_db: float
    snr_improvement_db: float
    input_rmse: float
    output_rmse: float
    rmse_reduction_percent: float
    input_mae: float
    output_mae: float
    processing_time: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return metrics as a standard dictionary."""
        return asdict(self)


def evaluate_audio_quality(
    reference: np.ndarray,
    noisy_signal: np.ndarray,
    filtered_signal: np.ndarray,
    timing_result: Optional[Dict[str, Any]] = None,
) -> AudioEvaluationResult:
    """
    Perform a complete quantitative comparative evaluation of audio restoration.

    Distinguishes strictly between:
    1. Original clean ground-truth reference
    2. Input noisy/degraded signal
    3. Filtered/restored output signal

    Parameters
    ----------
    reference : np.ndarray
        Clean reference audio array.
    noisy_signal : np.ndarray
        Degraded/noisy audio array of identical shape.
    filtered_signal : np.ndarray
        Restored/filtered audio array of identical shape.
    timing_result : Optional[Dict[str, Any]], optional
        Optional timing benchmark dictionary from measure_processing_time.

    Returns
    -------
    AudioEvaluationResult
        Data container containing input/output SNR, SNR improvement, RMSE, RMSE reduction,
        MAE, and optional processing time statistics.

    Raises
    ------
    ValueError or TypeError
        If any input is invalid, shape-mismatched, non-finite, or empty.
    """
    # Validate reference vs noisy
    ref_f64, noisy_f64 = _validate_signal_pair(reference, noisy_signal, "reference", "noisy_signal")
    # Validate reference vs filtered
    _, filt_f64 = _validate_signal_pair(reference, filtered_signal, "reference", "filtered_signal")

    # 1. SNR Metrics
    in_snr = calculate_snr(ref_f64, noisy_f64)
    out_snr = calculate_snr(ref_f64, filt_f64)

    # Handle infinite SNR edge cases:
    # If both input and output are inf, improvement is 0.0 dB
    if math.isinf(in_snr) and math.isinf(out_snr):
        snr_gain = 0.0
    elif math.isinf(out_snr):
        snr_gain = float("inf")
    elif math.isinf(in_snr):
        snr_gain = float("-inf")
    else:
        snr_gain = float(out_snr - in_snr)

    # 2. RMSE Metrics
    in_rmse = calculate_rmse(ref_f64, noisy_f64)
    out_rmse = calculate_rmse(ref_f64, filt_f64)
    if in_rmse == 0.0:
        rmse_reduction = 0.0
    else:
        rmse_reduction = float(((in_rmse - out_rmse) / in_rmse) * 100.0)

    # 3. MAE Metrics
    in_mae = calculate_mae(ref_f64, noisy_f64)
    out_mae = calculate_mae(ref_f64, filt_f64)

    return AudioEvaluationResult(
        input_snr_db=float(in_snr),
        output_snr_db=float(out_snr),
        snr_improvement_db=snr_gain,
        input_rmse=float(in_rmse),
        output_rmse=float(out_rmse),
        rmse_reduction_percent=rmse_reduction,
        input_mae=float(in_mae),
        output_mae=float(out_mae),
        processing_time=timing_result,
    )
