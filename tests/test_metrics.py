"""
AudioClean DSP - Unit Tests for Quantitative Evaluation Metrics (Phase 6)
Verifies reference-based objective evaluation metrics: SNR, RMSE, MAE, SNR Improvement,
RMSE Reduction, stereo handling, edge cases, and processing time benchmarking.
"""

import time
import numpy as np
import pytest

from src.metrics import (
    AudioEvaluationResult,
    calculate_mae,
    calculate_rmse,
    calculate_snr,
    evaluate_audio_quality,
    measure_processing_time,
)


# -----------------------------------------------------------------------------
# 1. MATHEMATICAL CORRECTNESS WITH DETERMINISTIC SYNTHETIC SIGNALS
# -----------------------------------------------------------------------------

def test_snr_analytical_value_mono():
    """
    Test SNR calculation against an exact analytical hand-calculated value.
    Let reference s[n] = [2, -2, 2, -2].
        P_signal = (2^2 + (-2)^2 + 2^2 + (-2)^2) / 4 = 16 / 4 = 4.0
    Let error e[n] = [0.2, -0.2, 0.2, -0.2].
        P_noise = (0.04 + 0.04 + 0.04 + 0.04) / 4 = 0.04
    Analytical SNR = 10 * log10(4.0 / 0.04) = 10 * log10(100) = 20.0 dB.
    """
    ref = np.array([2.0, -2.0, 2.0, -2.0], dtype=np.float64)
    noisy = ref + np.array([0.2, -0.2, 0.2, -0.2], dtype=np.float64)

    snr = calculate_snr(ref, noisy)
    assert isinstance(snr, float)
    assert np.isclose(snr, 20.0, atol=1e-12)


def test_snr_identical_signals_perfect_reconstruction():
    """Identical signals yield zero error power, which must return positive infinity."""
    ref = np.array([0.5, -0.3, 0.8, -0.1], dtype=np.float64)
    snr = calculate_snr(ref, ref.copy())
    assert snr == float("inf")


def test_snr_zero_energy_reference_rejected():
    """A purely silent reference has zero power, making SNR mathematically undefined."""
    ref = np.zeros(100, dtype=np.float64)
    eval_sig = np.ones(100, dtype=np.float64) * 0.1
    with pytest.raises(ValueError, match="zero power"):
        calculate_snr(ref, eval_sig)


def test_snr_multichannel_overall_and_per_channel():
    """
    Verify multichannel pooling vs per-channel calculation on 2D stereo signal.
    Channel 0: P_sig = 4.0, P_noise = 0.04 -> SNR = 20 dB.
    Channel 1: P_sig = 1.0, P_noise = 0.10 -> SNR = 10 dB.
    Overall: Total P_sig = (4.0 + 1.0)/2 = 2.5
             Total P_noise = (0.04 + 0.10)/2 = 0.07
             Overall SNR = 10 * log10(2.5 / 0.07) = 15.5284 dB.
    """
    ref_ch0 = np.array([2.0, -2.0, 2.0, -2.0])
    noise_ch0 = np.array([0.2, -0.2, 0.2, -0.2])

    ref_ch1 = np.array([1.0, -1.0, 1.0, -1.0])
    noise_ch1 = np.array([np.sqrt(0.1), -np.sqrt(0.1), np.sqrt(0.1), -np.sqrt(0.1)])

    ref_stereo = np.column_stack([ref_ch0, ref_ch1])
    noisy_stereo = np.column_stack([ref_ch0 + noise_ch0, ref_ch1 + noise_ch1])

    # 1. Default overall SNR
    overall = calculate_snr(ref_stereo, noisy_stereo, per_channel=False)
    expected_overall = 10.0 * np.log10(2.5 / 0.07)
    assert np.isclose(overall, expected_overall, atol=1e-10)

    # 2. Per-channel tuple
    overall_ret, ch_snrs = calculate_snr(ref_stereo, noisy_stereo, per_channel=True)
    assert np.isclose(overall_ret, expected_overall, atol=1e-10)
    assert len(ch_snrs) == 2
    assert np.isclose(ch_snrs[0], 20.0, atol=1e-10)
    assert np.isclose(ch_snrs[1], 10.0, atol=1e-10)


def test_rmse_and_mae_hand_calculated_vector():
    """
    Verify RMSE and MAE against exact hand-calculated figures.
    ref  = [1.0,  2.0, -1.0, 0.0]
    eval = [1.2,  1.7, -0.6, 0.5]
    diff = [0.2, -0.3,  0.4, 0.5]
    sq   = [0.04, 0.09, 0.16, 0.25] -> sum = 0.54 -> mean = 0.135
    RMSE = sqrt(0.135) = 0.3674234614174767
    MAE  = (0.2 + 0.3 + 0.4 + 0.5) / 4 = 1.4 / 4 = 0.35
    """
    ref = np.array([1.0, 2.0, -1.0, 0.0], dtype=np.float64)
    eval_sig = np.array([1.2, 1.7, -0.6, 0.5], dtype=np.float64)

    rmse = calculate_rmse(ref, eval_sig)
    mae = calculate_mae(ref, eval_sig)

    assert np.isclose(rmse, np.sqrt(0.135), atol=1e-12)
    assert np.isclose(mae, 0.35, atol=1e-12)


def test_rmse_reduction_percentage_calculation():
    """Test RMSE reduction formula under normal enhancement and zero-reduction edge case."""
    ref = np.array([1.0, -1.0, 1.0, -1.0])
    noisy = ref + np.array([0.4, -0.4, 0.4, -0.4])    # Error RMS = 0.4
    filtered = ref + np.array([0.1, -0.1, 0.1, -0.1]) # Error RMS = 0.1

    result = evaluate_audio_quality(ref, noisy, filtered)
    # in_rmse = 0.4, out_rmse = 0.1
    # reduction = ((0.4 - 0.1) / 0.4) * 100 = 75.0%
    assert np.isclose(result.input_rmse, 0.4, atol=1e-12)
    assert np.isclose(result.output_rmse, 0.1, atol=1e-12)
    assert np.isclose(result.rmse_reduction_percent, 75.0, atol=1e-12)

    # Edge case 1: Noisy and filtered are both identical to reference (zero input RMSE, zero output RMSE)
    res_zero = evaluate_audio_quality(ref, ref.copy(), ref.copy())
    assert res_zero.input_rmse == 0.0
    assert res_zero.output_rmse == 0.0
    assert res_zero.rmse_reduction_percent == 0.0

    # Edge case 2: Noisy was identical to reference (input RMSE == 0), but filtering introduced error (output RMSE > 0)
    # Error increased from 0 to positive, so reduction is negative infinity
    res_degraded_from_clean = evaluate_audio_quality(ref, ref.copy(), filtered)
    assert res_degraded_from_clean.input_rmse == 0.0
    assert res_degraded_from_clean.output_rmse > 0.0
    assert res_degraded_from_clean.rmse_reduction_percent == float("-inf")


def test_evaluate_audio_quality_structure_and_values():
    """Test evaluate_audio_quality returns a correctly populated AudioEvaluationResult."""
    np.random.seed(42)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    ref = 0.8 * np.sin(2 * np.pi * 440.0 * t)
    noise = 0.2 * np.sin(2 * np.pi * 3000.0 * t)
    noisy = ref + noise
    # Simulated partial restoration (attenuating 3000 Hz by factor of 4)
    filtered = ref + 0.05 * np.sin(2 * np.pi * 3000.0 * t)

    res = evaluate_audio_quality(ref, noisy, filtered)
    assert isinstance(res, AudioEvaluationResult)

    d = res.as_dict()
    assert "input_snr_db" in d
    assert "output_snr_db" in d
    assert "snr_improvement_db" in d
    assert "input_rmse" in d
    assert "output_rmse" in d
    assert "rmse_reduction_percent" in d
    assert "input_mae" in d
    assert "output_mae" in d

    # Verification of improvement
    assert res.output_snr_db > res.input_snr_db
    assert res.snr_improvement_db > 0.0
    assert res.output_rmse < res.input_rmse
    assert res.rmse_reduction_percent > 0.0
    assert res.output_mae < res.input_mae


def test_evaluate_audio_quality_negative_improvement():
    """Verify that when filtering distorts or adds noise, SNR improvement is correctly negative."""
    ref = np.array([1.0, -1.0, 1.0, -1.0])
    noisy = ref + np.array([0.05, -0.05, 0.05, -0.05])
    # Severe distortion/over-filtering adds large error
    severely_degraded = ref + np.array([0.8, -0.8, 0.8, -0.8])

    res = evaluate_audio_quality(ref, noisy, severely_degraded)
    assert res.snr_improvement_db < 0.0
    assert res.output_snr_db < res.input_snr_db
    assert res.rmse_reduction_percent < 0.0


# -----------------------------------------------------------------------------
# 2. VALIDATION AND EXCEPTION HANDLING
# -----------------------------------------------------------------------------

def test_validation_mismatched_shapes():
    """Mismatched signal shapes must raise ValueError."""
    ref = np.ones(100)
    eval_short = np.ones(90)
    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_snr(ref, eval_short)

    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_rmse(ref, eval_short)

    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_mae(ref, eval_short)

    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluate_audio_quality(ref, ref, eval_short)


def test_validation_non_finite_values():
    """Signals containing NaN or Inf must raise ValueError."""
    ref = np.array([1.0, np.nan, 2.0])
    eval_sig = np.array([1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="non-finite"):
        calculate_snr(ref, eval_sig)

    ref_inf = np.array([1.0, np.inf, 2.0])
    with pytest.raises(ValueError, match="non-finite"):
        calculate_rmse(ref_inf, eval_sig)


def test_validation_empty_arrays():
    """Empty arrays must raise ValueError."""
    empty = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="empty"):
        calculate_snr(empty, empty)


def test_validation_type_errors():
    """Non-NumPy arrays must raise TypeError."""
    with pytest.raises(TypeError, match="NumPy ndarray"):
        calculate_snr([1.0, 2.0], np.array([1.0, 2.0]))

    with pytest.raises(TypeError, match="NumPy ndarray"):
        calculate_rmse(np.array([1.0, 2.0]), [1.0, 2.0])


def test_validation_unsupported_dimensions():
    """3D or higher arrays must raise ValueError."""
    arr3d = np.ones((10, 2, 2))
    with pytest.raises(ValueError, match="Unsupported signal dimensionality"):
        calculate_snr(arr3d, arr3d)


# -----------------------------------------------------------------------------
# 3. PROCESSING TIME MEASUREMENT
# -----------------------------------------------------------------------------

def test_measure_processing_time_behavior():
    """Verify warm-up runs, timing execution, and returned dictionary statistics."""
    calls = []

    def mock_filter(x, multiplier=2.0):
        calls.append(1)
        time.sleep(0.001)  # 1 ms sleep
        return x * multiplier

    arr = np.ones(100)
    bench = measure_processing_time(mock_filter, arr, multiplier=3.0, num_trials=10, warm_up=3)

    # 3 warmups + 10 trials = 13 total calls
    assert len(calls) == 13
    assert bench["num_trials"] == 10
    assert bench["warm_up"] == 3
    assert bench["median_ms"] >= 0.5  # Slept 1ms
    assert bench["mean_ms"] >= 0.5
    assert bench["median_seconds"] == bench["median_ms"] / 1000.0


def test_measure_processing_time_invalid_parameters():
    """Invalid callable or non-positive trial counts must raise appropriate errors."""
    with pytest.raises(TypeError, match="must be callable"):
        measure_processing_time("not_a_callable")

    with pytest.raises(ValueError, match="num_trials must be an integer >= 1"):
        measure_processing_time(lambda: None, num_trials=0)

    with pytest.raises(ValueError, match="warm_up must be an integer >= 0"):
        measure_processing_time(lambda: None, warm_up=-1)


def test_measure_processing_time_exception_propagation():
    """If the target callable raises an exception, it must propagate directly."""
    def broken_filter():
        raise RuntimeError("Filter crashed")

    with pytest.raises(RuntimeError, match="Filter crashed"):
        measure_processing_time(broken_filter)
# -----------------------------------------------------------------------------
# 4. EDGE CASE AUDIT & DEFECT VERIFICATION
# -----------------------------------------------------------------------------

def test_metrics_rejects_complex_valued_arrays():
    """Signals containing complex values must raise TypeError rather than silently truncating imag."""
    ref = np.array([1.0 + 0.1j, 2.0 - 0.2j])
    sig = np.array([1.0, 2.0])
    with pytest.raises(TypeError, match="complex values"):
        calculate_snr(ref, sig)

    with pytest.raises(TypeError, match="complex values"):
        calculate_rmse(sig, ref)


def test_metrics_rejects_non_numeric_and_boolean_arrays():
    """Boolean arrays and string/object arrays must raise TypeError."""
    bool_arr = np.array([True, False, True])
    with pytest.raises(TypeError, match="boolean dtype"):
        calculate_snr(bool_arr, bool_arr)

    obj_arr = np.array(["0.5", "1.2"], dtype=object)
    float_arr = np.array([0.5, 1.2])
    with pytest.raises(TypeError, match="non-numeric"):
        calculate_snr(obj_arr, float_arr)


def test_measure_processing_time_rejects_boolean_parameters():
    """num_trials and warm_up must reject boolean values explicitly."""
    with pytest.raises(TypeError, match="cannot be a boolean"):
        measure_processing_time(lambda x: x, np.ones(10), num_trials=True)

    with pytest.raises(TypeError, match="cannot be a boolean"):
        measure_processing_time(lambda x: x, np.ones(10), warm_up=False)


def test_evaluate_audio_quality_both_infinite_snr():
    """When both input and output are identical to clean reference (perfect reconstruction), Delta SNR is 0.0 dB."""
    clean = np.array([0.5, -0.5, 0.2, -0.2])
    res = evaluate_audio_quality(clean, clean.copy(), clean.copy())
    assert res.input_snr_db == float("inf")
    assert res.output_snr_db == float("inf")
    assert res.snr_improvement_db == 0.0
    assert res.input_rmse == 0.0
    assert res.output_rmse == 0.0
    assert res.rmse_reduction_percent == 0.0


def test_snr_per_channel_with_zero_energy_channel():
    """In stereo audio where one channel has zero signal energy, verify correct per-channel handling."""
    # Channel 0: Active signal + small noise -> positive finite SNR
    # Channel 1: Pure silence + noise -> SNR is mathematically -inf
    ch0_ref = np.array([1.0, -1.0, 1.0, -1.0])
    ch0_noise = np.array([0.1, -0.1, 0.1, -0.1])

    ch1_ref = np.array([0.0, 0.0, 0.0, 0.0])
    ch1_noise = np.array([0.1, -0.1, 0.1, -0.1])

    ref_stereo = np.column_stack([ch0_ref, ch1_ref])
    noisy_stereo = np.column_stack([ch0_ref + ch0_noise, ch1_ref + ch1_noise])

    # Overall SNR is computed over total pooled energy across all channels
    overall, ch_snrs = calculate_snr(ref_stereo, noisy_stereo, per_channel=True)
    assert not np.isnan(overall)
    assert overall > 0.0
    assert np.isclose(ch_snrs[0], 20.0, atol=1e-10)
    assert ch_snrs[1] == float("-inf")
