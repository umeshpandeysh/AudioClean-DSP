"""
AudioClean DSP - Unit Tests for FIR vs IIR Comparison Engine (src/comparison.py)
Tests objective comparative benchmarking, causal delay interval alignment, zero-phase mode,
multichannel stereo inputs, different signal lengths, topology rejection, frequency response data,
speedup calculation, and timing statistics.
"""

import numpy as np
import pytest

from src.comparison import (
    FilterComparisonReport,
    FilterFrequencyResponseData,
    SingleFilterBenchmark,
    align_causal_signal,
    compare_fir_and_iir,
)
from src.fir_filters import design_fir_filter
from src.iir_filters import design_iir_filter, design_iir_notch
from src.noise_generator import add_high_frequency_noise, add_white_gaussian_noise


@pytest.fixture
def clean_and_noisy_audio():
    """Deterministic synthetic audio fixture with known frequency components."""
    sr = 16000
    duration = 0.5
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # In-band signal: 400 Hz tone
    clean = 0.7 * np.sin(2 * np.pi * 400.0 * t)
    # Add out-of-band 3500 Hz noise and WGN
    noisy, _ = add_high_frequency_noise(clean, sr, freq=3500.0, amplitude=0.3)
    noisy, _ = add_white_gaussian_noise(noisy, target_snr_db=12.0, seed=42)
    return clean, noisy, sr


def test_align_causal_signal_validation_and_behavior():
    """Verify sample shift, trailing edge zero padding, and comprehensive parameter validation."""
    # Mono
    sig_mono = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    aligned_mono = align_causal_signal(sig_mono, delay_samples=2)
    expected_mono = np.array([3.0, 4.0, 5.0, 0.0, 0.0])
    assert np.allclose(aligned_mono, expected_mono)

    # Stereo
    sig_stereo = np.column_stack([sig_mono, sig_mono * 2])
    aligned_stereo = align_causal_signal(sig_stereo, delay_samples=1)
    expected_stereo = np.column_stack([
        np.array([2.0, 3.0, 4.0, 5.0, 0.0]),
        np.array([4.0, 6.0, 8.0, 10.0, 0.0]),
    ])
    assert np.allclose(aligned_stereo, expected_stereo)

    # Zero delay returns copy
    assert np.allclose(align_causal_signal(sig_mono, 0), sig_mono)

    # Negative delay raises ValueError
    with pytest.raises(ValueError, match="must be non-negative"):
        align_causal_signal(sig_mono, -1)

    # Boolean delay raises TypeError
    with pytest.raises(TypeError, match="must be an integer"):
        align_causal_signal(sig_mono, True)

    # Float delay raises TypeError
    with pytest.raises(TypeError, match="must be an integer"):
        align_causal_signal(sig_mono, 2.5)

    # Delay >= length raises ValueError
    with pytest.raises(ValueError, match="must be strictly less than signal length"):
        align_causal_signal(sig_mono, 5)

    # Non-ndarray raises TypeError
    with pytest.raises(TypeError, match="must be a numpy.ndarray"):
        align_causal_signal([1.0, 2.0, 3.0], 1)

    # Boolean array raises TypeError
    with pytest.raises(TypeError, match="must be a numeric"):
        align_causal_signal(np.array([True, False, True]), 1)

    # Complex array raises TypeError
    with pytest.raises(TypeError, match="must be a numeric"):
        align_causal_signal(np.array([1.0 + 1.0j, 2.0]), 1)

    # 3D array raises ValueError
    with pytest.raises(ValueError, match="must be a 1D .* or 2D"):
        align_causal_signal(np.ones((5, 2, 2)), 1)

    # 2D array with 3 channels raises ValueError
    with pytest.raises(ValueError, match="must have 1 or 2 channels"):
        align_causal_signal(np.ones((5, 3)), 1)

    # Empty array raises ValueError
    with pytest.raises(ValueError, match="cannot be empty"):
        align_causal_signal(np.array([]), 0)


def test_compare_fir_and_iir_zero_phase_mode(clean_and_noisy_audio):
    """Test side-by-side comparison in zero-phase mode including frequency response."""
    clean, noisy, sr = clean_and_noisy_audio

    fir_d = design_fir_filter(sr, "lowpass", cutoff_hz=1200.0, numtaps=51, window="hamming")
    iir_d = design_iir_filter(sr, "lowpass", cutoff_hz=1200.0, order=4)

    report, fir_out, iir_out = compare_fir_and_iir(
        clean, noisy, fir_d, iir_d, sample_rate=sr, mode="zero_phase", timing_trials=10, timing_warmup=2
    )

    assert isinstance(report, FilterComparisonReport)
    assert isinstance(report.fir_benchmark, SingleFilterBenchmark)
    assert isinstance(report.iir_benchmark, SingleFilterBenchmark)
    assert report.mode == "zero_phase"
    assert report.evaluation_interval == (0, len(clean))
    assert report.evaluation_sample_count == len(clean)

    # Full-length outputs preserved
    assert len(fir_out) == len(clean)
    assert len(iir_out) == len(clean)

    # Both lowpass filters should suppress the 3500 Hz noise and improve SNR
    assert report.fir_benchmark.snr_improvement_db > 0.0
    assert report.iir_benchmark.snr_improvement_db > 0.0
    assert report.fir_benchmark.output_rmse < report.fir_benchmark.input_rmse
    assert report.iir_benchmark.output_rmse < report.iir_benchmark.input_rmse

    # Timing fields verified
    assert report.fir_benchmark.median_processing_time_ms > 0.0
    assert report.iir_benchmark.median_processing_time_ms > 0.0
    assert report.speedup_ratio is not None and report.speedup_ratio > 0.0

    # Frequency response verified
    freq_resp = report.frequency_response
    assert isinstance(freq_resp, FilterFrequencyResponseData)
    assert len(freq_resp.frequencies_hz) == 2048
    assert freq_resp.frequencies_hz[0] == 0.0
    assert freq_resp.frequencies_hz[-1] == sr / 2.0
    assert len(freq_resp.fir_magnitude) == 2048
    assert len(freq_resp.iir_magnitude) == 2048
    assert len(freq_resp.fir_phase_radians) == 2048
    assert len(freq_resp.iir_phase_radians) == 2048


def test_compare_fir_and_iir_causal_mode_exact_sample_correspondence():
    """
    Verify exact sample correspondence in causal mode:
    Ensures that FIR output segment [2*tau : N] is evaluated against reference[tau : N - tau],
    and baseline noisy and IIR are evaluated over the exact same interval [tau : N - tau].
    """
    sr = 8000
    N = 1000
    t = np.linspace(0, N / sr, N, endpoint=False)
    # Simple low-frequency test sinusoid
    clean = np.sin(2 * np.pi * 100.0 * t)
    noisy = clean + 0.05 * np.sin(2 * np.pi * 3000.0 * t)

    fir_d = design_fir_filter(sr, "lowpass", cutoff_hz=500.0, numtaps=31, window="hamming")
    iir_d = design_iir_filter(sr, "lowpass", cutoff_hz=500.0, order=4)

    tau = int(round(fir_d.group_delay_samples))  # (31 - 1) / 2 = 15 samples
    assert tau == 15

    report, fir_out, iir_out = compare_fir_and_iir(
        clean, noisy, fir_d, iir_d, mode="causal", timing_trials=5, timing_warmup=1
    )

    # Full output length preserved
    assert len(fir_out) == N
    assert len(iir_out) == N

    # Evaluation interval must be [tau, N - tau] with N - 2*tau samples
    assert report.evaluation_interval == (tau, N - tau)
    assert report.evaluation_sample_count == N - 2 * tau

    # Verify steady-state correspondence:
    fir_steady_segment = fir_out[2 * tau :]
    clean_segment = clean[tau : N - tau]
    assert len(fir_steady_segment) == N - 2 * tau
    assert len(clean_segment) == N - 2 * tau
    steady_state_error = np.max(np.abs(fir_steady_segment[50:-50] - clean_segment[50:-50]))
    assert steady_state_error < 0.05

    # Causal IIR phase distortion: because IIR has non-linear group delay, it cannot be aligned by a single shift
    assert report.iir_benchmark.group_delay_samples is None


def test_compare_fir_and_iir_causal_deterministic_slices_and_metrics():
    """
    Deterministic test verifying exact array slices and objective metrics against
    independently calculated expected values for causal FIR vs IIR comparison.
    """
    import math
    from src.metrics import evaluate_audio_quality

    sr = 8000
    N = 400
    t = np.linspace(0, N / sr, N, endpoint=False)
    clean = 0.8 * np.sin(2 * np.pi * 150.0 * t)
    noise = 0.15 * np.sin(2 * np.pi * 3200.0 * t) + 0.05 * np.cos(2 * np.pi * 2800.0 * t)
    noisy = clean + noise

    fir_d = design_fir_filter(sr, "lowpass", cutoff_hz=800.0, numtaps=21, window="hamming")
    iir_d = design_iir_filter(sr, "lowpass", cutoff_hz=800.0, order=4)

    tau = int(round(fir_d.group_delay_samples))  # (21 - 1) // 2 = 10
    assert tau == 10
    assert 2 * tau == 20

    report, fir_out, iir_out = compare_fir_and_iir(
        clean, noisy, fir_d, iir_d, mode="causal", timing_trials=3, timing_warmup=1
    )

    # 1. Full length output preservation
    assert fir_out.shape == (N,)
    assert iir_out.shape == (N,)

    # 2. Shared valid reference interval
    expected_interval = (tau, N - tau)  # (10, 390)
    expected_count = N - 2 * tau  # 380
    assert report.evaluation_interval == expected_interval
    assert report.evaluation_sample_count == expected_count

    # 3. Exact manual slices
    eval_ref = clean[tau : N - tau]
    eval_noisy = noisy[tau : N - tau]
    eval_fir = fir_out[2 * tau : N]
    eval_iir = iir_out[tau : N - tau]

    assert len(eval_ref) == expected_count
    assert len(eval_noisy) == expected_count
    assert len(eval_fir) == expected_count
    assert len(eval_iir) == expected_count

    # 4. Independently calculate metrics using exact slices
    exp_fir = evaluate_audio_quality(eval_ref, eval_noisy, eval_fir)
    exp_iir = evaluate_audio_quality(eval_ref, eval_noisy, eval_iir)

    # 5. Assert exact match with report metrics
    assert math.isclose(report.fir_benchmark.input_snr_db, exp_fir.input_snr_db, rel_tol=1e-9)
    assert math.isclose(report.fir_benchmark.output_snr_db, exp_fir.output_snr_db, rel_tol=1e-9)
    assert math.isclose(report.fir_benchmark.snr_improvement_db, exp_fir.snr_improvement_db, rel_tol=1e-9)
    assert math.isclose(report.fir_benchmark.input_rmse, exp_fir.input_rmse, rel_tol=1e-9)
    assert math.isclose(report.fir_benchmark.output_rmse, exp_fir.output_rmse, rel_tol=1e-9)
    assert math.isclose(report.fir_benchmark.rmse_reduction_percent, exp_fir.rmse_reduction_percent, rel_tol=1e-9)
    assert math.isclose(report.fir_benchmark.input_mae, exp_fir.input_mae, rel_tol=1e-9)
    assert math.isclose(report.fir_benchmark.output_mae, exp_fir.output_mae, rel_tol=1e-9)

    assert math.isclose(report.iir_benchmark.input_snr_db, exp_iir.input_snr_db, rel_tol=1e-9)
    assert math.isclose(report.iir_benchmark.output_snr_db, exp_iir.output_snr_db, rel_tol=1e-9)
    assert math.isclose(report.iir_benchmark.snr_improvement_db, exp_iir.snr_improvement_db, rel_tol=1e-9)
    assert math.isclose(report.iir_benchmark.input_rmse, exp_iir.input_rmse, rel_tol=1e-9)
    assert math.isclose(report.iir_benchmark.output_rmse, exp_iir.output_rmse, rel_tol=1e-9)
    assert math.isclose(report.iir_benchmark.rmse_reduction_percent, exp_iir.rmse_reduction_percent, rel_tol=1e-9)

    expected_snr_diff = float(exp_fir.snr_improvement_db - exp_iir.snr_improvement_db)
    assert math.isclose(report.snr_difference_db, expected_snr_diff, rel_tol=1e-9)


def test_compare_fir_and_iir_stereo_audio():
    """Verify multichannel 2D stereo handling."""
    sr = 16000
    N = 4000
    t = np.linspace(0, N / sr, N, endpoint=False)
    clean_l = 0.5 * np.sin(2 * np.pi * 300.0 * t)
    clean_r = 0.5 * np.cos(2 * np.pi * 500.0 * t)
    clean_stereo = np.column_stack([clean_l, clean_r])

    noisy_stereo = clean_stereo + 0.1 * np.random.RandomState(123).randn(*clean_stereo.shape)

    fir_d = design_fir_filter(sr, "lowpass", cutoff_hz=1000.0, numtaps=31)
    iir_d = design_iir_filter(sr, "lowpass", cutoff_hz=1000.0, order=4)

    report, fir_out, iir_out = compare_fir_and_iir(
        clean_stereo, noisy_stereo, fir_d, iir_d, mode="zero_phase", timing_trials=5, timing_warmup=1
    )

    assert fir_out.shape == clean_stereo.shape
    assert iir_out.shape == clean_stereo.shape
    assert report.fir_benchmark.snr_improvement_db > 0.0


def test_compare_fir_and_iir_topology_and_compatibility_rejection():
    """Verify rejection of incompatible filter topologies and mismatched configurations."""
    sr = 16000
    clean = np.ones(1000)
    noisy = np.ones(1000)

    fir_lp = design_fir_filter(16000, "lowpass", cutoff_hz=2000.0, numtaps=31)
    fir_hp = design_fir_filter(16000, "highpass", cutoff_hz=2000.0, numtaps=31)
    iir_lp = design_iir_filter(16000, "lowpass", cutoff_hz=2000.0, order=4)
    iir_hp = design_iir_filter(16000, "highpass", cutoff_hz=2000.0, order=4)
    iir_notch = design_iir_notch(16000, notch_freq=50.0, quality_factor=30.0)

    # 1. Reject incompatible topologies (lowpass vs highpass)
    with pytest.raises(ValueError, match="Incompatible filter topologies"):
        compare_fir_and_iir(clean, noisy, fir_lp, iir_hp)

    with pytest.raises(ValueError, match="Incompatible filter topologies"):
        compare_fir_and_iir(clean, noisy, fir_hp, iir_lp)

    # 2. Reject comparing lowpass with notch
    with pytest.raises(ValueError, match="Incompatible filter topologies"):
        compare_fir_and_iir(clean, noisy, fir_lp, iir_notch)

    # 3. Reject mismatched sample rates
    fir_8k = design_fir_filter(8000, "lowpass", cutoff_hz=2000.0, numtaps=31)
    with pytest.raises(ValueError, match="Sampling rate mismatch"):
        compare_fir_and_iir(clean, noisy, fir_8k, iir_lp)

    # 4. Signal length too short for causal FIR delay
    short_clean = np.ones(10)
    short_noisy = np.ones(10)
    with pytest.raises(ValueError, match="too short to evaluate causal FIR"):
        compare_fir_and_iir(short_clean, short_noisy, fir_lp, iir_lp, mode="causal")


def test_compare_fir_and_iir_speedup_ratio_zero_latency():
    """Verify that speedup ratio handles zero or non-positive latencies by returning None rather than 1.0."""
    clean = np.ones(1000)
    noisy = np.ones(1000)
    fir_d = design_fir_filter(16000, "lowpass", cutoff_hz=2000.0, numtaps=31)
    iir_d = design_iir_filter(16000, "lowpass", cutoff_hz=2000.0, order=4)

    # Mock measure_processing_time with zero latency
    import src.comparison
    orig_measure = src.comparison.measure_processing_time
    try:
        src.comparison.measure_processing_time = lambda *a, **k: {
            "median_seconds": 0.0, "median_ms": 0.0, "mean_ms": 0.0, "std_ms": 0.0,
            "min_ms": 0.0, "max_ms": 0.0, "num_trials": 1, "warm_up": 0
        }
        report, _, _ = compare_fir_and_iir(clean, noisy, fir_d, iir_d, mode="zero_phase")
        assert report.speedup_ratio is None
    finally:
        src.comparison.measure_processing_time = orig_measure


def test_compare_fir_and_iir_cutoff_compatibility_rejection():
    """Verify rejection of mismatched cutoff frequencies and notch specifications."""
    sr = 16000
    clean = np.ones(1000)
    noisy = np.ones(1000)

    # 1. Lowpass cutoff mismatch (2000 Hz vs 2200 Hz)
    fir_lp_2k = design_fir_filter(sr, "lowpass", cutoff_hz=2000.0, numtaps=31)
    iir_lp_22k = design_iir_filter(sr, "lowpass", cutoff_hz=2200.0, order=4)
    with pytest.raises(ValueError, match="Cutoff frequency mismatch for lowpass"):
        compare_fir_and_iir(clean, noisy, fir_lp_2k, iir_lp_22k)

    # 2. Highpass cutoff mismatch (1500 Hz vs 1800 Hz)
    fir_hp_15k = design_fir_filter(sr, "highpass", cutoff_hz=1500.0, numtaps=31)
    iir_hp_18k = design_iir_filter(sr, "highpass", cutoff_hz=1800.0, order=4)
    with pytest.raises(ValueError, match="Cutoff frequency mismatch for highpass"):
        compare_fir_and_iir(clean, noisy, fir_hp_15k, iir_hp_18k)

    # 3. Bandpass cutoff edge mismatch
    fir_bp = design_fir_filter(sr, "bandpass", cutoff_hz=(500.0, 2000.0), numtaps=31)
    iir_bp_diff = design_iir_filter(sr, "bandpass", cutoff_hz=(600.0, 2000.0), order=3)
    with pytest.raises(ValueError, match="Bandpass cutoff mismatch"):
        compare_fir_and_iir(clean, noisy, fir_bp, iir_bp_diff)

    # 4. Bandstop cutoff edge mismatch
    fir_bs = design_fir_filter(sr, "bandstop", cutoff_hz=(500.0, 2000.0), numtaps=31)
    iir_bs_diff = design_iir_filter(sr, "bandstop", cutoff_hz=(500.0, 2100.0), order=3)
    with pytest.raises(ValueError, match="Bandstop cutoff mismatch"):
        compare_fir_and_iir(clean, noisy, fir_bs, iir_bs_diff)

    # 5. Notch vs Bandstop mismatch
    iir_notch = design_iir_notch(sr, notch_freq=50.0, quality_factor=30.0)
    fir_bs_mismatch = design_fir_filter(sr, "bandstop", cutoff_hz=(40.0, 60.0), numtaps=31)
    with pytest.raises(ValueError, match="Notch vs FIR bandstop specification mismatch"):
        compare_fir_and_iir(clean, noisy, fir_bs_mismatch, iir_notch)

    # 6. Matching Notch vs Bandstop succeeds
    # f0 = 100 Hz, Q = 10 -> BW = 10 Hz -> edges = (95.0, 105.0)
    iir_notch_100 = design_iir_notch(sr, notch_freq=100.0, quality_factor=10.0)
    fir_bs_match = design_fir_filter(sr, "bandstop", cutoff_hz=(95.0, 105.0), numtaps=31)
    report, _, _ = compare_fir_and_iir(clean, noisy, fir_bs_match, iir_notch_100, mode="zero_phase")
    assert report.sample_rate == sr


def test_compare_fir_and_iir_frequency_grid_validation():
    """Verify that compare_fir_and_iir validates frequency response grids."""
    import src.comparison

    clean = np.ones(1000)
    noisy = np.ones(1000)
    fir_lp = design_fir_filter(16000, "lowpass", cutoff_hz=2000.0, numtaps=31)
    iir_lp = design_iir_filter(16000, "lowpass", cutoff_hz=2000.0, order=4)

    orig_get_iir_resp = src.comparison.get_iir_frequency_response

    # Test corrupted frequency grid (NaN)
    try:
        src.comparison.get_iir_frequency_response = lambda design, worN: (
            np.full(worN, np.nan), np.ones(worN), np.ones(worN), np.zeros(worN)
        )
        with pytest.raises(ValueError, match="grid contains non-finite"):
            compare_fir_and_iir(clean, noisy, fir_lp, iir_lp)
    finally:
        src.comparison.get_iir_frequency_response = orig_get_iir_resp

    # Test mismatched grid values
    try:
        src.comparison.get_iir_frequency_response = lambda design, worN: (
            np.linspace(0, 4000, worN), np.ones(worN), np.ones(worN), np.zeros(worN)
        )
        with pytest.raises(ValueError, match="Frequency response grid mismatch"):
            compare_fir_and_iir(clean, noisy, fir_lp, iir_lp)
    finally:
        src.comparison.get_iir_frequency_response = orig_get_iir_resp

