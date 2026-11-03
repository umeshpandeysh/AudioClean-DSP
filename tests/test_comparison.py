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
    """Verify sample shift, trailing edge zero padding, and parameter validation."""
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
    Ensures that FIR output segment [tau : N] is evaluated against reference[0 : N - tau],
    and baseline noisy and IIR are evaluated over the exact same interval [0 : N - tau].
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

    # Evaluation interval must be [0, N - tau]
    assert report.evaluation_interval == (0, N - tau)
    assert report.evaluation_sample_count == N - tau

    # Verify sample correspondence:
    # fir_out[tau:] aligns with clean[:N-tau] with high precision for an in-band tone (100 Hz << 500 Hz)
    fir_aligned_segment = fir_out[tau:]
    clean_segment = clean[: N - tau]
    # The max absolute error in steady state between aligned FIR and clean tone is minimal
    steady_state_error = np.max(np.abs(fir_aligned_segment[50:-50] - clean_segment[50:-50]))
    assert steady_state_error < 0.05

    # Causal IIR phase distortion: because IIR has non-linear group delay, it cannot be aligned by a single shift
    assert report.iir_benchmark.group_delay_samples is None


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
