"""
AudioClean DSP - Unit Tests for FIR vs IIR Comparison Engine (src/comparison.py)
Tests objective comparative benchmarking, causal delay alignment, zero-phase mode,
multichannel stereo inputs, different signal lengths, invalid configurations, and timing statistics.
"""

import numpy as np
import pytest

from src.comparison import (
    FilterComparisonReport,
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


def test_align_causal_signal_mono_and_stereo():
    """Verify sample shift and trailing edge zero padding in align_causal_signal."""
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

    # Delay >= length raises ValueError
    with pytest.raises(ValueError, match="must be less than signal length"):
        align_causal_signal(sig_mono, 5)


def test_compare_fir_and_iir_zero_phase_mode(clean_and_noisy_audio):
    """Test side-by-side comparison in zero-phase mode."""
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
    assert report.speedup_ratio > 0.0


def test_compare_fir_and_iir_causal_mode_with_delay_alignment(clean_and_noisy_audio):
    """Test causal filtering with FIR delay compensation enabled."""
    clean, noisy, sr = clean_and_noisy_audio

    fir_d = design_fir_filter(sr, "lowpass", cutoff_hz=1500.0, numtaps=41, window="hamming")
    iir_d = design_iir_filter(sr, "lowpass", cutoff_hz=1500.0, order=4)

    report_aligned, _, _ = compare_fir_and_iir(
        clean, noisy, fir_d, iir_d, mode="causal", align_causal_fir=True, timing_trials=5, timing_warmup=1
    )
    report_unaligned, _, _ = compare_fir_and_iir(
        clean, noisy, fir_d, iir_d, mode="causal", align_causal_fir=False, timing_trials=5, timing_warmup=1
    )

    assert report_aligned.align_causal_delay is True
    assert report_unaligned.align_causal_delay is False

    # Delay alignment significantly improves reference-based SNR for causal FIR
    assert (
        report_aligned.fir_benchmark.snr_improvement_db >
        report_unaligned.fir_benchmark.snr_improvement_db
    )


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


def test_compare_fir_and_iir_invalid_inputs():
    """Verify rigorous exception raising on invalid inputs."""
    sr = 16000
    clean = np.ones(1000)
    noisy = np.ones(1000)

    fir_16k = design_fir_filter(16000, "lowpass", cutoff_hz=2000.0, numtaps=31)
    iir_8k = design_iir_filter(8000, "lowpass", cutoff_hz=2000.0, order=4)
    iir_16k = design_iir_filter(16000, "lowpass", cutoff_hz=2000.0, order=4)

    # 1. Type validation
    with pytest.raises(TypeError, match="fir_design must be FIRFilterDesign"):
        compare_fir_and_iir(clean, noisy, "not_fir", iir_16k)

    with pytest.raises(TypeError, match="iir_design must be IIRFilterDesign"):
        compare_fir_and_iir(clean, noisy, fir_16k, "not_iir")

    # 2. Sample rate mismatch between filters
    with pytest.raises(ValueError, match="Sampling rate mismatch"):
        compare_fir_and_iir(clean, noisy, fir_16k, iir_8k)

    # 3. Explicit sample rate mismatch with filters
    with pytest.raises(ValueError, match="Specified sample_rate"):
        compare_fir_and_iir(clean, noisy, fir_16k, iir_16k, sample_rate=8000)

    # 4. Mode validation
    with pytest.raises(ValueError, match="mode must be 'zero_phase' or 'causal'"):
        compare_fir_and_iir(clean, noisy, fir_16k, iir_16k, mode="invalid_mode")

    # 5. Length mismatch between clean and noisy
    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_fir_and_iir(clean, np.ones(900), fir_16k, iir_16k)
