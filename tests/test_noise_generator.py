"""
Automated Unit Tests for Noise Generation Engine (src/noise_generator.py)
Tests statistical properties of WGN, SNR calibration accuracy, 50 Hz hum frequency validation via FFT,
harmonic placement, Nyquist enforcement, and composite noise safety.
"""

import numpy as np
import pytest
from scipy.fft import rfft, rfftfreq

from src.noise_generator import (
    add_high_frequency_noise,
    add_noise_combination,
    add_sinusoidal_hum,
    add_white_gaussian_noise,
)


def test_white_gaussian_noise_mean_and_power():
    """Verify generated white noise has approximately zero mean and correct theoretical variance."""
    sr = 44100
    duration = 2.0  # Large sample size for statistical stability
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    clean_signal = 0.8 * np.sin(2 * np.pi * 500.0 * t)

    p_signal = float(np.mean(clean_signal**2))
    target_snr_db = 15.0
    expected_p_noise = p_signal / (10.0 ** (target_snr_db / 10.0))

    noisy, noise = add_white_gaussian_noise(clean_signal, target_snr_db=target_snr_db, seed=42)

    # 1. Check zero mean: sample mean should be near zero (|mean| < 0.005)
    sample_mean = float(np.mean(noise))
    assert abs(sample_mean) < 0.005

    # 2. Check noise power
    sample_p_noise = float(np.mean(noise**2))
    assert np.isclose(sample_p_noise, expected_p_noise, rtol=0.05)


def test_achieved_snr_calibration():
    """Verify achieved SNR is within 0.2 dB of target across multiple SNR benchmarks."""
    sr = 48000
    duration = 3.0  # 144,000 samples for high statistical precision
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Multitone test signal
    clean_signal = 0.5 * np.sin(2 * np.pi * 400.0 * t) + 0.3 * np.cos(2 * np.pi * 1200.0 * t)
    p_signal = float(np.mean(clean_signal**2))

    for target_snr in [0.0, 6.0, 10.0, 20.0]:
        noisy, noise = add_white_gaussian_noise(clean_signal, target_snr_db=target_snr, seed=123)
        p_noise = float(np.mean(noise**2))
        achieved_snr = 10.0 * np.log10(p_signal / p_noise)

        discrepancy = abs(achieved_snr - target_snr)
        assert discrepancy <= 0.20, (
            f"Target SNR: {target_snr} dB, Achieved: {achieved_snr:.3f} dB "
            f"(discrepancy {discrepancy:.3f} dB exceeds 0.20 dB tolerance)"
        )


def test_wgn_zero_power_signal_rejected():
    """Verify adding WGN to a silent signal raises ValueError."""
    silent = np.zeros(1000)
    with pytest.raises(ValueError, match="Cannot calibrate SNR on a silent"):
        add_white_gaussian_noise(silent, target_snr_db=10.0)


def test_sinusoidal_hum_frequency_spectral_accuracy():
    """Verify generated 50 Hz hum has its primary spectral energy peak precisely at 50 Hz via FFT."""
    sr = 8000  # 8 kHz sample rate
    duration = 2.0
    num_samples = int(sr * duration)
    clean = np.zeros(num_samples)

    hum_freq = 50.0
    noisy, hum = add_sinusoidal_hum(clean, sample_rate=sr, freq=hum_freq, amplitude=0.5)

    # Compute FFT to verify peak location
    yf = np.abs(rfft(hum))
    xf = rfftfreq(num_samples, 1.0 / sr)

    peak_idx = int(np.argmax(yf))
    peak_freq = xf[peak_idx]

    # Peak must align exactly with 50 Hz (within frequency resolution bin df = sr / N = 0.5 Hz)
    assert abs(peak_freq - hum_freq) < 0.5


def test_sinusoidal_hum_harmonics_spectral_peaks():
    """Verify 50 Hz hum with harmonics [2, 3] has distinct spectral peaks at 50, 100, and 150 Hz."""
    sr = 8000
    duration = 4.0  # 0.25 Hz bin resolution
    num_samples = int(sr * duration)
    clean = np.zeros(num_samples)

    noisy, hum = add_sinusoidal_hum(
        clean, sample_rate=sr, freq=50.0, amplitude=0.4, harmonics=[2, 3], harmonic_decay=0.5
    )

    yf = np.abs(rfft(hum))
    xf = rfftfreq(num_samples, 1.0 / sr)

    for target_freq in [50.0, 100.0, 150.0]:
        # Locate closest frequency bin
        bin_idx = int(np.argmin(np.abs(xf - target_freq)))
        # Verify local maximum around bin_idx
        local_window = yf[max(0, bin_idx - 2) : bin_idx + 3]
        assert yf[bin_idx] == np.max(local_window)
        assert yf[bin_idx] > 10.0  # Significant peak magnitude


def test_high_frequency_noise_nyquist_enforcement():
    """Verify high-frequency noise generator rejects frequencies at or above the Nyquist limit."""
    sr = 16000
    nyquist = sr / 2.0  # 8000 Hz
    clean = np.zeros(1000)

    # Valid: 7000 Hz (< 8000 Hz)
    noisy, hf = add_high_frequency_noise(clean, sample_rate=sr, freq=7000.0, amplitude=0.2)
    assert hf.shape == clean.shape

    # Invalid: exactly Nyquist (8000 Hz)
    with pytest.raises(ValueError, match="Nyquist"):
        add_high_frequency_noise(clean, sample_rate=sr, freq=8000.0)

    # Invalid: above Nyquist (9000 Hz)
    with pytest.raises(ValueError, match="Nyquist"):
        add_high_frequency_noise(clean, sample_rate=sr, freq=9000.0)

    # Invalid: non-positive frequency
    with pytest.raises(ValueError):
        add_high_frequency_noise(clean, sample_rate=sr, freq=0.0)


def test_noise_combination_and_clipping_prevention():
    """Verify combining multiple noise sources scales signal properly to avoid clipping when requested."""
    signal = 0.8 * np.ones(100)
    noise1 = 0.5 * np.ones(100)
    noise2 = 0.4 * np.ones(100)

    # Without clipping prevention, 0.8 + 0.5 + 0.4 = 1.7 > 1.0
    noisy_scaled, total_noise = add_noise_combination(
        signal, [noise1, noise2], prevent_clipping=True
    )

    # Scaled signal must not exceed 1.0
    assert np.max(np.abs(noisy_scaled)) <= 1.0
    assert np.isclose(np.max(np.abs(noisy_scaled)), 1.0)
