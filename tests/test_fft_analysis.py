"""
Automated Unit Tests for FFT and Spectrogram Analysis Module (src/fft_analysis.py)
Tests discrete Fourier transform frequency bins, single-sided magnitude scaling, DC handling,
windowing, zero-padding, STFT spectrogram dimensions, and validation edge cases.
"""

import numpy as np
import pytest

from src.fft_analysis import compute_fft, compute_spectrogram


# =====================================================================
# 1. FFT SPECTRUM TESTS
# =====================================================================

def test_fft_frequency_bins_and_resolution():
    """Verify compute_fft returns correct frequency axis and resolution df = fs / N."""
    sr = 8000
    N = 1000
    t = np.linspace(0, N / sr, N, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 500.0 * t)

    freqs, mag, mag_db = compute_fft(sig, sample_rate=sr, window="rectangular")

    expected_num_bins = N // 2 + 1  # 501 bins for real FFT
    assert len(freqs) == expected_num_bins
    assert len(mag) == expected_num_bins
    assert len(mag_db) == expected_num_bins
    assert freqs[0] == 0.0
    assert np.isclose(freqs[-1], sr / 2.0)

    # Resolution df = sr / N = 8000 / 1000 = 8.0 Hz
    df = freqs[1] - freqs[0]
    assert np.isclose(df, 8.0)


def test_fft_dominant_frequency_detection():
    """Verify FFT peak locates a known 400 Hz tone exactly on its frequency bin."""
    sr = 8000
    duration = 1.0  # 1 Hz resolution
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.7 * np.sin(2 * np.pi * 400.0 * t)

    freqs, mag, mag_db = compute_fft(tone, sample_rate=sr, window="hann")

    peak_idx = int(np.argmax(mag))
    detected_freq = freqs[peak_idx]

    assert abs(detected_freq - 400.0) < 1.0


def test_fft_single_sided_magnitude_scaling():
    """Verify single-sided spectrum scaling calibrates peak amplitude to match sinusoid amplitude A."""
    sr = 8000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    amplitude = 0.65
    # Choose integer frequency 200 Hz exactly aligned with 1 Hz bins
    tone = amplitude * np.sin(2 * np.pi * 200.0 * t)

    freqs, mag, _ = compute_fft(tone, sample_rate=sr, window="rectangular")
    peak_idx = int(np.argmax(mag))

    # For rectangular window with exact bin alignment, peak magnitude matches amplitude
    assert np.isclose(mag[peak_idx], amplitude, atol=1e-3)


def test_fft_dc_handling_not_doubled():
    """Verify DC constant component (f = 0 Hz) has correct magnitude and is not doubled."""
    sr = 4000
    sig = np.full(500, 0.4)  # Constant DC of 0.4

    freqs, mag, _ = compute_fft(sig, sample_rate=sr, window="rectangular")

    # DC bin at index 0 must be 0.4, NOT 0.8
    assert np.isclose(mag[0], 0.4, atol=1e-3)
    # AC bins must be near zero
    assert np.max(mag[1:]) < 1e-10


def test_fft_silent_signal_safe():
    """Verify silent (all-zero) signal computes cleanly without NaNs or zero-division."""
    silent = np.zeros(1000)
    freqs, mag, mag_db = compute_fft(silent, sample_rate=8000)

    assert np.all(mag == 0.0)
    assert not np.any(np.isnan(mag_db))
    assert np.all(mag_db <= -100.0)


def test_fft_zero_padding_interpolates_spectrum():
    """Verify zero-padding to larger n_fft yields higher density frequency bins."""
    sr = 8000
    sig = np.sin(2 * np.pi * 300.0 * np.linspace(0, 0.1, 800, endpoint=False))

    # Request n_fft = 2048 (> 800)
    freqs, mag, _ = compute_fft(sig, sample_rate=sr, n_fft=2048)

    assert len(freqs) == 2048 // 2 + 1  # 1025 bins
    assert np.isclose(freqs[1] - freqs[0], sr / 2048.0)


def test_fft_stereo_downmixes_properly():
    """Verify passing a 2D stereo signal computes FFT cleanly without shape errors."""
    stereo = np.ones((500, 2)) * 0.3
    freqs, mag, _ = compute_fft(stereo, sample_rate=4000, window="rectangular")
    assert np.isclose(mag[0], 0.3, atol=1e-3)


# =====================================================================
# 2. SPECTROGRAM TESTS
# =====================================================================

def test_spectrogram_axes_and_dimensions():
    """Verify spectrogram returns valid time and frequency axes with correct matrix shapes."""
    sr = 8000
    duration = 1.0
    num_samples = int(sr * duration)
    t_sig = np.linspace(0, duration, num_samples, endpoint=False)
    sig = np.sin(2 * np.pi * 500.0 * t_sig)

    nperseg = 512
    times, freqs, Sxx_db = compute_spectrogram(sig, sample_rate=sr, nperseg=nperseg)

    expected_freq_bins = nperseg // 2 + 1
    assert len(freqs) == expected_freq_bins
    assert Sxx_db.shape[0] == expected_freq_bins
    assert Sxx_db.shape[1] == len(times)
    assert freqs[0] == 0.0
    assert np.isclose(freqs[-1], sr / 2.0)
    assert times[0] >= 0.0
    assert times[-1] <= duration


def test_spectrogram_50hz_tonal_detection():
    """Verify spectrogram concentrates power at 50 Hz for an electrical hum signal."""
    sr = 8000
    duration = 2.0
    t_sig = np.linspace(0, duration, int(sr * duration), endpoint=False)
    hum = np.sin(2 * np.pi * 50.0 * t_sig)

    times, freqs, Sxx_db = compute_spectrogram(hum, sample_rate=sr, nperseg=1024)

    # Locate frequency bin closest to 50 Hz
    bin_50 = int(np.argmin(np.abs(freqs - 50.0)))

    # Mean energy across time at 50 Hz must be substantially higher than at 500 Hz
    bin_500 = int(np.argmin(np.abs(freqs - 500.0)))
    power_50 = np.mean(Sxx_db[bin_50, :])
    power_500 = np.mean(Sxx_db[bin_500, :])

    assert power_50 - power_500 > 30.0  # At least 30 dB higher at 50 Hz


# =====================================================================
# 3. VALIDATION & ERROR HANDLING TESTS
# =====================================================================

def test_fft_validation_errors():
    """Verify parameter validations in compute_fft."""
    sig = np.ones(100)

    with pytest.raises(ValueError, match="strictly positive"):
        compute_fft(sig, sample_rate=0)

    with pytest.raises(ValueError, match="Unsupported window"):
        compute_fft(sig, sample_rate=8000, window="unsupported_window")

    with pytest.raises(ValueError, match="n_fft must be a strictly positive integer"):
        compute_fft(sig, sample_rate=8000, n_fft=-10)


def test_spectrogram_validation_errors():
    """Verify parameter validations in compute_spectrogram."""
    sig = np.ones(500)

    # nperseg > signal length
    with pytest.raises(ValueError, match="too short for spectrogram"):
        compute_spectrogram(sig, sample_rate=8000, nperseg=1024)

    # noverlap >= nperseg
    with pytest.raises(ValueError, match="noverlap"):
        compute_spectrogram(sig, sample_rate=8000, nperseg=256, noverlap=256)

    # invalid scaling
    with pytest.raises(ValueError, match="Unsupported scaling"):
        compute_spectrogram(sig, sample_rate=8000, nperseg=256, scaling="invalid")


def test_fft_odd_length_handling():
    """Verify compute_fft handles odd signal lengths without indexing errors."""
    sr = 8000
    N = 1001  # Odd length
    t = np.linspace(0, N / sr, N, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 300.0 * t)

    freqs, mag, mag_db = compute_fft(sig, sample_rate=sr, window="rectangular")
    expected_bins = (N + 1) // 2  # 501 bins
    assert len(freqs) == expected_bins
    assert len(mag) == expected_bins
    assert len(mag_db) == expected_bins
    assert not np.any(np.isnan(mag))
    assert not np.any(np.isnan(mag_db))


def test_fft_non_bin_centered_sinusoid():
    """Verify spectral peak detection for non-bin-centered sinusoid."""
    sr = 8000
    N = 1000
    t = np.linspace(0, N / sr, N, endpoint=False)
    # Bin resolution is 8 Hz. Choose 303.5 Hz which lies between bins (304 Hz is closest)
    sig = 0.8 * np.sin(2 * np.pi * 303.5 * t)

    freqs, mag, _ = compute_fft(sig, sample_rate=sr, window="hann")
    peak_idx = int(np.argmax(mag))
    detected_freq = freqs[peak_idx]

    # Closest bin should be within 1 bin width (8 Hz)
    assert abs(detected_freq - 303.5) <= 8.0


def test_fft_and_spectrogram_channel_selection():
    """Verify explicit channel selection on stereo audio for FFT and spectrogram."""
    sr = 8000
    t = np.linspace(0, 1.0, 8000, endpoint=False)
    ch0 = 0.4 * np.sin(2 * np.pi * 200.0 * t)
    ch1 = 0.9 * np.sin(2 * np.pi * 600.0 * t)
    stereo = np.column_stack([ch0, ch1])

    # Analyze ch0
    freqs0, mag0, _ = compute_fft(stereo, sample_rate=sr, channel=0)
    assert abs(freqs0[int(np.argmax(mag0))] - 200.0) < 1.0

    # Analyze ch1
    freqs1, mag1, _ = compute_fft(stereo, sample_rate=sr, channel=1)
    assert abs(freqs1[int(np.argmax(mag1))] - 600.0) < 1.0

    # Test spectrogram channel selection
    t_axis, f_axis, sxx0 = compute_spectrogram(stereo, sample_rate=sr, channel=0)
    assert sxx0.ndim == 2

    # Invalid channel index
    with pytest.raises(ValueError, match="out of bounds"):
        compute_fft(stereo, sample_rate=sr, channel=3)


def test_fft_truncation_control():
    """Verify n_fft < len(audio) raises ValueError unless allow_truncation=True."""
    sig = np.ones(2000)
    sr = 8000

    with pytest.raises(ValueError, match="smaller than signal length"):
        compute_fft(sig, sample_rate=sr, n_fft=1024, allow_truncation=False)

    # When allow_truncation is True, it succeeds
    freqs, mag, _ = compute_fft(sig, sample_rate=sr, n_fft=1024, allow_truncation=True)
    assert len(freqs) == 1024 // 2 + 1

