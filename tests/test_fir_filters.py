"""
Automated Unit Tests for FIR Digital Filter Engine (src/fir_filters.py)
Tests filter synthesis, Type-I symmetry, frequency response attenuation,
causal group delay, zero-phase filtering, multi-channel handling, and parameter validation.
"""

import numpy as np
import pytest

from src.fir_filters import (
    FIRFilterDesign,
    apply_fir_filter,
    design_fir_filter,
    get_fir_frequency_response,
)


# =====================================================================
# 1. DESIGN TESTS
# =====================================================================

def test_fir_design_coefficients_and_symmetry():
    """Verify FIR design generates expected tap count, finite values, and exact Type-I symmetry."""
    sr = 16000
    numtaps = 65  # Odd tap count
    cutoff = 2000.0

    design = design_fir_filter(
        sample_rate=sr,
        filter_type="lowpass",
        cutoff_hz=cutoff,
        numtaps=numtaps,
        window="hamming",
    )

    assert isinstance(design, FIRFilterDesign)
    assert len(design.coefficients) == numtaps
    assert np.all(np.isfinite(design.coefficients))
    assert design.order == numtaps - 1
    assert design.group_delay_samples == (numtaps - 1) / 2.0
    assert np.isclose(design.group_delay_seconds, ((numtaps - 1) / 2.0) / sr)

    # Check Type-I symmetry: h[n] == h[M - 1 - n]
    assert np.allclose(design.coefficients, design.coefficients[::-1], atol=1e-12)


@pytest.mark.parametrize("filter_type, cutoff", [
    ("lowpass", 1500.0),
    ("highpass", 2500.0),
    ("bandpass", (1000.0, 3000.0)),
    ("bandstop", (1800.0, 2200.0)),
])
def test_all_four_filter_types_can_be_designed(filter_type, cutoff):
    """Verify all 4 canonical FIR filter types synthesize successfully."""
    design = design_fir_filter(
        sample_rate=16000,
        filter_type=filter_type,
        cutoff_hz=cutoff,
        numtaps=51,
        window="hamming",
    )
    assert design.filter_type == filter_type
    assert len(design.coefficients) == 51
    assert np.allclose(design.coefficients, design.coefficients[::-1], atol=1e-12)


@pytest.mark.parametrize("window", ["hamming", "hann", "blackman"])
def test_all_supported_windows_work(window):
    """Verify all 3 supported window topologies synthesize symmetric coefficients."""
    design = design_fir_filter(
        sample_rate=8000,
        filter_type="lowpass",
        cutoff_hz=1000.0,
        numtaps=41,
        window=window,
    )
    assert design.window == window
    assert len(design.coefficients) == 41
    assert np.allclose(design.coefficients, design.coefficients[::-1], atol=1e-12)


# =====================================================================
# 2. FREQUENCY RESPONSE & ATTENUATION TESTS
# =====================================================================

def test_fir_lowpass_attenuation():
    """Verify lowpass filter preserves passband sinusoid and strongly attenuates stopband sinusoid."""
    sr = 8000
    cutoff = 1000.0
    numtaps = 101  # Sharp transition band
    design = design_fir_filter(sr, "lowpass", cutoff, numtaps, window="blackman")

    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Passband: 300 Hz (< 1000 Hz)
    sig_pass = np.sin(2 * np.pi * 300.0 * t)
    # Stopband: 2500 Hz (>> 1000 Hz)
    sig_stop = np.sin(2 * np.pi * 2500.0 * t)

    # Evaluate in steady-state region away from reflection boundary transients
    steady = slice(3 * numtaps, -3 * numtaps)

    out_pass = apply_fir_filter(sig_pass, design, mode="zero_phase")
    out_stop = apply_fir_filter(sig_stop, design, mode="zero_phase")

    rms_pass_in = np.sqrt(np.mean(sig_pass[steady]**2))
    rms_pass_out = np.sqrt(np.mean(out_pass[steady]**2))
    rms_stop_out = np.sqrt(np.mean(out_stop[steady]**2))

    # Passband signal should have minimal attenuation (> -1 dB)
    assert rms_pass_out / rms_pass_in > 0.90
    # Stopband signal should be strongly attenuated (at least 30 dB down -> ratio < 0.01)
    assert rms_stop_out / rms_pass_in < 0.01


def test_fir_highpass_attenuation():
    """Verify highpass filter strongly attenuates low frequencies and preserves high frequencies."""
    sr = 8000
    cutoff = 2000.0
    numtaps = 101
    design = design_fir_filter(sr, "highpass", cutoff, numtaps, window="blackman")

    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    sig_stop = np.sin(2 * np.pi * 400.0 * t)   # 400 Hz << 2000 Hz
    sig_pass = np.sin(2 * np.pi * 3200.0 * t)  # 3200 Hz >> 2000 Hz

    steady = slice(3 * numtaps, -3 * numtaps)

    out_stop = apply_fir_filter(sig_stop, design, mode="zero_phase")
    out_pass = apply_fir_filter(sig_pass, design, mode="zero_phase")

    rms_pass_in = np.sqrt(np.mean(sig_pass[steady]**2))
    rms_pass_out = np.sqrt(np.mean(out_pass[steady]**2))
    rms_stop_out = np.sqrt(np.mean(out_stop[steady]**2))

    assert rms_pass_out / rms_pass_in > 0.90
    assert rms_stop_out / rms_pass_in < 0.01


def test_fir_bandpass_attenuation():
    """Verify bandpass filter passes in-band frequencies and rejects both lower and higher stopbands."""
    sr = 10000
    band = (1500.0, 3000.0)
    numtaps = 121
    design = design_fir_filter(sr, "bandpass", band, numtaps, window="blackman")

    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    sig_low_stop = np.sin(2 * np.pi * 500.0 * t)   # Lower stopband (500 Hz)
    sig_pass = np.sin(2 * np.pi * 2200.0 * t)       # Passband (2200 Hz)
    sig_high_stop = np.sin(2 * np.pi * 4200.0 * t)  # Upper stopband (4200 Hz)

    steady = slice(3 * numtaps, -3 * numtaps)

    out_low = apply_fir_filter(sig_low_stop, design, mode="zero_phase")
    out_pass = apply_fir_filter(sig_pass, design, mode="zero_phase")
    out_high = apply_fir_filter(sig_high_stop, design, mode="zero_phase")

    rms_in = np.sqrt(np.mean(sig_pass[steady]**2))
    assert np.sqrt(np.mean(out_pass[steady]**2)) / rms_in > 0.90
    assert np.sqrt(np.mean(out_low[steady]**2)) / rms_in < 0.01
    assert np.sqrt(np.mean(out_high[steady]**2)) / rms_in < 0.01


def test_fir_bandstop_attenuation():
    """Verify bandstop filter strongly suppresses frequencies inside its notch band."""
    sr = 8000
    notch_band = (900.0, 1100.0)
    numtaps = 121
    design = design_fir_filter(sr, "bandstop", notch_band, numtaps, window="blackman")

    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    sig_pass_low = np.sin(2 * np.pi * 400.0 * t)    # Passband below (400 Hz)
    sig_stop = np.sin(2 * np.pi * 1000.0 * t)       # Rejected notch (1000 Hz)
    sig_pass_high = np.sin(2 * np.pi * 2500.0 * t)  # Passband above (2500 Hz)

    steady = slice(3 * numtaps, -3 * numtaps)

    out_pass_low = apply_fir_filter(sig_pass_low, design, mode="zero_phase")
    out_stop = apply_fir_filter(sig_stop, design, mode="zero_phase")
    out_pass_high = apply_fir_filter(sig_pass_high, design, mode="zero_phase")

    rms_in = np.sqrt(np.mean(sig_pass_low[steady]**2))
    assert np.sqrt(np.mean(out_pass_low[steady]**2)) / rms_in > 0.90
    assert np.sqrt(np.mean(out_pass_high[steady]**2)) / rms_in > 0.90
    assert np.sqrt(np.mean(out_stop[steady]**2)) / rms_in < 0.01


def test_get_fir_frequency_response():
    """Verify frequency response output shapes, frequency limits, and dB flooring."""
    sr = 8000
    design = design_fir_filter(sr, "lowpass", 1000.0, 51)
    freqs, mag, mag_db, phase = get_fir_frequency_response(design, worN=1024)

    assert len(freqs) == 1024
    assert len(mag) == 1024
    assert len(mag_db) == 1024
    assert len(phase) == 1024
    assert freqs[0] == 0.0
    assert np.isclose(freqs[-1], sr / 2.0)
    # At DC (0 Hz), lowpass magnitude should be approximately 1.0 (0 dB)
    assert np.isclose(mag[0], 1.0, atol=0.05)
    assert np.isclose(mag_db[0], 0.0, atol=0.5)
    # Check that magnitude_db contains no NaNs or -Infs
    assert np.all(np.isfinite(mag_db))


# =====================================================================
# 3. FILTERING BEHAVIOR TESTS (CAUSAL VS ZERO-PHASE)
# =====================================================================

def test_filtering_preserves_length_and_multichannel():
    """Verify filtering preserves sample count for both mono and stereo signals."""
    sr = 16000
    design = design_fir_filter(sr, "lowpass", 2000.0, 45)

    # 1D Mono
    mono_in = np.random.default_rng(0).normal(0, 0.1, 2000)
    mono_causal = apply_fir_filter(mono_in, design, mode="causal")
    mono_zero = apply_fir_filter(mono_in, design, mode="zero_phase")
    assert mono_causal.shape == (2000,)
    assert mono_zero.shape == (2000,)

    # 2D Stereo
    stereo_in = np.random.default_rng(0).normal(0, 0.1, (2000, 2))
    stereo_causal = apply_fir_filter(stereo_in, design, mode="causal")
    stereo_zero = apply_fir_filter(stereo_in, design, mode="zero_phase")
    assert stereo_causal.shape == (2000, 2)
    assert stereo_zero.shape == (2000, 2)


def test_causal_impulse_response_group_delay():
    """Verify causal filtering of an impulse delta[n] produces a peak exactly at the group delay."""
    sr = 8000
    numtaps = 51  # group delay = (51 - 1) / 2 = 25 samples
    design = design_fir_filter(sr, "lowpass", 1000.0, numtaps, window="hamming")

    # Unit impulse: delta[0] = 1, delta[n>0] = 0
    impulse = np.zeros(200)
    impulse[0] = 1.0

    response = apply_fir_filter(impulse, design, mode="causal")

    # Peak of response must be at index 25
    peak_idx = int(np.argmax(np.abs(response)))
    expected_delay = int(design.group_delay_samples)
    assert peak_idx == expected_delay == 25


def test_zero_phase_eliminates_phase_shift():
    """Verify zero-phase filtering aligns passband sinusoid peaks with zero sample delay."""
    sr = 8000
    numtaps = 71
    design = design_fir_filter(sr, "lowpass", 1500.0, numtaps, window="hamming")

    # Passband sinusoid (200 Hz)
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    sig_in = np.sin(2 * np.pi * 200.0 * t)

    sig_out = apply_fir_filter(sig_in, design, mode="zero_phase")

    # Test alignment in settled steady-state region (away from boundaries)
    test_slice = slice(200, 800)
    # Cross correlation maximum between input and zero-phase filtered output must be at lag 0
    corr = np.correlate(sig_out[test_slice], sig_in[test_slice], mode="full")
    max_corr_lag = np.argmax(corr) - (len(sig_in[test_slice]) - 1)
    assert max_corr_lag == 0


# =====================================================================
# 4. PARAMETER VALIDATION & ERROR HANDLING TESTS
# =====================================================================

def test_validation_even_taps_rejected():
    """Verify even tap counts are strictly rejected with an informative error."""
    with pytest.raises(ValueError, match="ODD integer"):
        design_fir_filter(8000, "lowpass", 1000.0, numtaps=50)

    with pytest.raises(ValueError, match="at least 3"):
        design_fir_filter(8000, "lowpass", 1000.0, numtaps=1)


def test_validation_nyquist_and_cutoff_bounds():
    """Verify cutoff frequencies violating the Nyquist bound or non-positive are rejected."""
    sr = 8000  # Nyquist = 4000 Hz

    # At Nyquist
    with pytest.raises(ValueError, match="Nyquist"):
        design_fir_filter(sr, "lowpass", 4000.0, 51)

    # Above Nyquist
    with pytest.raises(ValueError, match="Nyquist"):
        design_fir_filter(sr, "lowpass", 4500.0, 51)

    # Zero or negative
    with pytest.raises(ValueError, match="strictly positive"):
        design_fir_filter(sr, "lowpass", 0.0, 51)
    with pytest.raises(ValueError, match="strictly positive"):
        design_fir_filter(sr, "lowpass", -100.0, 51)


def test_validation_band_edges():
    """Verify band-pass / band-stop edge ordering and count validations."""
    sr = 8000

    # Inverted band edges (f_low >= f_high)
    with pytest.raises(ValueError, match="strictly less"):
        design_fir_filter(sr, "bandpass", [2000.0, 1000.0], 51)

    # Single frequency for bandpass
    with pytest.raises(ValueError, match="requires exactly two"):
        design_fir_filter(sr, "bandpass", [1000.0], 51)

    # Three frequencies for bandpass
    with pytest.raises(ValueError, match="requires exactly two"):
        design_fir_filter(sr, "bandpass", [500.0, 1000.0, 1500.0], 51)


def test_validation_unsupported_types_and_windows():
    """Verify unsupported filter types or window names raise informative ValueErrors."""
    with pytest.raises(ValueError, match="Unsupported filter_type"):
        design_fir_filter(8000, "invalid_type", 1000.0, 51)

    with pytest.raises(ValueError, match="Unsupported window"):
        design_fir_filter(8000, "lowpass", 1000.0, 51, window="kaiser_unsupported")


def test_validation_zero_phase_input_too_short():
    """Verify filtfilt raises an informative ValueError when input is too short for padding."""
    sr = 8000
    numtaps = 51  # padlen = 3 * 51 = 153 samples
    design = design_fir_filter(sr, "lowpass", 1000.0, numtaps)

    # Short signal of only 100 samples (<= 153)
    short_signal = np.ones(100)
    with pytest.raises(ValueError, match="too short for zero-phase filtering"):
        apply_fir_filter(short_signal, design, mode="zero_phase")

    # Causal mode should still succeed on the short signal
    causal_out = apply_fir_filter(short_signal, design, mode="causal")
    assert len(causal_out) == 100
