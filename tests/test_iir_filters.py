"""
Automated Unit Tests for IIR Digital Filter Engine (src/iir_filters.py)
Tests Butterworth synthesis (lowpass, highpass, bandpass, bandstop), IIR notch filters,
SOS representations, frequency attenuation, causal vs. zero-phase filtering,
multichannel support, and validation constraints.
"""

import numpy as np
import pytest

from src.iir_filters import (
    IIRFilterDesign,
    apply_iir_filter,
    check_sos_stability,
    design_iir_filter,
    design_iir_notch,
    get_iir_frequency_response,
)


# =====================================================================
# 1. DESIGN & SOS MATRIX TESTS
# =====================================================================

def test_butterworth_lowpass_design_and_sos():
    """Verify Butterworth lowpass creates valid SOS matrix with finite coefficients."""
    sr = 8000
    order = 4
    cutoff = 1000.0

    design = design_iir_filter(sample_rate=sr, filter_type="lowpass", cutoff_hz=cutoff, order=order)

    assert isinstance(design, IIRFilterDesign)
    assert design.filter_type == "lowpass"
    assert design.order == 4
    assert design.design_method == "butterworth"
    # Order 4 lowpass has ceil(4/2) = 2 second-order sections of shape (2, 6)
    assert design.sos.shape == (2, 6)
    assert np.all(np.isfinite(design.sos))


@pytest.mark.parametrize("filter_type, cutoff", [
    ("lowpass", 1200.0),
    ("highpass", 2400.0),
    ("bandpass", (1000.0, 3000.0)),
    ("bandstop", (1500.0, 2500.0)),
])
def test_all_four_butterworth_topologies(filter_type, cutoff):
    """Verify all 4 Butterworth filter topologies design successfully."""
    design = design_iir_filter(sample_rate=16000, filter_type=filter_type, cutoff_hz=cutoff, order=4)
    assert design.filter_type == filter_type
    assert np.all(np.isfinite(design.sos))
    assert design.sos.ndim == 2
    assert design.sos.shape[1] == 6


def test_iir_notch_design_50hz():
    """Verify 50 Hz notch filter synthesizes a single 2nd-order section (1, 6)."""
    sr = 8000
    design = design_iir_notch(sample_rate=sr, notch_freq=50.0, quality_factor=30.0)

    assert isinstance(design, IIRFilterDesign)
    assert design.filter_type == "notch"
    assert design.design_method == "iirnotch"
    assert design.notch_freq == 50.0
    assert design.quality_factor == 30.0
    assert design.order == 2
    assert design.sos.shape == (1, 6)
    assert np.all(np.isfinite(design.sos))


def test_iir_notch_configurable_frequency():
    """Verify notch filter designs accurately at non-standard frequencies (e.g. 120 Hz)."""
    sr = 16000
    design = design_iir_notch(sample_rate=sr, notch_freq=120.0, quality_factor=25.0)
    assert design.notch_freq == 120.0
    assert design.sos.shape == (1, 6)


# =====================================================================
# 2. FREQUENCY RESPONSE & ATTENUATION TESTS
# =====================================================================

def test_butterworth_lowpass_attenuation():
    """Verify Butterworth lowpass preserves passband and suppresses stopband."""
    sr = 8000
    cutoff = 1000.0
    order = 6  # 36 dB/octave roll-off
    design = design_iir_filter(sr, "lowpass", cutoff, order=order)

    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    sig_pass = np.sin(2 * np.pi * 300.0 * t)   # 300 Hz in passband
    sig_stop = np.sin(2 * np.pi * 2500.0 * t)  # 2500 Hz in stopband

    # Evaluate away from filter startup transients
    steady = slice(200, -200)

    out_pass = apply_iir_filter(sig_pass, design, mode="zero_phase")
    out_stop = apply_iir_filter(sig_stop, design, mode="zero_phase")

    rms_in = np.sqrt(np.mean(sig_pass[steady]**2))
    rms_pass = np.sqrt(np.mean(out_pass[steady]**2))
    rms_stop = np.sqrt(np.mean(out_stop[steady]**2))

    assert rms_pass / rms_in > 0.95
    assert rms_stop / rms_in < 0.001  # > 60 dB attenuation


def test_butterworth_highpass_attenuation():
    """Verify Butterworth highpass attenuates low frequencies and preserves high frequencies."""
    sr = 8000
    cutoff = 2000.0
    order = 6
    design = design_iir_filter(sr, "highpass", cutoff, order=order)

    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    sig_stop = np.sin(2 * np.pi * 400.0 * t)   # 400 Hz stopband
    sig_pass = np.sin(2 * np.pi * 3200.0 * t)  # 3200 Hz passband

    steady = slice(200, -200)
    out_stop = apply_iir_filter(sig_stop, design, mode="zero_phase")
    out_pass = apply_iir_filter(sig_pass, design, mode="zero_phase")

    rms_in = np.sqrt(np.mean(sig_pass[steady]**2))
    assert np.sqrt(np.mean(out_pass[steady]**2)) / rms_in > 0.95
    assert np.sqrt(np.mean(out_stop[steady]**2)) / rms_in < 0.001


def test_butterworth_bandpass_attenuation():
    """Verify Butterworth bandpass passes in-band tone and attenuates both out-of-band tones."""
    sr = 10000
    band = (1500.0, 3000.0)
    design = design_iir_filter(sr, "bandpass", band, order=4)

    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    sig_low = np.sin(2 * np.pi * 500.0 * t)
    sig_mid = np.sin(2 * np.pi * 2200.0 * t)
    sig_high = np.sin(2 * np.pi * 4400.0 * t)

    steady = slice(200, -200)
    out_low = apply_iir_filter(sig_low, design, mode="zero_phase")
    out_mid = apply_iir_filter(sig_mid, design, mode="zero_phase")
    out_high = apply_iir_filter(sig_high, design, mode="zero_phase")

    rms_in = np.sqrt(np.mean(sig_mid[steady]**2))
    assert np.sqrt(np.mean(out_mid[steady]**2)) / rms_in > 0.95
    assert np.sqrt(np.mean(out_low[steady]**2)) / rms_in < 0.01
    assert np.sqrt(np.mean(out_high[steady]**2)) / rms_in < 0.01


def test_iir_notch_50hz_rejection():
    """Verify 50 Hz notch filter sharply attenuates a 50 Hz hum while preserving a 300 Hz tone."""
    sr = 8000
    design = design_iir_notch(sr, notch_freq=50.0, quality_factor=30.0)

    # Use 2.5s duration to allow high-Q (narrow 1.67 Hz bandwidth) IIR poles to settle
    duration = 2.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    hum_50hz = np.sin(2 * np.pi * 50.0 * t)
    tone_300hz = np.sin(2 * np.pi * 300.0 * t)

    # Evaluate in steady-state region
    steady = slice(int(sr * 1.0), -int(sr * 0.5))
    out_hum = apply_iir_filter(hum_50hz, design, mode="zero_phase")
    out_tone = apply_iir_filter(tone_300hz, design, mode="zero_phase")

    rms_in = np.sqrt(np.mean(hum_50hz[steady]**2))
    # 50 Hz must be suppressed by at least 30 dB (ratio < 0.03)
    assert np.sqrt(np.mean(out_hum[steady]**2)) / rms_in < 0.03
    # 300 Hz must be preserved (> 98% amplitude)
    assert np.sqrt(np.mean(out_tone[steady]**2)) / rms_in > 0.98


def test_get_iir_frequency_response():
    """Verify IIR frequency response outputs exact frequency axis from 0 to Nyquist."""
    sr = 8000
    design = design_iir_filter(sr, "lowpass", 1000.0, order=4)
    freqs, mag, mag_db, phase = get_iir_frequency_response(design, worN=1024)

    assert len(freqs) == 1024
    assert len(mag) == 1024
    assert len(mag_db) == 1024
    assert len(phase) == 1024
    assert freqs[0] == 0.0
    assert np.isclose(freqs[-1], sr / 2.0)
    # At DC (0 Hz), lowpass Butterworth magnitude is unity (0 dB)
    assert np.isclose(mag[0], 1.0, atol=1e-3)
    assert np.isclose(mag_db[0], 0.0, atol=1e-2)
    assert np.all(np.isfinite(mag_db))


# =====================================================================
# 3. FILTERING BEHAVIOR & MULTICHANNEL TESTS
# =====================================================================

def test_iir_filtering_preserves_length_and_multichannel():
    """Verify IIR filtering preserves length for mono and stereo in causal and zero-phase modes."""
    sr = 16000
    design = design_iir_filter(sr, "lowpass", 2000.0, order=4)

    # 1D Mono
    mono_in = np.random.default_rng(42).normal(0, 0.1, 2000)
    mono_causal = apply_iir_filter(mono_in, design, mode="causal")
    mono_zero = apply_iir_filter(mono_in, design, mode="zero_phase")
    assert mono_causal.shape == (2000,)
    assert mono_zero.shape == (2000,)

    # 2D Stereo
    stereo_in = np.random.default_rng(42).normal(0, 0.1, (2000, 2))
    stereo_causal = apply_iir_filter(stereo_in, design, mode="causal")
    stereo_zero = apply_iir_filter(stereo_in, design, mode="zero_phase")
    assert stereo_causal.shape == (2000, 2)
    assert stereo_zero.shape == (2000, 2)


def test_iir_zero_phase_alignment():
    """Verify zero-phase IIR filtering eliminates phase delay for a passband tone."""
    sr = 8000
    design = design_iir_filter(sr, "lowpass", 1500.0, order=4)

    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    sig_in = np.sin(2 * np.pi * 200.0 * t)
    sig_out = apply_iir_filter(sig_in, design, mode="zero_phase")

    test_slice = slice(200, 800)
    corr = np.correlate(sig_out[test_slice], sig_in[test_slice], mode="full")
    max_lag = np.argmax(corr) - (len(sig_in[test_slice]) - 1)
    assert max_lag == 0


# =====================================================================
# 4. VALIDATION & ERROR HANDLING TESTS
# =====================================================================

def test_iir_validation_invalid_sample_rate():
    """Verify non-positive sample rates raise ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        design_iir_filter(sample_rate=0, filter_type="lowpass", cutoff_hz=1000.0)
    with pytest.raises(ValueError, match="strictly positive"):
        design_iir_notch(sample_rate=-8000, notch_freq=50.0)


def test_iir_validation_cutoff_bounds():
    """Verify cutoff frequencies violating Nyquist or non-positive are rejected."""
    sr = 8000  # Nyquist = 4000 Hz

    with pytest.raises(ValueError, match="Nyquist"):
        design_iir_filter(sr, "lowpass", 4000.0)
    with pytest.raises(ValueError, match="Nyquist"):
        design_iir_filter(sr, "lowpass", 4500.0)
    with pytest.raises(ValueError, match="strictly positive"):
        design_iir_filter(sr, "lowpass", 0.0)


def test_iir_validation_band_edges():
    """Verify bandpass/bandstop parameter ordering and count validation."""
    sr = 8000

    # Inverted band edges
    with pytest.raises(ValueError, match="strictly less"):
        design_iir_filter(sr, "bandpass", [2500.0, 1000.0])
    # Single cutoff for bandpass
    with pytest.raises(ValueError, match="requires exactly two"):
        design_iir_filter(sr, "bandpass", [1000.0])


def test_iir_validation_order_and_q():
    """Verify invalid filter orders and non-positive Q are rejected."""
    with pytest.raises(ValueError, match="order must be an integer >= 1"):
        design_iir_filter(8000, "lowpass", 1000.0, order=0)

    with pytest.raises(ValueError, match="quality_factor must be strictly positive"):
        design_iir_notch(8000, notch_freq=50.0, quality_factor=0.0)


def test_iir_validation_short_signal_zero_phase():
    """Verify sosfiltfilt raises ValueError when input signal is shorter than padding requirement."""
    sr = 8000
    design = design_iir_filter(sr, "lowpass", 1000.0, order=6)  # 3 sections, padlen = 3 * 7 = 21

    short_signal = np.ones(15)
    with pytest.raises(ValueError, match="too short for zero-phase IIR filtering"):
        apply_iir_filter(short_signal, design, mode="zero_phase")

    # Causal mode should succeed on short signal
    causal_out = apply_iir_filter(short_signal, design, mode="causal")
    assert len(causal_out) == 15


def test_iir_pole_stability_all_poles_inside_unit_circle():
    """Verify check_sos_stability confirms stability of designed filters and detects unstable poles."""
    sr = 16000

    # Test all filter types
    designs = [
        design_iir_filter(sr, "lowpass", 2000.0, order=6),
        design_iir_filter(sr, "highpass", 3000.0, order=4),
        design_iir_filter(sr, "bandpass", (1000.0, 4000.0), order=3),
        design_iir_filter(sr, "bandstop", (1500.0, 2500.0), order=4),
        design_iir_notch(sr, notch_freq=50.0, quality_factor=30.0),
    ]

    for d in designs:
        stable, max_pole = check_sos_stability(d.sos)
        assert stable is True
        assert max_pole < 1.0

    # Artificially create an unstable second-order section: pole at z = 1.3
    # A(z) = (z - 1.3)(z - 0.5) = z^2 - 1.8z + 0.65 -> [b0, b1, b2, a0, a1, a2] = [1, 0, 0, 1, -1.8, 0.65]
    unstable_sos = np.array([[1.0, 0.0, 0.0, 1.0, -1.8, 0.65]], dtype=np.float64)
    stable_flag, max_pole_mag = check_sos_stability(unstable_sos)
    assert stable_flag is False
    assert np.isclose(max_pole_mag, 1.3)


def test_iir_order_semantics_prototype_vs_resulting():
    """Verify order semantics: lowpass order=N, bandpass/bandstop resulting order=2N."""
    sr = 8000

    # Lowpass: prototype order 4 -> resulting order 4
    lp = design_iir_filter(sr, "lowpass", 1000.0, order=4)
    assert lp.prototype_order == 4
    assert lp.resulting_order == 4
    assert lp.order == 4

    # Bandpass: prototype order 3 -> resulting order 6
    bp = design_iir_filter(sr, "bandpass", (500.0, 1500.0), order=3)
    assert bp.prototype_order == 3
    assert bp.resulting_order == 6
    assert bp.order == 6

    # Bandstop: prototype order 2 -> resulting order 4
    bs = design_iir_filter(sr, "bandstop", (800.0, 1200.0), order=2)
    assert bs.prototype_order == 2
    assert bs.resulting_order == 4
    assert bs.order == 4

    # Notch: prototype order 2 -> resulting order 2
    notch = design_iir_notch(sr, 50.0)
    assert notch.prototype_order == 2
    assert notch.resulting_order == 2
    assert notch.order == 2


def test_iir_zero_phase_padding_odd_order_scipy_match():
    """Verify odd-order Butterworth filter padding matches SciPy default and accepts length 19."""
    sr = 8000
    # Order 5 lowpass: 3 sections, 1 first-order section with b2=0, a2=0
    # ntaps = 2*3 + 1 - 1 = 6 -> padlen = 3 * 6 = 18
    # The old naive formula required > 21 samples and rejected length 19.
    design_odd = design_iir_filter(sr, "lowpass", cutoff_hz=1000.0, order=5)

    # Signal of length 19: greater than padlen=18 -> MUST SUCCEED
    sig_19 = np.ones(19, dtype=np.float64)
    out_19 = apply_iir_filter(sig_19, design_odd, mode="zero_phase")
    assert len(out_19) == 19
    assert np.all(np.isfinite(out_19))

    # Signal of length 18: equal to padlen=18 -> MUST RAISE ValueError
    sig_18 = np.ones(18, dtype=np.float64)
    with pytest.raises(ValueError, match="too short for zero-phase IIR filtering"):
        apply_iir_filter(sig_18, design_odd, mode="zero_phase")


def test_iir_zero_phase_padding_even_order_scipy_match():
    """Verify even-order Butterworth filter padding matches SciPy default and enforces threshold."""
    sr = 8000
    # Order 4 lowpass: 2 second-order sections, no b2=0
    # ntaps = 2*2 + 1 - 0 = 5 -> padlen = 3 * 5 = 15
    design_even = design_iir_filter(sr, "lowpass", cutoff_hz=1000.0, order=4)

    # Signal of length 16: greater than padlen=15 -> MUST SUCCEED
    sig_16 = np.ones(16, dtype=np.float64)
    out_16 = apply_iir_filter(sig_16, design_even, mode="zero_phase")
    assert len(out_16) == 16
    assert np.all(np.isfinite(out_16))

    # Signal of length 15: equal to padlen=15 -> MUST RAISE ValueError
    sig_15 = np.ones(15, dtype=np.float64)
    with pytest.raises(ValueError, match="too short for zero-phase IIR filtering"):
        apply_iir_filter(sig_15, design_even, mode="zero_phase")


