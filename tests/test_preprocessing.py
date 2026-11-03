"""
Automated Unit Tests for Preprocessing Module (src/preprocessing.py)
Tests downmixing, peak normalization, audio validation, and length alignment.
"""

import numpy as np
import pytest

from src.preprocessing import (
    align_signal_lengths,
    calculate_duration,
    convert_to_mono,
    normalize_audio,
    validate_audio,
)


def test_convert_to_mono_from_stereo():
    """Verify stereo downmixing correctly averages Left and Right channels."""
    left = np.array([0.2, 0.4, 0.6])
    right = np.array([0.4, 0.8, 0.2])
    stereo = np.column_stack([left, right])

    mono = convert_to_mono(stereo)
    expected = 0.5 * (left + right)

    assert mono.ndim == 1
    assert len(mono) == 3
    assert np.allclose(mono, expected)


def test_convert_to_mono_already_mono():
    """Verify convert_to_mono returns an identical independent copy when given mono."""
    mono_in = np.array([0.1, -0.2, 0.5])
    mono_out = convert_to_mono(mono_in)

    assert mono_out.ndim == 1
    assert np.allclose(mono_out, mono_in)
    # Ensure original array is not modified
    mono_out[0] = 999.0
    assert mono_in[0] == 0.1


def test_convert_to_mono_invalid_channels():
    """Verify convert_to_mono rejects 3+ channels."""
    tri_channel = np.ones((100, 3))
    with pytest.raises(ValueError, match="Unsupported channel count"):
        convert_to_mono(tri_channel)


def test_normalize_audio_peak():
    """Verify normalize_audio correctly rescales maximum absolute value to target peak."""
    data = np.array([-0.25, 0.5, -0.1, 0.2])
    normalized = normalize_audio(data, target_peak=1.0)

    assert np.isclose(np.max(np.abs(normalized)), 1.0)
    assert np.isclose(normalized[1], 1.0)
    assert np.isclose(normalized[0], -0.5)


def test_normalize_audio_silent():
    """Verify normalizing a zero-valued array returns all zeros without division by zero."""
    silent = np.zeros(100)
    result = normalize_audio(silent, target_peak=1.0)

    assert np.all(result == 0.0)
    assert not np.any(np.isnan(result))


def test_normalize_audio_invalid_target():
    """Verify invalid target_peak values are rejected."""
    data = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        normalize_audio(data, target_peak=0.0)

    with pytest.raises(ValueError):
        normalize_audio(data, target_peak=1.5)


def test_validate_audio_valid_cases():
    """Verify validate_audio returns True for standard mono and stereo arrays."""
    assert validate_audio(np.ones(1000), 44100) is True
    assert validate_audio(np.ones((1000, 2)), 48000) is True


def test_validate_audio_empty():
    """Verify validate_audio rejects empty arrays."""
    with pytest.raises(ValueError, match="empty"):
        validate_audio(np.array([]), 44100)


def test_validate_audio_nan_and_inf():
    """Verify validate_audio rejects NaN and infinite values."""
    with pytest.raises(ValueError, match="NaN or Infinity"):
        validate_audio(np.array([1.0, np.nan, 3.0]), 44100)

    with pytest.raises(ValueError, match="NaN or Infinity"):
        validate_audio(np.array([1.0, np.inf, 3.0]), 44100)


def test_validate_audio_invalid_sample_rate():
    """Verify non-positive or non-integer sample rates raise exceptions."""
    data = np.ones(100)
    with pytest.raises(ValueError):
        validate_audio(data, 0)

    with pytest.raises(ValueError):
        validate_audio(data, -44100)

    with pytest.raises(TypeError):
        validate_audio(data, "44100")  # type: ignore


def test_calculate_duration():
    """Verify duration formula T = N / fs."""
    assert np.isclose(calculate_duration(44100, 44100), 1.0)
    assert np.isclose(calculate_duration(22050, 44100), 0.5)

    with pytest.raises(ValueError):
        calculate_duration(-10, 44100)
    with pytest.raises(ValueError):
        calculate_duration(100, 0)


def test_align_signal_lengths_truncate():
    """Verify truncate mode cuts longer signal to match shorter signal."""
    sig1 = np.arange(10)
    sig2 = np.arange(15)
    a1, a2 = align_signal_lengths(sig1, sig2, mode="truncate")

    assert len(a1) == 10
    assert len(a2) == 10
    assert np.array_equal(a1, np.arange(10))
    assert np.array_equal(a2, np.arange(10))


def test_align_signal_lengths_pad():
    """Verify pad mode zero-pads shorter signal to match longer signal."""
    sig1 = np.ones(5)
    sig2 = np.ones(10)
    a1, a2 = align_signal_lengths(sig1, sig2, mode="pad")

    assert len(a1) == 10
    assert len(a2) == 10
    assert np.array_equal(a1[:5], np.ones(5))
    assert np.array_equal(a1[5:], np.zeros(5))
