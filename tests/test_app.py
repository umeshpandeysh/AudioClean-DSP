"""Tests for Streamlit application helper logic and workflow contracts."""

import io
import numpy as np
import pytest
import soundfile as sf
from app import audio_to_bytes, validate_reference_match


def test_audio_to_bytes_valid_wav():
    sr = 44100
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_bytes = audio_to_bytes(data, sr)
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 0

    # Read back through soundfile to verify standard PCM_16 WAV format
    buf = io.BytesIO(wav_bytes)
    read_data, read_sr = sf.read(buf, dtype="float64")
    assert read_sr == sr
    assert len(read_data) == len(data)
    assert np.allclose(read_data, data, atol=1e-4)


def test_audio_to_bytes_clipping_protection():
    sr = 16000
    overshoot = np.array([-2.5, -0.5, 0.0, 0.5, 2.5], dtype=np.float64)
    wav_bytes = audio_to_bytes(overshoot, sr)
    buf = io.BytesIO(wav_bytes)
    read_data, _ = sf.read(buf, dtype="float64")
    assert np.max(read_data) <= 1.0
    assert np.min(read_data) >= -1.0


def test_validate_reference_match_compatible():
    ref = np.ones((1000,), dtype=np.float64)
    noisy = np.zeros((1000,), dtype=np.float64)
    ok, msg = validate_reference_match(ref, 44100, noisy, 44100)
    assert ok is True
    assert msg == ""


def test_validate_reference_match_sample_rate_mismatch():
    ref = np.ones((1000,), dtype=np.float64)
    noisy = np.zeros((1000,), dtype=np.float64)
    ok, msg = validate_reference_match(ref, 48000, noisy, 44100)
    assert ok is False
    assert "Sample rate mismatch" in msg


def test_validate_reference_match_shape_mismatch():
    ref = np.ones((1000,), dtype=np.float64)
    noisy = np.zeros((900,), dtype=np.float64)
    ok, msg = validate_reference_match(ref, 44100, noisy, 44100)
    assert ok is False
    assert "Array shape mismatch" in msg


def test_validate_reference_match_non_finite():
    ref = np.array([0.0, np.nan, 0.5], dtype=np.float64)
    noisy = np.zeros((3,), dtype=np.float64)
    ok, msg = validate_reference_match(ref, 44100, noisy, 44100)
    assert ok is False
    assert "non-finite" in msg
