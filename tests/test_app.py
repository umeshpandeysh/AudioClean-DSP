"""Automated tests for Streamlit application contracts, state persistence and helper logic."""

import io
import numpy as np
import pytest
import soundfile as sf
from streamlit.testing.v1 import AppTest
from app import (
    audio_to_bytes,
    compute_file_hash,
    validate_reference_match,
    apply_linked_normalization,
)


def test_audio_to_bytes_valid_wav():
    sr = 44100
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_bytes = audio_to_bytes(data, sr)
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 0

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


def test_compute_file_hash_deterministic():
    b1 = b"audio_test_content_123"
    b2 = b"audio_test_content_123"
    b3 = b"audio_test_content_456"
    assert compute_file_hash(b1) == compute_file_hash(b2)
    assert compute_file_hash(b1) != compute_file_hash(b3)


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


def test_apply_linked_normalization_preserves_relative_gain():
    ref = np.array([0.2, -0.4, 0.5], dtype=np.float64)
    noise = np.array([0.1, 0.1, -0.1], dtype=np.float64)
    noisy = ref + noise  # peak is 0.5 + (-0.1) = 0.4, max abs is 0.4
    orig_peak = np.max(np.abs(noisy))

    norm_noisy, norm_ref, _, gain = apply_linked_normalization(noisy, ref, None, target_peak=1.0)
    assert np.isclose(np.max(np.abs(norm_noisy)), 1.0)
    assert np.isclose(gain, 1.0 / orig_peak)
    assert np.allclose(norm_ref, ref * gain)

    # SNR before and after linked normalization MUST be identical
    snr_before = np.sum(ref**2) / np.sum((noisy - ref)**2)
    snr_after = np.sum(norm_ref**2) / np.sum((norm_noisy - norm_ref)**2)
    assert np.isclose(snr_before, snr_after)


def test_apply_linked_normalization_silent_signal():
    noisy = np.zeros((10,), dtype=np.float64)
    ref = np.zeros((10,), dtype=np.float64)
    norm_noisy, norm_ref, _, gain = apply_linked_normalization(noisy, ref, None, target_peak=1.0)
    assert gain == 1.0
    assert np.array_equal(norm_noisy, noisy)


def test_streamlit_app_interactive_session_state_and_navigation():
    """Verify AppTest launch, session state initialization, and persistence across reruns."""
    from pathlib import Path
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15)
    at.run()
    assert not at.exception

    # Verify session state defaults
    assert at.session_state.sample_rate == 44100
    assert at.session_state.active_ingest_mode == "Workflow A"
    assert at.session_state.audio_data is not None
    assert at.session_state.reference_audio is not None
    assert at.session_state.noisy_audio is not None

    # Simulate synthesizing noise in session state
    at.session_state.noisy_audio = at.session_state.noisy_audio + 0.05
    orig_mean = float(np.mean(at.session_state.noisy_audio))

    # Rerun app (simulating user interaction or workspace switch)
    at.run()
    assert not at.exception

    # Verify processed noisy_audio is preserved and not overwritten on rerun
    assert np.isclose(float(np.mean(at.session_state.noisy_audio)), orig_mean)
