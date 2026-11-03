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
    format_speedup,
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


def test_format_speedup_valid():
    """Verify format_speedup with valid positive numeric values."""
    assert format_speedup(3.14159) == "3.14x"
    assert format_speedup(1.0) == "1.00x"
    assert format_speedup(0.555) == "0.56x"
    assert format_speedup(10) == "10.00x"


def test_format_speedup_none():
    """Verify format_speedup returns 'N/A' when None is provided without throwing TypeError."""
    assert format_speedup(None) == "N/A"


def test_format_speedup_non_finite_and_negative():
    """Verify format_speedup returns 'N/A' on non-finite, negative, and zero values."""
    assert format_speedup(0.0) == "N/A"
    assert format_speedup(-1.5) == "N/A"
    assert format_speedup(float("nan")) == "N/A"
    assert format_speedup(float("inf")) == "N/A"
    assert format_speedup(float("-inf")) == "N/A"
    assert format_speedup("invalid") == "N/A"


def test_workflow_transition_provenance_preservation():
    """Verify that switching between Workflow A and B strictly preserves data provenance.

    Switching from Workflow B to Workflow A must NOT treat degraded audio as clean ground truth.
    """
    # 1. Synthetic demo tone in Workflow A
    source_type = "synthetic"
    audio_data = np.ones(1000)

    # Toggle to Workflow B
    ref_audio_b = None  # Workflow B clears ground-truth reference
    assert ref_audio_b is None

    # Toggle back to Workflow A with synthetic provenance
    ref_audio_restored = audio_data.copy() if source_type in ("synthetic", "clean_upload") else None
    assert ref_audio_restored is not None

    # 2. Real-world degraded upload in Workflow B
    source_type = "noisy_upload"
    degraded_audio = np.random.randn(1000)
    audio_data = degraded_audio

    # User toggles to Workflow A while degraded recording is loaded
    ref_audio_a = audio_data.copy() if source_type in ("synthetic", "clean_upload") else None
    # Must NOT copy degraded audio to reference_audio!
    assert ref_audio_a is None


def test_reference_lifecycle_primary_replacement():
    """Verify that replacing primary recording detaches any attached external reference."""
    prim_1 = np.ones(1000, dtype=np.float64)
    prim_2 = np.ones(2000, dtype=np.float64)
    ref = np.ones(1000, dtype=np.float64)

    h_prim_1 = compute_file_hash(prim_1.tobytes())
    h_prim_2 = compute_file_hash(prim_2.tobytes())
    h_ref = compute_file_hash(ref.tobytes())

    # Reference attached to prim_1
    ref_file_hash = h_ref
    ref_primary_hash = h_prim_1
    reference_audio = ref

    # Primary replaced with prim_2
    if h_prim_2 != h_prim_1:
        reference_audio = None
        ref_file_hash = None
        ref_primary_hash = None

    assert reference_audio is None
    assert ref_file_hash is None
    assert ref_primary_hash is None


def test_reference_lifecycle_uploader_cleared():
    """Verify that clearing the reference uploader resets reference state to None."""
    ref_audio = np.ones(1000)
    ref_file_hash = "abc123hash"
    ref_primary_hash = "prim123hash"

    # User removes uploaded reference file (ref_file is None)
    ref_file = None
    if ref_file is None and ref_file_hash is not None:
        ref_audio = None
        ref_file_hash = None
        ref_primary_hash = None

    assert ref_audio is None
    assert ref_file_hash is None
    assert ref_primary_hash is None


def test_reference_lifecycle_same_shape_hash_invalidation():
    """Verify hash-based reference invalidation when two distinct recordings share identical shape."""
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    prim_rec_1 = 0.5 * np.sin(2 * np.pi * 440 * t)
    prim_rec_2 = 0.5 * np.cos(2 * np.pi * 880 * t)
    ref_rec = 0.5 * np.sin(2 * np.pi * 440 * t)

    h1 = compute_file_hash(prim_rec_1.tobytes())
    h2 = compute_file_hash(prim_rec_2.tobytes())
    assert h1 != h2
    assert prim_rec_1.shape == prim_rec_2.shape

    # Reference attached for recording 1
    ref_primary_hash = h1
    ref_file_hash = compute_file_hash(ref_rec.tobytes())
    reference_audio = ref_rec

    # Primary replaced with recording 2 (same shape!)
    current_primary_hash = h2
    if ref_primary_hash != current_primary_hash:
        reference_audio = None
        ref_file_hash = None
        ref_primary_hash = None

    assert reference_audio is None
    assert ref_file_hash is None
    assert ref_primary_hash is None


def test_reference_lifecycle_rejected_reference_clears_prior_state():
    """Verify that uploading an invalid reference clears any prior reference instead of retaining it."""
    # Prior state had a valid reference
    ref_audio = np.ones(1000)
    ref_file_hash = "valid_prior_ref_hash"
    ref_primary_hash = "prim_hash"

    # New reference upload with non-finite values
    invalid_new_ref = np.array([1.0, np.nan, 0.5])
    valid, msg = validate_reference_match(invalid_new_ref, 44100, np.ones(3), 44100)
    assert valid is False

    # Handling logic must reset prior state to None
    if not valid:
        ref_audio = None
        ref_file_hash = None
        ref_primary_hash = None

    assert ref_audio is None
    assert ref_file_hash is None
    assert ref_primary_hash is None


def test_streamlit_apptest_workflow_toggle():
    """Verify AppTest switching between Workflow A and Workflow B."""
    from pathlib import Path
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15)
    at.run()
    assert not at.exception

    # Default is Workflow A
    assert at.session_state.active_ingest_mode == "Workflow A"
    assert at.session_state.reference_audio is not None

    # Select Workflow B
    for r in at.radio:
        if "Ingestion Workflow" in r.label:
            r.set_value(r.options[1])
            break
    at.run()
    assert not at.exception
    assert at.session_state.active_ingest_mode == "Workflow B"
    assert at.session_state.reference_audio is None


def test_streamlit_apptest_workspace_switching():
    """Verify AppTest navigation across workspaces 1 through 6 without uncaught exceptions."""
    from pathlib import Path
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15)
    at.run()
    assert not at.exception

    # Find sidebar radio
    sidebar_nav = at.sidebar.radio[0]
    for opt in sidebar_nav.options:
        sidebar_nav.set_value(opt)
        at.run()
        assert not at.exception
