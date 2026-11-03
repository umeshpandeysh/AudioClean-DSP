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
    reset_reference_uploader,
    reset_primary_uploader,
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


def make_test_wav_bytes(freq: float = 440.0, sr: int = 44100, duration: float = 0.1) -> bytes:
    """Helper to generate valid in-memory PCM_16 WAV bytes for test uploads."""
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    data = 0.5 * np.sin(2.0 * np.pi * freq * t)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_apptest_reference_detached_on_same_shape_primary_replacement():
    """Requirement 2: Replace primary A with primary B (same shape/sr) and verify reference A detaches and cannot silently reattach."""
    from pathlib import Path
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15)
    at.run()

    # 1. Switch to Workflow B
    for r in at.radio:
        if "Ingestion Workflow" in r.label:
            r.set_value(r.options[1])
            break
    at.run()
    assert not at.exception

    # 2. Upload primary A (440 Hz) and reference A (440 Hz)
    wav_prim_a = make_test_wav_bytes(freq=440.0)
    wav_prim_b = make_test_wav_bytes(freq=880.0)  # Identical shape & sr, different samples
    wav_ref_a = make_test_wav_bytes(freq=440.0)

    pu = [f for f in at.file_uploader if "Audio Recording" in f.label][0]
    pu.upload("primary_a.wav", wav_prim_a)
    at.run()
    assert not at.exception

    ru = [f for f in at.file_uploader if "Reference" in f.label][0]
    ru.upload("ref_a.wav", wav_ref_a)
    at.run()
    assert not at.exception
    assert at.session_state.reference_audio is not None
    assert at.session_state.ref_file_hash is not None

    # 3. Replace primary A with primary B (same shape, same sr)
    pu = [f for f in at.file_uploader if "Audio Recording" in f.label][0]
    pu.upload("primary_b.wav", wav_prim_b)
    at.run()
    assert not at.exception

    # 4. Verify reference A is detached and cleared
    assert at.session_state.reference_audio is None
    assert at.session_state.ref_file_hash is None
    assert at.session_state.ref_primary_hash is None

    # 5. Verify reference A cannot silently reattach on a subsequent rerun
    at.run()
    assert not at.exception
    assert at.session_state.reference_audio is None
    assert at.session_state.ref_file_hash is None


def test_apptest_reference_removal_persists_across_reruns():
    """Requirement 2: Remove the reference uploader and verify it stays removed after rerun."""
    from pathlib import Path
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15)
    at.run()

    # Switch to Workflow B
    for r in at.radio:
        if "Ingestion Workflow" in r.label:
            r.set_value(r.options[1])
            break
    at.run()

    # Upload primary and reference
    wav_prim = make_test_wav_bytes(freq=440.0)
    wav_ref = make_test_wav_bytes(freq=440.0)
    pu = [f for f in at.file_uploader if "Audio Recording" in f.label][0]
    pu.upload("primary.wav", wav_prim)
    at.run()

    ru = [f for f in at.file_uploader if "Reference" in f.label][0]
    ru.upload("ref.wav", wav_ref)
    at.run()
    assert at.session_state.reference_audio is not None

    # User removes reference uploader
    ru = [f for f in at.file_uploader if "Reference" in f.label][0]
    ru.set_value(None)
    at.run()
    assert not at.exception
    assert at.session_state.reference_audio is None
    assert at.session_state.ref_file_hash is None

    # Rerun again to confirm it stays removed and does not resurrect
    at.run()
    assert not at.exception
    assert at.session_state.reference_audio is None


def test_apptest_workflow_b_to_a_degraded_audio_no_reference():
    """Requirement 2: Switch Workflow B -> Workflow A after uploading degraded audio and verify no ground-truth reference is created."""
    from pathlib import Path
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15)
    at.run()

    # Switch to Workflow B
    for r in at.radio:
        if "Ingestion Workflow" in r.label:
            r.set_value(r.options[1])
            break
    at.run()

    # Upload real-world degraded audio
    wav_deg = make_test_wav_bytes(freq=300.0)
    pu = [f for f in at.file_uploader if "Audio Recording" in f.label][0]
    pu.upload("degraded.wav", wav_deg)
    at.run()
    assert at.session_state.source_type == "noisy_upload"
    assert at.session_state.reference_audio is None

    # Switch to Workflow A
    for r in at.radio:
        if "Ingestion Workflow" in r.label:
            r.set_value(r.options[0])
            break
    at.run()
    assert not at.exception
    assert at.session_state.active_ingest_mode == "Workflow A"
    # Ground-truth reference MUST remain None
    assert at.session_state.reference_audio is None


def test_apptest_workflow_switch_after_primary_change_no_leakage():
    """Requirement 2: Switch workflows after changing the primary recording and verify old reference state cannot return."""
    from pathlib import Path
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15)
    at.run()

    # Switch to Workflow B
    for r in at.radio:
        if "Ingestion Workflow" in r.label:
            r.set_value(r.options[1])
            break
    at.run()

    pu = [f for f in at.file_uploader if "Audio Recording" in f.label][0]
    pu.upload("prim1.wav", make_test_wav_bytes(freq=440.0))
    at.run()

    ru = [f for f in at.file_uploader if "Reference" in f.label][0]
    ru.upload("ref1.wav", make_test_wav_bytes(freq=440.0))
    at.run()
    assert at.session_state.reference_audio is not None

    # Change primary recording
    pu = [f for f in at.file_uploader if "Audio Recording" in f.label][0]
    pu.upload("prim2.wav", make_test_wav_bytes(freq=660.0))
    at.run()
    assert at.session_state.reference_audio is None

    # Switch to Workflow A
    for r in at.radio:
        if "Ingestion Workflow" in r.label:
            r.set_value(r.options[0])
            break
    at.run()
    assert at.session_state.reference_audio is None

    # Switch back to Workflow B
    for r in at.radio:
        if "Ingestion Workflow" in r.label:
            r.set_value(r.options[1])
            break
    at.run()
    assert not at.exception
    assert at.session_state.reference_audio is None


def test_apptest_primary_removal_resets_reference_and_restores_demo():
    """Verify that removing primary upload invalidates reference uploader and restores demo signal."""
    from pathlib import Path
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15)
    at.run()

    # Upload primary
    pu = [f for f in at.file_uploader if "Audio Recording" in f.label][0]
    pu.upload("custom.wav", make_test_wav_bytes(freq=550.0))
    at.run()
    assert at.session_state.source_type == "clean_upload"
    assert at.session_state.primary_file_hash is not None

    # Clear primary upload
    pu = [f for f in at.file_uploader if "Audio Recording" in f.label][0]
    pu.set_value(None)
    at.run()
    assert not at.exception
    assert at.session_state.source_type == "synthetic"
    assert at.session_state.primary_file_hash is None


def test_uploader_lifecycle_reset_helpers():
    """Verify reset_reference_uploader and reset_primary_uploader helper functions."""
    import streamlit as st
    st.session_state.reference_audio = np.ones(10)
    st.session_state.ref_file_hash = "hash_123"
    st.session_state.ref_primary_hash = "prim_hash"
    st.session_state.ref_uploader_version = 0
    st.session_state["ref_uploader_v0"] = "mock_file"

    reset_reference_uploader()
    assert st.session_state.reference_audio is None
    assert st.session_state.ref_file_hash is None
    assert st.session_state.ref_primary_hash is None
    assert "ref_uploader_v0" not in st.session_state
    assert st.session_state.ref_uploader_version == 1

    st.session_state.primary_uploader_version = 0
    st.session_state.primary_file_hash = "prim_hash"
    st.session_state["primary_audio_uploader_v0"] = "mock_prim"

    reset_primary_uploader()
    assert st.session_state.primary_file_hash is None
    assert "primary_audio_uploader_v0" not in st.session_state
    assert st.session_state.primary_uploader_version == 1


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
