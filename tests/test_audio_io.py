"""
Automated Unit Tests for Audio I/O Module (src/audio_io.py)
Tests loading, saving, metadata calculation, and error handling for WAV audio.
"""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.audio_io import get_audio_info, load_audio, save_audio


@pytest.fixture
def temp_audio_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for audio test files."""
    audio_dir = tmp_path / "audio_tests"
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir


@pytest.fixture
def synthetic_mono_wav(temp_audio_dir: Path) -> Path:
    """Generate a synthetic 1-second 440 Hz sinusoidal mono WAV file."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    file_path = temp_audio_dir / "test_mono.wav"
    sf.write(str(file_path), sine, samplerate=sr, subtype="PCM_16")
    return file_path


@pytest.fixture
def synthetic_stereo_wav(temp_audio_dir: Path) -> Path:
    """Generate a synthetic 0.5-second two-channel stereo WAV file."""
    sr = 48000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    ch1 = 0.4 * np.sin(2 * np.pi * 300.0 * t)
    ch2 = 0.6 * np.cos(2 * np.pi * 600.0 * t)
    stereo = np.column_stack([ch1, ch2])

    file_path = temp_audio_dir / "test_stereo.wav"
    sf.write(str(file_path), stereo, samplerate=sr, subtype="PCM_16")
    return file_path


def test_load_mono_audio(synthetic_mono_wav: Path):
    """Verify loading mono audio returns correct shape, sample rate, and float64 dtype."""
    data, sr = load_audio(synthetic_mono_wav)
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.float64
    assert sr == 44100
    assert data.ndim == 1
    assert len(data) == 44100
    # Peak amplitude should be ~0.5
    assert np.isclose(np.max(np.abs(data)), 0.5, atol=1e-3)


def test_load_stereo_audio(synthetic_stereo_wav: Path):
    """Verify loading stereo audio returns a 2D array with 2 channels."""
    data, sr = load_audio(synthetic_stereo_wav)
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.float64
    assert sr == 48000
    assert data.ndim == 2
    assert data.shape == (24000, 2)


def test_load_nonexistent_file(temp_audio_dir: Path):
    """Verify FileNotFoundError is raised when target audio file does not exist."""
    missing_file = temp_audio_dir / "nonexistent.wav"
    with pytest.raises(FileNotFoundError):
        load_audio(missing_file)


def test_load_corrupted_file(temp_audio_dir: Path):
    """Verify RuntimeError is raised when file contains corrupt non-audio data."""
    corrupt_file = temp_audio_dir / "corrupted.wav"
    corrupt_file.write_text("This is not a valid RIFF WAV audio header.")
    with pytest.raises(RuntimeError):
        load_audio(corrupt_file)


def test_save_and_reload_mono(temp_audio_dir: Path):
    """Verify saving a mono array to disk and reloading produces matching samples."""
    sr = 22050
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    original = 0.7 * np.sin(2 * np.pi * 1000.0 * t)

    out_path = temp_audio_dir / "saved_mono.wav"
    save_audio(out_path, original, sample_rate=sr, subtype="PCM_16")

    assert out_path.exists()
    reloaded, reloaded_sr = load_audio(out_path)

    assert reloaded_sr == sr
    assert len(reloaded) == len(original)
    # Quantization error for 16-bit PCM should be < 1e-4
    assert np.allclose(reloaded, original, atol=1e-4)


def test_save_preserves_existing_file_without_overwrite(temp_audio_dir: Path):
    """Verify save_audio raises FileExistsError if overwrite=False to protect original recordings."""
    file_path = temp_audio_dir / "protected.wav"
    file_path.write_text("existing content")

    dummy_data = np.zeros(100)
    with pytest.raises(FileExistsError):
        save_audio(file_path, dummy_data, sample_rate=44100, overwrite=False)


def test_save_clipping_handling(temp_audio_dir: Path):
    """Verify save_audio raises ValueError if audio exceeds [-1, 1] and allow_clipping=False."""
    overscaled_data = np.array([0.5, 1.25, -0.8])
    out_path = temp_audio_dir / "clipped.wav"

    with pytest.raises(ValueError, match="exceeds unity"):
        save_audio(out_path, overscaled_data, sample_rate=16000, allow_clipping=False)

    # When allow_clipping=True, it should saturate and succeed
    save_audio(out_path, overscaled_data, sample_rate=16000, allow_clipping=True, overwrite=True)
    reloaded, _ = load_audio(out_path)
    assert np.max(reloaded) <= 1.0


def test_get_audio_info(synthetic_stereo_wav: Path):
    """Verify metadata extraction computes accurate duration, peak, RMS, and channels."""
    data, sr = load_audio(synthetic_stereo_wav)
    info = get_audio_info(data, sr)

    assert info["sample_rate"] == 48000
    assert info["num_channels"] == 2
    assert info["num_samples"] == 24000
    assert np.isclose(info["duration_seconds"], 0.5)
    assert info["peak_amplitude"] > 0.0
    assert info["rms_amplitude"] > 0.0
    assert info["crest_factor_db"] > 0.0


def test_get_audio_info_invalid():
    """Verify get_audio_info raises ValueError on empty or invalid data."""
    with pytest.raises(ValueError):
        get_audio_info(np.array([]), 44100)

    with pytest.raises(ValueError):
        get_audio_info(np.array([0.1, np.nan]), 44100)

    with pytest.raises(ValueError):
        get_audio_info(np.array([0.1, 0.2]), 0)
