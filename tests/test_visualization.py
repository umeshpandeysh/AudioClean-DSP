"""
Automated Unit Tests for Visualization Module (src/visualization.py)
Tests waveform, FFT spectrum, spectrogram, filter response, phase, and comparison plotting.
Verifies return of Matplotlib Figure objects, safe file export, and non-interactive execution.
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.fft_analysis import compute_fft, compute_spectrogram
from src.fir_filters import design_fir_filter, get_fir_frequency_response
from src.iir_filters import design_iir_filter, get_iir_frequency_response
from src.visualization import (
    plot_fft_spectrum,
    plot_filter_response,
    plot_phase_response,
    plot_signal_comparison,
    plot_spectrogram,
    plot_waveform,
)


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Ensure all created Matplotlib figures are closed after each test to prevent memory leaks."""
    yield
    plt.close("all")


def test_plot_waveform_mono_and_stereo():
    """Verify plot_waveform returns valid Matplotlib Figure objects for mono and stereo inputs."""
    sr = 8000
    mono = np.sin(2 * np.pi * 400 * np.linspace(0, 0.1, 800))
    stereo = np.column_stack([mono, mono * 0.5])

    fig_mono = plot_waveform(mono, sample_rate=sr, title="Mono Test")
    assert isinstance(fig_mono, matplotlib.figure.Figure)
    assert len(fig_mono.axes) == 1

    fig_stereo = plot_waveform(stereo, sample_rate=sr, title="Stereo Test")
    assert isinstance(fig_stereo, matplotlib.figure.Figure)


def test_plot_fft_spectrum():
    """Verify plot_fft_spectrum generates valid spectral plots in linear and dB modes."""
    sr = 8000
    sig = np.sin(2 * np.pi * 500 * np.linspace(0, 0.25, 2000))
    freqs, mag, mag_db = compute_fft(sig, sample_rate=sr)

    fig_db = plot_fft_spectrum(freqs, mag_db, db_scale=True, max_freq=2000.0)
    assert isinstance(fig_db, matplotlib.figure.Figure)

    fig_lin = plot_fft_spectrum(freqs, mag, db_scale=False, max_freq=2000.0)
    assert isinstance(fig_lin, matplotlib.figure.Figure)


def test_plot_spectrogram():
    """Verify plot_spectrogram returns figure with plot axis and colorbar."""
    sr = 8000
    sig = np.sin(2 * np.pi * 300 * np.linspace(0, 0.5, 4000))
    times, freqs, Sxx_db = compute_spectrogram(sig, sample_rate=sr, nperseg=512)

    fig = plot_spectrogram(times, freqs, Sxx_db, max_freq=2000.0)
    assert isinstance(fig, matplotlib.figure.Figure)
    # Spectrogram figure contains 2 axes: the spectrogram mesh and the colorbar
    assert len(fig.axes) == 2


def test_plot_filter_response_fir_and_iir():
    """Verify plot_filter_response handles both FIR and IIR frequency response data."""
    sr = 8000
    # FIR
    fir = design_fir_filter(sr, "lowpass", 1000.0, numtaps=51)
    f_fir, _, m_fir, _ = get_fir_frequency_response(fir)
    fig_fir = plot_filter_response(f_fir, m_fir, title="FIR LP", cutoff_hz=1000.0)
    assert isinstance(fig_fir, matplotlib.figure.Figure)

    # IIR
    iir = design_iir_filter(sr, "highpass", 1500.0, order=4)
    f_iir, _, m_iir, _ = get_iir_frequency_response(iir)
    fig_iir = plot_filter_response(f_iir, m_iir, title="IIR HP", cutoff_hz=1500.0)
    assert isinstance(fig_iir, matplotlib.figure.Figure)


def test_plot_phase_response():
    """Verify plot_phase_response produces valid phase curves."""
    sr = 8000
    fir = design_fir_filter(sr, "lowpass", 1000.0, numtaps=31)
    f, _, _, phase = get_fir_frequency_response(fir)

    fig = plot_phase_response(f, phase)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_signal_comparison():
    """Verify plot_signal_comparison creates a 3-panel comparative display."""
    sr = 8000
    t = np.linspace(0, 0.1, 800)
    clean = np.sin(2 * np.pi * 400 * t)
    noisy = clean + 0.2 * np.random.default_rng(0).normal(0, 1, 800)
    filtered = clean * 0.95

    fig = plot_signal_comparison(clean, noisy, filtered, sample_rate=sr)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 3


def test_plot_saving_and_overwrite_protection(tmp_path: Path):
    """Verify figure saving creates files on disk and protects against unpermitted overwrites."""
    out_dir = tmp_path / "plot_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "test_waveform.png"

    sig = np.zeros(500)
    fig = plot_waveform(sig, sample_rate=8000, save_path=out_file, overwrite=False)

    assert out_file.exists()
    assert out_file.stat().st_size > 0

    # Second write with overwrite=False must raise FileExistsError
    with pytest.raises(FileExistsError):
        plot_waveform(sig, sample_rate=8000, save_path=out_file, overwrite=False)

    # Second write with overwrite=True should succeed
    plot_waveform(sig, sample_rate=8000, save_path=out_file, overwrite=True)


def test_plot_signal_comparison_mismatch_validation():
    """Verify plot_signal_comparison validates lengths unless allow_length_mismatch=True."""
    orig = np.ones(1000)
    noisy = np.ones(950)
    filtered = np.ones(1000)

    # Must raise ValueError when lengths differ and allow_length_mismatch is False
    with pytest.raises(ValueError, match="Signal length mismatch"):
        plot_signal_comparison(orig, noisy, filtered, sample_rate=8000, allow_length_mismatch=False)

    # Must succeed when allow_length_mismatch is True
    fig = plot_signal_comparison(orig, noisy, filtered, sample_rate=8000, allow_length_mismatch=True)
    assert fig is not None

