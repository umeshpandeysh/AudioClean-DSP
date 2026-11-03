"""
AudioClean DSP - End-to-End Integration Test Suite
Validates the complete signal processing pipeline from Phases 1 through 5:
Audio I/O -> Preprocessing -> Noise Synthesis -> FIR & IIR Filtering -> FFT & Spectrogram -> Visualization.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.audio_io import get_audio_info, load_audio, save_audio
from src.fft_analysis import compute_fft, compute_spectrogram
from src.fir_filters import apply_fir_filter, design_fir_filter, get_fir_frequency_response
from src.iir_filters import apply_iir_filter, design_iir_filter, design_iir_notch, get_iir_frequency_response
from src.noise_generator import add_high_frequency_noise, add_sinusoidal_hum, add_white_gaussian_noise
from src.preprocessing import convert_to_mono, normalize_audio, validate_audio
from src.visualization import (
    plot_fft_spectrum,
    plot_filter_response,
    plot_signal_comparison,
    plot_spectrogram,
    plot_waveform,
)


def test_complete_dsp_pipeline_e2e(tmp_path: Path):
    """
    Execute full DSP pipeline end-to-end:
    1. Create synthetic clean audio signal (fundamental speech-like tone + harmonic).
    2. Save and reload WAV using audio_io.
    3. Preprocess and normalize.
    4. Inject controlled noise: 50 Hz electrical hum + WGN (target SNR = 10 dB).
    5. Clean via IIR 50 Hz notch filter (zero-phase) followed by FIR lowpass filter (zero-phase).
    6. Analyze spectrum before and after filtering using FFT and STFT spectrogram.
    7. Generate diagnostic figures and verify non-empty, finite data structures.
    """
    sr = 16000
    duration = 1.0
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Clean signal: 400 Hz tone + 800 Hz harmonic
    clean_signal = 0.5 * np.sin(2 * np.pi * 400.0 * t) + 0.25 * np.sin(2 * np.pi * 800.0 * t)

    # 1. WAV Save & Reload
    wav_path = tmp_path / "test_input.wav"
    save_audio(wav_path, clean_signal, sr, overwrite=True)
    loaded_audio, loaded_sr = load_audio(wav_path)
    assert loaded_sr == sr
    assert np.allclose(clean_signal, loaded_audio, atol=1e-3)

    info = get_audio_info(loaded_audio, loaded_sr)
    assert info["sample_rate"] == sr
    assert info["num_channels"] == 1
    assert abs(info["duration_seconds"] - 1.0) < 0.01

    # 2. Preprocess & Normalize
    norm_audio = normalize_audio(loaded_audio, target_peak=0.9)
    validate_audio(norm_audio, sr)
    assert np.isclose(np.max(np.abs(norm_audio)), 0.9, atol=1e-3)

    # 3. Controlled Noise Generation
    # Add 50 Hz powerline hum + harmonics [2, 3]
    hum_audio, _ = add_sinusoidal_hum(norm_audio, sr, freq=50.0, amplitude=0.2, harmonics=[2, 3])
    # Add White Gaussian Noise at 10 dB SNR
    noisy_audio, _ = add_white_gaussian_noise(hum_audio, target_snr_db=10.0, seed=42)
    assert len(noisy_audio) == num_samples
    assert np.all(np.isfinite(noisy_audio))

    # 4. Filter Cascade:
    # Step A: IIR 50 Hz Notch filter (zero-phase)
    notch_design = design_iir_notch(sr, notch_freq=50.0, quality_factor=30.0)
    assert notch_design.order == 2
    dehummed_audio = apply_iir_filter(noisy_audio, notch_design, mode="zero_phase")

    # Step B: FIR Lowpass filter (cutoff 1200 Hz, zero-phase)
    fir_design = design_fir_filter(sr, "lowpass", cutoff_hz=1200.0, numtaps=101, window="hamming")
    filtered_audio = apply_fir_filter(dehummed_audio, fir_design, mode="zero_phase")

    assert len(filtered_audio) == num_samples
    assert np.all(np.isfinite(filtered_audio))

    # 5. FFT & Spectrogram Analysis
    freqs_clean, mag_clean, _ = compute_fft(norm_audio, sr, window="hann")
    freqs_noisy, mag_noisy, _ = compute_fft(noisy_audio, sr, window="hann")
    freqs_filt, mag_filt, _ = compute_fft(filtered_audio, sr, window="hann")

    # Peak detection: Clean has dominant peak at 400 Hz
    peak_clean_freq = freqs_clean[int(np.argmax(mag_clean))]
    assert abs(peak_clean_freq - 400.0) < 2.0

    # Noisy signal has strong 50 Hz energy
    bin_50 = int(np.argmin(np.abs(freqs_noisy - 50.0)))
    bin_50_filt = int(np.argmin(np.abs(freqs_filt - 50.0)))
    # Filtered signal must attenuate 50 Hz hum substantially compared to noisy
    assert mag_filt[bin_50_filt] < mag_noisy[bin_50] * 0.1

    # Spectrogram verification
    times_sp, freqs_sp, sxx_filt = compute_spectrogram(filtered_audio, sr, nperseg=512)
    assert sxx_filt.shape[0] == 512 // 2 + 1
    assert sxx_filt.shape[1] == len(times_sp)

    # 6. Visualization Figures
    fig_wave = plot_waveform(filtered_audio, sr)
    fig_fft = plot_fft_spectrum(freqs_filt, mag_filt)
    fig_sp = plot_spectrogram(times_sp, freqs_sp, sxx_filt)
    fig_comp = plot_signal_comparison(norm_audio, noisy_audio, filtered_audio, sr)

    w_fir, _, h_fir_db, _ = get_fir_frequency_response(fir_design)
    fig_fir = plot_filter_response(w_fir, h_fir_db, cutoff_hz=1200.0)

    for fig in (fig_wave, fig_fft, fig_sp, fig_comp, fig_fir):
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_e2e_restoration_quantitative_metrics():
    """
    Verify restoration performance metrics on aligned signals:
    - Input SNR & Output SNR
    - SNR Improvement (Delta SNR)
    - RMSE before & after restoration
    Ensures that filtering achieves measurable, objective signal enhancement.
    """
    sr = 16000
    duration = 1.0
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # 1. Clean Reference Signal (speech-like harmonic structure: 500 Hz, 1000 Hz, 1500 Hz)
    clean = (0.5 * np.sin(2 * np.pi * 500.0 * t) +
             0.3 * np.sin(2 * np.pi * 1000.0 * t) +
             0.2 * np.sin(2 * np.pi * 1500.0 * t))
    clean = normalize_audio(clean, target_peak=0.85)

    # 2. Injected Realistic Degradations
    # 50 Hz mains hum + 100 Hz harmonic
    degraded, _ = add_sinusoidal_hum(clean, sr, freq=50.0, amplitude=0.25, harmonics=[2])
    # 4500 Hz high-frequency tonal interference
    degraded, _ = add_high_frequency_noise(degraded, sr, freq=4500.0, amplitude=0.2)
    # Wideband White Gaussian Noise at 12 dB target SNR
    degraded, _ = add_white_gaussian_noise(degraded, target_snr_db=12.0, seed=123)

    # 3. Two-Stage Restoration Cascade (Zero-Phase for sample-exact alignment)
    # Stage 1: 50 Hz IIR Notch (Q=30)
    notch_d = design_iir_notch(sr, notch_freq=50.0, quality_factor=30.0)
    stage1 = apply_iir_filter(degraded, notch_d, mode="zero_phase")

    # Stage 2: FIR Lowpass (cutoff 2200 Hz, 121 taps, Blackman window)
    fir_d = design_fir_filter(sr, "lowpass", cutoff_hz=2200.0, numtaps=121, window="blackman")
    restored = apply_fir_filter(stage1, fir_d, mode="zero_phase")

    # Ensure equal length and non-empty
    assert len(clean) == len(degraded) == len(restored) == num_samples

    # 4. Compute Reference-Based Quantitative Metrics
    p_signal = float(np.mean(clean ** 2))

    # Error before filtering (input noise)
    error_in = degraded - clean
    p_noise_in = float(np.mean(error_in ** 2))
    snr_in = float(10.0 * np.log10(p_signal / p_noise_in))
    rmse_in = float(np.sqrt(p_noise_in))

    # Error after filtering (residual noise and distortion)
    error_out = restored - clean
    p_noise_out = float(np.mean(error_out ** 2))
    snr_out = float(10.0 * np.log10(p_signal / p_noise_out))
    rmse_out = float(np.sqrt(p_noise_out))

    snr_improvement = snr_out - snr_in

    # 5. Assertions: Quantifiable Objective Improvement
    # Filtering must substantially reduce noise power:
    assert snr_improvement > 5.0, f"Expected SNR gain > 5.0 dB, got {snr_improvement:.2f} dB"
    assert snr_out > snr_in, f"Output SNR ({snr_out:.2f} dB) must exceed input SNR ({snr_in:.2f} dB)"
    assert rmse_out < rmse_in, f"Output RMSE ({rmse_out:.4f}) must be strictly less than input RMSE ({rmse_in:.4f})"
    assert rmse_out < (rmse_in * 0.6), f"Output RMSE did not drop significantly: {rmse_out} vs {rmse_in}"

