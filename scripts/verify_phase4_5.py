"""
AudioClean DSP — Phase 4 and Phase 5 Verification Script
Executes reproducible DSP experiments A, B, C, and D:
- Exp A: Low-pass filtering (100 Hz preserved, 1000 Hz attenuated)
- Exp B: 50 Hz notch rejection (50 Hz hum rejected, 400 Hz tone preserved)
- Exp C: Multitone FFT comparison before and after filtering
- Exp D: Spectrogram generation and tonal localization
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.fft_analysis import compute_fft, compute_spectrogram
from src.iir_filters import apply_iir_filter, design_iir_filter, design_iir_notch
from src.visualization import (
    plot_fft_spectrum,
    plot_filter_response,
    plot_signal_comparison,
    plot_spectrogram,
    plot_waveform,
)


def run_experiment_a():
    print("=" * 70)
    print("EXPERIMENT A: BUTTERWORTH LOW-PASS FILTERING (100 Hz vs 1000 Hz)")
    print("=" * 70)
    sr = 8000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 100 Hz (Passband) + 1000 Hz (Stopband)
    tone_100 = 0.5 * np.sin(2 * np.pi * 100.0 * t)
    tone_1000 = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    signal = tone_100 + tone_1000

    # Butterworth LP order 6 with cutoff 300 Hz
    design = design_iir_filter(sample_rate=sr, filter_type="lowpass", cutoff_hz=300.0, order=6)
    filtered = apply_iir_filter(signal, design, mode="zero_phase")

    # Analyze FFT before and after
    f_in, mag_in, _ = compute_fft(signal, sample_rate=sr, window="hann")
    f_out, mag_out, _ = compute_fft(filtered, sample_rate=sr, window="hann")

    idx_100 = int(np.argmin(np.abs(f_in - 100.0)))
    idx_1000 = int(np.argmin(np.abs(f_in - 1000.0)))

    in_100 = mag_in[idx_100]
    out_100 = mag_out[idx_100]
    in_1000 = mag_in[idx_1000]
    out_1000 = mag_out[idx_1000]

    atten_100_db = 20 * np.log10(out_100 / in_100)
    atten_1000_db = 20 * np.log10(max(out_1000, 1e-12) / in_1000)

    print(f"Filter: Butterworth Lowpass (Order: 6, Cutoff: 300.0 Hz, fs: {sr} Hz)")
    print(f"{'Component':<15} | {'Input Mag':<12} | {'Output Mag':<12} | {'Attenuation (dB)':<18}")
    print("-" * 65)
    print(f"{'100 Hz (Pass)':<15} | {in_100:>10.4f} | {out_100:>10.4f} | {atten_100_db:>14.2f} dB")
    print(f"{'1000 Hz (Stop)':<15} | {in_1000:>10.4f} | {out_1000:>10.4f} | {atten_1000_db:>14.2f} dB")

    # Save verification plot
    plot_path = Path("results/plots/exp_a_lowpass_comparison.png")
    fig = plot_signal_comparison(
        signal, signal, filtered, sample_rate=sr, time_range=(0.1, 0.2),
        title="Exp A: Butterworth LP (100 Hz preserved, 1000 Hz removed)",
        save_path=plot_path, overwrite=True
    )
    print(f"Plot saved: {plot_path.name}\n")


def run_experiment_b():
    print("=" * 70)
    print("EXPERIMENT B: 50 HZ IIR NOTCH INTERFERENCE REJECTION")
    print("=" * 70)
    sr = 8000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    hum_50hz = 0.4 * np.sin(2 * np.pi * 50.0 * t)
    audio_400hz = 0.6 * np.sin(2 * np.pi * 400.0 * t)
    composite = hum_50hz + audio_400hz

    notch_design = design_iir_notch(sample_rate=sr, notch_freq=50.0, quality_factor=30.0)
    cleaned = apply_iir_filter(composite, notch_design, mode="zero_phase")

    f_in, mag_in, _ = compute_fft(composite, sample_rate=sr, window="hann")
    f_out, mag_out, _ = compute_fft(cleaned, sample_rate=sr, window="hann")

    idx_50 = int(np.argmin(np.abs(f_in - 50.0)))
    idx_400 = int(np.argmin(np.abs(f_in - 400.0)))

    in_50 = mag_in[idx_50]
    out_50 = mag_out[idx_50]
    in_400 = mag_in[idx_400]
    out_400 = mag_out[idx_400]

    atten_50_db = 20 * np.log10(max(out_50, 1e-12) / in_50)
    atten_400_db = 20 * np.log10(out_400 / in_400)

    print(f"Filter: 2nd-Order IIR Notch (f0: 50.0 Hz, Q: 30.0, delta_f: 1.67 Hz)")
    print(f"{'Component':<15} | {'Input Mag':<12} | {'Output Mag':<12} | {'Attenuation (dB)':<18}")
    print("-" * 65)
    print(f"{'50 Hz (Mains)':<15} | {in_50:>10.4f} | {out_50:>10.4f} | {atten_50_db:>14.2f} dB")
    print(f"{'400 Hz (Signal)':<15} | {in_400:>10.4f} | {out_400:>10.4f} | {atten_400_db:>14.2f} dB")

    plot_path = Path("results/plots/exp_b_notch_fft.png")
    fig = plot_fft_spectrum(
        f_out, 20 * np.log10(np.maximum(mag_out, 1e-6)),
        title="Exp B: FFT Spectrum After 50 Hz Notch Rejection",
        max_freq=1000.0, save_path=plot_path, overwrite=True
    )
    print(f"Plot saved: {plot_path.name}\n")


def run_experiment_c():
    print("=" * 70)
    print("EXPERIMENT C: MULTITONE FFT SPECTRAL REJECTION COMPARISON")
    print("=" * 70)
    sr = 10000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 400 Hz, 1200 Hz, 2400 Hz, 4000 Hz
    s = (
        0.5 * np.sin(2 * np.pi * 400.0 * t)
        + 0.5 * np.sin(2 * np.pi * 1200.0 * t)
        + 0.5 * np.sin(2 * np.pi * 2400.0 * t)
        + 0.5 * np.sin(2 * np.pi * 4000.0 * t)
    )

    # Butterworth bandpass filter passing [1000, 2600] Hz
    design = design_iir_filter(sample_rate=sr, filter_type="bandpass", cutoff_hz=(1000.0, 2600.0), order=4)
    filtered = apply_iir_filter(s, design, mode="zero_phase")

    freqs, mag_in, mag_db_in = compute_fft(s, sample_rate=sr)
    _, mag_out, mag_db_out = compute_fft(filtered, sample_rate=sr)

    tones = [400.0, 1200.0, 2400.0, 4000.0]
    print(f"Filter: Butterworth Bandpass [1000.0 - 2600.0 Hz] (Order: 4, fs: {sr} Hz)")
    print(f"{'Tone':<12} | {'Zone':<12} | {'Input (dB)':<12} | {'Output (dB)':<12} | {'Change (dB)':<12}")
    print("-" * 70)
    for f in tones:
        idx = int(np.argmin(np.abs(freqs - f)))
        pin = mag_db_in[idx]
        pout = mag_db_out[idx]
        zone = "Passband" if 1000.0 <= f <= 2600.0 else "Stopband"
        print(f"{f:>8.0f} Hz | {zone:<12} | {pin:>10.2f} | {pout:>10.2f} | {pout - pin:>10.2f} dB")

    plot_path = Path("results/plots/exp_c_multitone_fft.png")
    fig = plot_fft_spectrum(
        freqs, mag_db_out, title="Exp C: Multitone Spectrum After Bandpass (1000-2600 Hz)",
        max_freq=5000.0, save_path=plot_path, overwrite=True
    )
    print(f"Plot saved: {plot_path.name}\n")


def run_experiment_d():
    print("=" * 70)
    print("EXPERIMENT D: SHORT-TIME FOURIER TRANSFORM SPECTROGRAM LOCALIZATION")
    print("=" * 70)
    sr = 8000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1st second: 300 Hz tone; 2nd second: 1200 Hz tone + continuous 50 Hz hum
    tone_seq = np.zeros_like(t)
    tone_seq[:int(sr * 1.0)] = 0.6 * np.sin(2 * np.pi * 300.0 * t[:int(sr * 1.0)])
    tone_seq[int(sr * 1.0):] = 0.6 * np.sin(2 * np.pi * 1200.0 * t[int(sr * 1.0):])
    hum = 0.25 * np.sin(2 * np.pi * 50.0 * t)
    signal = tone_seq + hum

    times, freqs, Sxx_db = compute_spectrogram(signal, sample_rate=sr, nperseg=512, noverlap=256)

    # Find dominant frequency bin in first half (t < 1.0s) and second half (t > 1.0s)
    t_mid_idx = int(np.argmin(np.abs(times - 0.5)))
    t_end_idx = int(np.argmin(np.abs(times - 1.5)))

    peak_freq_1 = freqs[int(np.argmax(Sxx_db[:, t_mid_idx]))]
    peak_freq_2 = freqs[int(np.argmax(Sxx_db[:, t_end_idx]))]

    print(f"Spectrogram Matrix Shape: {Sxx_db.shape} (Freq bins: {len(freqs)}, Time frames: {len(times)})")
    print(f"Detected dominant tone at t=0.50 s: {peak_freq_1:.1f} Hz (Expected: 300.0 Hz)")
    print(f"Detected dominant tone at t=1.50 s: {peak_freq_2:.1f} Hz (Expected: 1200.0 Hz)")

    plot_path = Path("results/plots/exp_d_spectrogram.png")
    fig = plot_spectrogram(
        times, freqs, Sxx_db, title="Exp D: STFT Spectrogram (300 Hz -> 1200 Hz with 50 Hz Hum)",
        max_freq=2000.0, save_path=plot_path, overwrite=True
    )
    print(f"Plot saved: {plot_path.name}\n")


if __name__ == "__main__":
    run_experiment_a()
    run_experiment_b()
    run_experiment_c()
    run_experiment_d()
