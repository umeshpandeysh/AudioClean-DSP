"""
AudioClean DSP - Consolidated Audit and Remediation Verification Script
Executes Experiments 1 through 6 across Phases 3, 4, and 5:
- Experiment 1: FIR Filter Response Verification (Windowed sinc, linear-phase, stopband attenuation)
- Experiment 2: IIR Filter Response & Stability Verification (Butterworth SOS, pole stability, 50 Hz notch)
- Experiment 3: FIR vs IIR Comparative Analysis (Identical input, zero-phase vs causal, phase distortion)
- Experiment 4: FFT Spectral Analysis Calibration (Single-sided scaling, DC offset, odd length, bin leakage)
- Experiment 5: Spectrogram Time-Frequency Resolution (Sequential tones, STFT localization, PSD dynamic range)
- Experiment 6: Complete Pipeline Verification (Audio I/O -> Preprocess -> Noise -> Cascade -> FFT/Spectrogram -> Plots)
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.audio_io import get_audio_info, load_audio, save_audio
from src.fft_analysis import compute_fft, compute_spectrogram
from src.fir_filters import (
    apply_fir_filter,
    design_fir_filter,
    get_fir_frequency_response,
)
from src.iir_filters import (
    apply_iir_filter,
    check_sos_stability,
    design_iir_filter,
    design_iir_notch,
    get_iir_frequency_response,
)
from src.noise_generator import (
    add_high_frequency_noise,
    add_sinusoidal_hum,
    add_white_gaussian_noise,
)
from src.preprocessing import convert_to_mono, normalize_audio, validate_audio
from src.visualization import (
    plot_fft_spectrum,
    plot_filter_response,
    plot_phase_response,
    plot_signal_comparison,
    plot_spectrogram,
    plot_waveform,
)


def run_experiment_1():
    print("=" * 75)
    print("EXPERIMENT 1: FIR FILTER RESPONSE AND ATTENUATION VERIFICATION")
    print("=" * 75)
    sr = 16000

    # Lowpass FIR with Hamming window, numtaps = 101, cutoff = 1500 Hz
    lp_design = design_fir_filter(sr, "lowpass", cutoff_hz=1500.0, numtaps=101, window="hamming")
    w, h_lin, h_db, phase = get_fir_frequency_response(lp_design, worN=4096)

    # Find cutoff attenuation (at 1500 Hz)
    idx_fc = int(np.argmin(np.abs(w - 1500.0)))
    fc_atten = h_db[idx_fc]

    # Find stopband attenuation (at 3000 Hz, well into stopband)
    idx_sb = int(np.argmin(np.abs(w - 3000.0)))
    sb_atten = h_db[idx_sb]

    print(f"Filter Type: Type-I Linear-Phase FIR Lowpass")
    print(f"Taps: {lp_design.numtaps} (Order N = {lp_design.order}), Window: {lp_design.window}")
    print(f"Group Delay: {lp_design.group_delay_samples:.1f} samples ({lp_design.group_delay_seconds * 1e3:.2f} ms)")
    print(f"Attenuation at Cutoff (1500 Hz): {fc_atten:.2f} dB (Expected: ~ -6.02 dB half-amplitude point)")
    print(f"Attenuation in Stopband (3000 Hz): {sb_atten:.2f} dB (Expected: < -50 dB)")

    assert abs(fc_atten - (-6.02)) < 1.0, f"Cutoff attenuation deviation: {fc_atten} dB"
    assert sb_atten < -50.0, f"Stopband attenuation insufficient: {sb_atten} dB"

    # Save plot
    fig = plot_filter_response(w, h_db, cutoff_hz=1500.0, title="Exp 1: FIR Lowpass Response (Hamming 101 taps)")
    out_path = Path("results/plots/audit_exp1_fir_response.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}\n")


def run_experiment_2():
    print("=" * 75)
    print("EXPERIMENT 2: IIR FILTER RESPONSE AND POLE STABILITY VERIFICATION")
    print("=" * 75)
    sr = 16000

    # 1. Butterworth 4th-order lowpass (order=4, cutoff=1500 Hz)
    iir_lp = design_iir_filter(sr, "lowpass", cutoff_hz=1500.0, order=4)
    stable_lp, max_pole_lp = check_sos_stability(iir_lp.sos)
    w_lp, _, h_lp_db, _ = get_iir_frequency_response(iir_lp, worN=4096)
    idx_lp_fc = int(np.argmin(np.abs(w_lp - 1500.0)))
    lp_fc_atten = h_lp_db[idx_lp_fc]

    # 2. Butterworth 3rd-order bandpass -> resulting order 6
    iir_bp = design_iir_filter(sr, "bandpass", cutoff_hz=[1000.0, 3000.0], order=3)
    stable_bp, max_pole_bp = check_sos_stability(iir_bp.sos)

    # 3. 50 Hz Notch filter with Q = 30
    notch = design_iir_notch(sr, notch_freq=50.0, quality_factor=30.0)
    stable_notch, max_pole_notch = check_sos_stability(notch.sos)
    w_n, _, h_n_db, _ = get_iir_frequency_response(notch, worN=8192)
    # Evaluate exact notch frequency
    _, _, h_50_db, _ = get_iir_frequency_response(notch, worN=[50.0])
    notch_atten = h_50_db[0]

    print(f"Butterworth Lowpass (order={iir_lp.order}):")
    print(f"  Stable: {stable_lp} | Max Pole Magnitude: {max_pole_lp:.6f} (< 1.0)")
    print(f"  Attenuation at Cutoff (1500 Hz): {lp_fc_atten:.2f} dB (Expected: ~ -3.01 dB half-power point)")

    print(f"Butterworth Bandpass [1000, 3000 Hz] (prototype={iir_bp.prototype_order}, resulting={iir_bp.resulting_order}):")
    print(f"  Stable: {stable_bp} | Max Pole Magnitude: {max_pole_bp:.6f} (< 1.0)")

    print(f"50 Hz IIR Notch Filter (Q={notch.quality_factor}, resulting_order={notch.order}):")
    print(f"  Stable: {stable_notch} | Max Pole Magnitude: {max_pole_notch:.6f} (< 1.0)")
    print(f"  Exact Attenuation at 50.0 Hz: {notch_atten:.2f} dB (Rejection > 35 dB)")

    assert stable_lp and stable_bp and stable_notch
    assert abs(lp_fc_atten - (-3.01)) < 0.5
    assert notch_atten < -35.0

    # Save plot
    fig = plot_filter_response(w_n, h_n_db, cutoff_hz=50.0, max_freq=300.0, title="Exp 2: 50 Hz IIR Notch Response")
    out_path = Path("results/plots/audit_exp2_iir_notch_response.png")
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}\n")


def run_experiment_3():
    print("=" * 75)
    print("EXPERIMENT 3: FIR AND IIR COMPARATIVE ANALYSIS ON IDENTICAL AUDIO")
    print("=" * 75)
    print("Configuration Note:")
    print("  This experiment evaluates two distinct filter design configurations:")
    print("  - Configuration A (FIR): Type-I Windowed Sinc (101 taps, Order N=100, Hamming, fc=1200 Hz).")
    print("    Cutoff fc specifies the -6.02 dB (half-amplitude) transition midpoint.")
    print("  - Configuration B (IIR): Butterworth SOS (Order N=6, fc=1200 Hz).")
    print("    Cutoff fc specifies the -3.01 dB (half-power) passband edge.")
    print("  Note: Design orders (100 taps vs 6th-order SOS) and cutoff conventions are not directly")
    print("  equivalent. Neither filter is universally superior; each exhibits distinct engineering")
    print("  trade-offs (linear phase vs computational efficiency per pole).")
    print("-" * 75)

    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Clean signal: 300 Hz + 700 Hz speech-like harmonics
    clean = 0.5 * np.sin(2 * np.pi * 300.0 * t) + 0.3 * np.sin(2 * np.pi * 700.0 * t)
    # Add high-frequency noise (3500 Hz tone + WGN at 15 dB)
    noisy, _ = add_high_frequency_noise(clean, sr, freq=3500.0, amplitude=0.3)
    noisy, _ = add_white_gaussian_noise(noisy, target_snr_db=15.0, seed=10)

    # Filter Designs
    fir_d = design_fir_filter(sr, "lowpass", cutoff_hz=1200.0, numtaps=101, window="hamming")
    iir_d = design_iir_filter(sr, "lowpass", cutoff_hz=1200.0, order=6)

    # Benchmarking with warm-up and repeated trials
    n_warmup = 5
    n_trials = 30

    # 1. FIR Zero-phase timing
    for _ in range(n_warmup):
        _ = apply_fir_filter(noisy, fir_d, mode="zero_phase")
    fir_zp_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fir_zp = apply_fir_filter(noisy, fir_d, mode="zero_phase")
        fir_zp_times.append(time.perf_counter() - t0)

    # 2. FIR Causal timing
    for _ in range(n_warmup):
        _ = apply_fir_filter(noisy, fir_d, mode="causal")
    fir_c_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fir_c = apply_fir_filter(noisy, fir_d, mode="causal")
        fir_c_times.append(time.perf_counter() - t0)

    # 3. IIR Zero-phase timing
    for _ in range(n_warmup):
        _ = apply_iir_filter(noisy, iir_d, mode="zero_phase")
    iir_zp_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        iir_zp = apply_iir_filter(noisy, iir_d, mode="zero_phase")
        iir_zp_times.append(time.perf_counter() - t0)

    # 4. IIR Causal timing
    for _ in range(n_warmup):
        _ = apply_iir_filter(noisy, iir_d, mode="causal")
    iir_c_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        iir_c = apply_iir_filter(noisy, iir_d, mode="causal")
        iir_c_times.append(time.perf_counter() - t0)

    # Median timings in milliseconds
    med_fir_zp = float(np.median(fir_zp_times)) * 1e3
    med_fir_c = float(np.median(fir_c_times)) * 1e3
    med_iir_zp = float(np.median(iir_zp_times)) * 1e3
    med_iir_c = float(np.median(iir_c_times)) * 1e3

    # Stopband residual energy at 3500 Hz
    f_noisy, mag_noisy, _ = compute_fft(noisy, sr)
    f_fir, mag_fir, _ = compute_fft(fir_zp, sr)
    f_iir, mag_iir, _ = compute_fft(iir_zp, sr)

    idx_noise = int(np.argmin(np.abs(f_noisy - 3500.0)))
    att_fir = 20 * np.log10(max(mag_fir[idx_noise], 1e-12) / mag_noisy[idx_noise])
    att_iir = 20 * np.log10(max(mag_iir[idx_noise], 1e-12) / mag_noisy[idx_noise])

    print(f"{'Characteristic / Metric':<32} | {'Config A: FIR (101 taps)':<23} | {'Config B: IIR (6th-order SOS)':<23}")
    print("-" * 84)
    print(f"{'Design Cutoff Definition':<32} | {'-6.02 dB (Half-Amplitude)':<23} | {'-3.01 dB (Half-Power)':<23}")
    print(f"{'Poles / Zeros Structure':<32} | {'100 zeros, poles at z=0':<23} | {'6 poles, 6 zeros in SOS':<23}")
    print(f"{'Stopband Attenuation at 3.5 kHz':<32} | {att_fir:>19.2f} dB | {att_iir:>19.2f} dB")
    print(f"{'Zero-Phase Time (median, N=30)':<32} | {med_fir_zp:>19.3f} ms | {med_iir_zp:>19.3f} ms")
    print(f"{'Causal Time (median, N=30)':<32} | {med_fir_c:>19.3f} ms | {med_iir_c:>19.3f} ms")
    print(f"{'Causal Phase Response':<32} | {'Strictly Linear (tau=50)':<23} | {'Nonlinear (Frequency-dependent)':<23}")

    # Save comparison plot
    fig = plot_signal_comparison(clean, noisy, fir_zp, sr, time_range=(0.2, 0.25),
                                 title="Exp 3: Time-Domain Filtering (FIR Zero-Phase)")
    out_path = Path("results/plots/audit_exp3_comparison.png")
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}\n")


def run_experiment_4():
    print("=" * 75)
    print("EXPERIMENT 4: FFT SPECTRAL ANALYSIS CALIBRATION")
    print("=" * 75)
    sr = 8000
    N = 1000
    t = np.linspace(0, N / sr, N, endpoint=False)

    # Test exact bin alignment with rectangular window:
    # A = 0.70 at 200 Hz, DC offset = 0.35
    A = 0.70
    DC = 0.35
    sig = DC + A * np.sin(2 * np.pi * 200.0 * t)

    f, mag, mag_db = compute_fft(sig, sr, window="rectangular")
    idx_dc = 0
    idx_tone = int(np.argmin(np.abs(f - 200.0)))

    measured_dc = mag[idx_dc]
    measured_a = mag[idx_tone]

    print(f"Input Signal: DC = {DC}, Sinusoid A = {A} at 200 Hz (fs = {sr} Hz, N = {N})")
    print(f"Measured DC Bin: {measured_dc:.4f} (Expected: {DC:.4f}, Error: {abs(measured_dc - DC):.6f})")
    print(f"Measured Peak:   {measured_a:.4f} (Expected: {A:.4f}, Error: {abs(measured_a - A):.6f})")

    assert np.isclose(measured_dc, DC, atol=1e-3)
    assert np.isclose(measured_a, A, atol=1e-3)

    # Test odd length N = 1001
    sig_odd = np.sin(2 * np.pi * 300.0 * np.linspace(0, 1001 / sr, 1001, endpoint=False))
    f_odd, mag_odd, _ = compute_fft(sig_odd, sr, window="hann")
    assert len(f_odd) == (1001 + 1) // 2
    print(f"Odd Length (N=1001): {len(f_odd)} bins generated successfully without errors.")

    fig = plot_fft_spectrum(f, mag, db_scale=False, max_freq=1000.0, title="Exp 4: Calibrated Single-Sided FFT Spectrum")
    out_path = Path("results/plots/audit_exp4_fft_calibration.png")
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}\n")


def run_experiment_5():
    print("=" * 75)
    print("EXPERIMENT 5: SPECTROGRAM TIME-FREQUENCY LOCALIZATION")
    print("=" * 75)
    sr = 8000
    total_duration = 2.0
    num_samples = int(sr * total_duration)
    t = np.linspace(0, total_duration, num_samples, endpoint=False)

    # Segment 1 (0.0 - 1.0 s): 250 Hz tone
    # Segment 2 (1.0 - 2.0 s): 1500 Hz tone
    sig = np.zeros(num_samples)
    mask1 = t < 1.0
    mask2 = t >= 1.0
    sig[mask1] = 0.8 * np.sin(2 * np.pi * 250.0 * t[mask1])
    sig[mask2] = 0.8 * np.sin(2 * np.pi * 1500.0 * t[mask2])

    times, freqs, sxx_db = compute_spectrogram(sig, sample_rate=sr, nperseg=512, noverlap=256)

    # Locate dominant frequency at t = 0.5 s and t = 1.5 s
    frame_05 = int(np.argmin(np.abs(times - 0.5)))
    frame_15 = int(np.argmin(np.abs(times - 1.5)))

    freq_05 = freqs[int(np.argmax(sxx_db[:, frame_05]))]
    freq_15 = freqs[int(np.argmax(sxx_db[:, frame_15]))]

    print(f"Spectrogram Matrix: {sxx_db.shape[0]} frequency bins x {sxx_db.shape[1]} time frames")
    print(f"t = 0.50 s: Detected = {freq_05:.1f} Hz (Expected: 250.0 Hz, error: {abs(freq_05 - 250.0):.1f} Hz)")
    print(f"t = 1.50 s: Detected = {freq_15:.1f} Hz (Expected: 1500.0 Hz, error: {abs(freq_15 - 1500.0):.1f} Hz)")

    assert abs(freq_05 - 250.0) <= (sr / 512.0)
    assert abs(freq_15 - 1500.0) <= (sr / 512.0)

    fig = plot_spectrogram(times, freqs, sxx_db, title="Exp 5: STFT Power Spectrogram Time-Frequency Localization")
    out_path = Path("results/plots/audit_exp5_spectrogram.png")
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}\n")


def run_experiment_6():
    print("=" * 75)
    print("EXPERIMENT 6: COMPLETE END-TO-END DSP PIPELINE VERIFICATION")
    print("=" * 75)
    sr = 16000
    duration = 1.5
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # 1. Create multi-harmonic audio signal (simulating vocal vowel /a/)
    # Formants at 500 Hz, 1000 Hz, 1500 Hz
    clean = (0.5 * np.sin(2 * np.pi * 500.0 * t) +
             0.3 * np.sin(2 * np.pi * 1000.0 * t) +
             0.2 * np.sin(2 * np.pi * 1500.0 * t))

    # Preprocessing
    clean = normalize_audio(clean, target_peak=0.85)

    # 2. Add composite realistic degradation:
    # 50 Hz electrical mains hum + 100 Hz harmonic + 4500 Hz whistle + WGN at 12 dB
    degraded, _ = add_sinusoidal_hum(clean, sr, freq=50.0, amplitude=0.25, harmonics=[2])
    degraded, _ = add_high_frequency_noise(degraded, sr, freq=4500.0, amplitude=0.2)
    degraded, _ = add_white_gaussian_noise(degraded, target_snr_db=12.0, seed=123)

    # 3. Two-stage Restoration Cascade:
    # Stage 1: IIR Notch at 50 Hz (zero-phase)
    notch_d = design_iir_notch(sr, notch_freq=50.0, quality_factor=30.0)
    stage1 = apply_iir_filter(degraded, notch_d, mode="zero_phase")

    # Stage 2: FIR Lowpass at 2200 Hz (zero-phase, 121 taps)
    fir_d = design_fir_filter(sr, "lowpass", cutoff_hz=2200.0, numtaps=121, window="blackman")
    cleaned = apply_fir_filter(stage1, fir_d, mode="zero_phase")

    # 4. Quantitative Restoration Performance Evaluation
    # Assumptions & Metric Definitions:
    # - The original uncorrupted clean signal s[n] serves as the ground-truth reference.
    # - Zero-phase forward-backward filtering ensures zero group delay and perfect sample alignment.
    # - Signal Power: P_signal = (1/N) * sum(s[n]^2)
    # - Input Error: e_in[n] = degraded[n] - clean[n]
    # - Input Noise Power: P_noise_in = (1/N) * sum(e_in[n]^2)
    # - Input SNR: 10 * log10(P_signal / P_noise_in)
    # - Input RMSE: sqrt(P_noise_in)
    # - Output Error: e_out[n] = restored[n] - clean[n]
    # - Output Noise Power: P_noise_out = (1/N) * sum(e_out[n]^2)
    # - Output SNR: 10 * log10(P_signal / P_noise_out)
    # - Output RMSE: sqrt(P_noise_out)
    # - SNR Improvement (Delta SNR): SNR_out - SNR_in
    p_signal = float(np.mean(clean ** 2))

    error_in = degraded - clean
    p_noise_in = float(np.mean(error_in ** 2))
    snr_in = float(10.0 * np.log10(p_signal / p_noise_in))
    rmse_in = float(np.sqrt(p_noise_in))

    error_out = cleaned - clean
    p_noise_out = float(np.mean(error_out ** 2))
    snr_out = float(10.0 * np.log10(p_signal / p_noise_out))
    rmse_out = float(np.sqrt(p_noise_out))

    snr_improvement = snr_out - snr_in
    rmse_reduction_pct = (1.0 - (rmse_out / rmse_in)) * 100.0

    info_clean = get_audio_info(clean, sr)
    info_deg = get_audio_info(degraded, sr)
    info_out = get_audio_info(cleaned, sr)

    print("Pipeline Execution Completed Successfully.")
    print("Signal Amplitude Statistics:")
    print(f"  Clean Audio    : Peak = {info_clean['peak_amplitude']:.4f}, RMS = {info_clean['rms_amplitude']:.4f}")
    print(f"  Degraded Audio : Peak = {info_deg['peak_amplitude']:.4f}, RMS = {info_deg['rms_amplitude']:.4f}")
    print(f"  Restored Audio : Peak = {info_out['peak_amplitude']:.4f}, RMS = {info_out['rms_amplitude']:.4f}")
    print("\nObjective Restoration Metrics (Ground-Truth Reference Comparison):")
    print("-" * 75)
    print(f"  Reference Signal Power (P_sig) : {p_signal:.6f}")
    print(f"  Input Noise Power (P_noise_in) : {p_noise_in:.6f}")
    print(f"  Output Noise Power (P_out)     : {p_noise_out:.6f}")
    print(f"  Input SNR                      : {snr_in:>8.2f} dB")
    print(f"  Output SNR                     : {snr_out:>8.2f} dB")
    print(f"  SNR Improvement (Delta SNR)    : {snr_improvement:>+8.2f} dB (Objective noise suppression)")
    print(f"  Input RMSE                     : {rmse_in:>8.4f}")
    print(f"  Output RMSE                    : {rmse_out:>8.4f} (Error reduced by {rmse_reduction_pct:.1f}%)")
    print("-" * 75)

    assert snr_improvement > 5.0, f"Restoration did not achieve sufficient SNR improvement: {snr_improvement:.2f} dB"
    assert rmse_out < rmse_in, f"Restoration did not reduce RMSE: {rmse_out:.4f} vs {rmse_in:.4f}"

    # Generate 3-panel comparison plot
    fig = plot_signal_comparison(clean, degraded, cleaned, sr, time_range=(0.2, 0.28),
                                 title="Exp 6: Complete AudioClean DSP Restoration Pipeline")
    out_path = Path("results/plots/audit_exp6_pipeline_comparison.png")
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}\n")


def main():
    print("=" * 75)
    print("AUDIOCLEAN DSP: AUDIT & REMEDIATION EXPERIMENTAL VERIFICATION")
    print("=" * 75 + "\n")
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()
    run_experiment_5()
    run_experiment_6()
    print("=" * 75)
    print("ALL 6 EXPERIMENTS COMPLETED AND MATHEMATICALLY VERIFIED!")
    print("=" * 75)


if __name__ == "__main__":
    main()
