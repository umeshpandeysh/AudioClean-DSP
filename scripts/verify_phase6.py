"""
AudioClean DSP - Phase 6 Quantitative Evaluation Verification Script
Executes an objective, reproducible end-to-end performance benchmarking experiment:
1. Synthesizes a clean speech-like harmonic test signal.
2. Superimposes controlled acoustic noise (50 Hz electrical hum + 4.5 kHz tonal hiss + calibrated WGN).
3. Applies a cascade restoration filter (IIR notch + FIR lowpass in zero-phase mode).
4. Calculates objective restoration metrics (Input SNR, Output SNR, SNR Gain, RMSE, MAE).
5. Benchmarks execution time with isolated warm-up and repeated trials.
6. Generates diagnostic plots and outputs a formatted academic engineering report.
"""

import sys
from pathlib import Path

# Add project root to sys.path for standalone script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.fir_filters import apply_fir_filter, design_fir_filter
from src.iir_filters import apply_iir_filter, design_iir_notch
from src.metrics import evaluate_audio_quality, measure_processing_time
from src.noise_generator import (
    add_high_frequency_noise,
    add_sinusoidal_hum,
    add_white_gaussian_noise,
)
from src.preprocessing import normalize_audio
from src.visualization import plot_signal_comparison


def run_phase6_experiment():
    print("=" * 80)
    print("AUDIOCLEAN DSP — PHASE 6: QUANTITATIVE PERFORMANCE EVALUATION")
    print("Deterministic End-to-End Filter Enhancement Benchmarking")
    print("=" * 80)

    # 1. Experimental Parameters
    sample_rate = 16000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    print(f"\n1. SIGNAL SYNTHESIS")
    print(f"   Sampling Frequency (fs) : {sample_rate} Hz")
    print(f"   Signal Duration         : {duration_sec:.2f} s ({num_samples} samples)")

    # Clean signal: Speech-like fundamental and harmonics (500 Hz, 1000 Hz, 1500 Hz)
    clean = (
        0.50 * np.sin(2 * np.pi * 500.0 * t)
        + 0.30 * np.sin(2 * np.pi * 1000.0 * t)
        + 0.20 * np.sin(2 * np.pi * 1500.0 * t)
    )
    clean = normalize_audio(clean, target_peak=0.85)
    print(f"   Clean Reference Signal  : Formants at 500 Hz, 1000 Hz, 1500 Hz (Normalized peak: 0.85)")

    # 2. Controlled Degradations
    print(f"\n2. CONTROLLED NOISE INJECTION")
    # A. 50 Hz powerline hum + 100 Hz second harmonic (amplitude = 0.25)
    degraded, _ = add_sinusoidal_hum(clean, sample_rate, freq=50.0, amplitude=0.25, harmonics=[2])
    # B. 4500 Hz high-frequency switching hiss (amplitude = 0.20)
    degraded, _ = add_high_frequency_noise(degraded, sample_rate, freq=4500.0, amplitude=0.20)
    # C. Wideband White Gaussian Noise at 12.0 dB target SNR (seed = 42 for strict reproducibility)
    degraded, _ = add_white_gaussian_noise(degraded, target_snr_db=12.0, seed=42)
    print(f"   - 50 Hz Mains Hum + 100 Hz 2nd Harmonic (Amplitude: 0.25)")
    print(f"   - 4500 Hz High-Frequency Switching Noise (Amplitude: 0.20)")
    print(f"   - Additive White Gaussian Noise (Target SNR: 12.0 dB, Seed: 42)")

    # 3. Design Restoration Filters
    print(f"\n3. RESTORATION FILTER DESIGN")
    # Stage 1: IIR Notch filter (50 Hz rejection, Q = 30.0)
    notch_filter = design_iir_notch(sample_rate, notch_freq=50.0, quality_factor=30.0)
    print(f"   Stage 1: 50 Hz IIR Notch Filter (Q = 30.0, Second-Order Sections)")

    # Stage 2: FIR Lowpass filter (Cutoff = 2200 Hz, 121 taps, Blackman window)
    fir_filter = design_fir_filter(
        sample_rate,
        filter_type="lowpass",
        cutoff_hz=2200.0,
        numtaps=121,
        window="blackman",
    )
    print(f"   Stage 2: Type-I FIR Lowpass Filter (fc = 2200 Hz, 121 taps, Blackman Window)")

    # 4. Processing Pipeline Callable (Isolated for Computation Timing)
    def filtering_pipeline(x):
        # Stage 1: IIR Notch
        s1 = apply_iir_filter(x, notch_filter, mode="zero_phase")
        # Stage 2: FIR Lowpass
        s2 = apply_fir_filter(s1, fir_filter, mode="zero_phase")
        return s2

    # Benchmark filtering execution time
    timing_stats = measure_processing_time(
        filtering_pipeline,
        degraded,
        num_trials=30,
        warm_up=5,
    )

    # Execute filtering to produce restored audio
    restored = filtering_pipeline(degraded)

    # 5. Quantitative Evaluation
    print(f"\n4. OBJECTIVE PERFORMANCE EVALUATION")
    eval_result = evaluate_audio_quality(clean, degraded, restored, timing_result=timing_stats)

    print("-" * 80)
    print(f"  Input SNR (Degraded vs Clean)    : {eval_result.input_snr_db:8.2f} dB")
    print(f"  Output SNR (Restored vs Clean)   : {eval_result.output_snr_db:8.2f} dB")
    print(f"  SNR Improvement (Delta SNR)      : {eval_result.snr_improvement_db:+8.2f} dB")
    print("-" * 80)
    print(f"  Input RMSE                       : {eval_result.input_rmse:8.4f}")
    print(f"  Output RMSE                      : {eval_result.output_rmse:8.4f}")
    print(f"  RMSE Reduction                   : {eval_result.rmse_reduction_percent:8.1f} %")
    print("-" * 80)
    print(f"  Input MAE                        : {eval_result.input_mae:8.4f}")
    print(f"  Output MAE                       : {eval_result.output_mae:8.4f}")
    print("-" * 80)
    print(f"  Filtering Computation Time       :")
    print(f"    - Median Latency               : {timing_stats['median_ms']:.3f} ms ({timing_stats['median_seconds']*1000:.3f} ms)")
    print(f"    - Mean Latency                 : {timing_stats['mean_ms']:.3f} ms +/- {timing_stats['std_ms']:.3f} ms")
    print(f"    - Range [Min, Max]             : [{timing_stats['min_ms']:.3f} ms, {timing_stats['max_ms']:.3f} ms]")
    print(f"    - Trials / Warm-up Iterations  : {timing_stats['num_trials']} trials ({timing_stats['warm_up']} warm-ups)")
    print("-" * 80)

    # 6. Save Comparative Diagnostic Waveform Plot
    results_dir = Path("results/plots")
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / "phase6_metrics_experiment.png"

    fig = plot_signal_comparison(
        original=clean,
        noisy=degraded,
        filtered=restored,
        sample_rate=sample_rate,
        title="AudioClean DSP — Phase 6 Objective Filtering Benchmark",
        time_range=(0.10, 0.15),  # 50 ms zoom for clear waveform comparison
        save_path=plot_path,
        overwrite=True,
    )
    plt.close(fig)
    print(f"\n5. DIAGNOSTIC PLOT SAVED")
    print(f"   Figure Path : {plot_path}")

    # 7. Verification Assertions
    assert eval_result.snr_improvement_db > 5.0, "SNR improvement failed to exceed 5.0 dB threshold."
    assert eval_result.output_rmse < eval_result.input_rmse, "Output RMSE was not reduced."
    assert eval_result.rmse_reduction_percent > 50.0, "RMSE reduction was under 50%."
    print(f"\n" + "=" * 80)
    print("PHASE 6 VERIFICATION EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_phase6_experiment()
