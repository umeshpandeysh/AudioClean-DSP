"""AudioClean DSP — Permanent Quantitative Benchmark Script.

Executes standardized computational benchmarks comparing FIR and IIR digital filter
execution times across standardized signal durations: 1.0s, 5.0s, 30.0s, and 60.0s.

Methodology:
- Sample rate: 44,100 Hz (mono)
- Input: Calibrated multi-tone harmonic signal + White Gaussian Noise (10 dB SNR)
- FIR Design: 101-tap lowpass filter, Hamming window, fc = 1500 Hz
- IIR Design: 4th-order Butterworth lowpass filter (2 SOS), fc = 1500 Hz
- Modes: Both 'zero_phase' (filtfilt / forward-backward) and 'causal' (lfilter / sosfilt)
- Timing protocol: 5 warm-up iterations, 30 timed trials, median execution time (ms)
- Random seed: 42
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fir_filters import apply_fir_filter, design_fir_filter
from src.iir_filters import apply_iir_filter, design_iir_filter
from src.metrics import measure_processing_time
from src.noise_generator import add_white_gaussian_noise
from src.preprocessing import normalize_audio


def generate_benchmark_signal(duration_sec: float, sample_rate: int = 44100, seed: int = 42) -> np.ndarray:
    """Generate deterministic synthetic multi-tone signal with additive WGN."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0.0, duration_sec, num_samples, endpoint=False)
    # Clean multi-tone test signal (harmonic components)
    clean = (
        0.4 * np.sin(2 * np.pi * 440 * t)
        + 0.25 * np.sin(2 * np.pi * 880 * t)
        + 0.15 * np.sin(2 * np.pi * 1760 * t)
        + 0.10 * np.sin(2 * np.pi * 3520 * t)
    )
    clean = normalize_audio(clean, target_peak=0.8)
    noisy, _ = add_white_gaussian_noise(clean, target_snr_db=10.0, seed=seed)
    return noisy


def run_benchmark_suite(
    durations: List[float] = [1.0, 5.0, 30.0, 60.0],
    sample_rate: int = 44100,
    trials: int = 30,
    warmup: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete benchmark suite across signal durations and modes."""
    print("=" * 78)
    print("AUDIOCLEAN DSP — STANDARDIZED FILTER PERFORMANCE BENCHMARK")
    print(f"Sample Rate: {sample_rate:,} Hz | Trials: {trials} | Warmup: {warmup} | Seed: {seed}")
    print("=" * 78)

    # Standard filter designs
    fir_design = design_fir_filter(
        sample_rate=sample_rate,
        filter_type="lowpass",
        cutoff_hz=1500.0,
        numtaps=101,
        window="hamming",
    )

    iir_design = design_iir_filter(
        sample_rate=sample_rate,
        filter_type="lowpass",
        cutoff_hz=1500.0,
        order=4,
    )

    results: Dict[str, List[Dict[str, Any]]] = {
        "zero_phase": [],
        "causal": [],
    }

    modes = ["zero_phase", "causal"]

    for mode in modes:
        print(f"\n--- Benchmark Mode: {mode.upper()} ---")
        for dur in durations:
            sig = generate_benchmark_signal(dur, sample_rate, seed)
            n_samples = len(sig)

            # Benchmark FIR
            fir_timing = measure_processing_time(
                apply_fir_filter,
                sig,
                fir_design,
                mode=mode,
                num_trials=trials,
                warm_up=warmup,
            )

            # Benchmark IIR
            iir_timing = measure_processing_time(
                apply_iir_filter,
                sig,
                iir_design,
                mode=mode,
                num_trials=trials,
                warm_up=warmup,
            )

            fir_med = fir_timing["median_ms"]
            iir_med = iir_timing["median_ms"]
            speedup = fir_med / iir_med if iir_med > 0 else 0.0

            entry = {
                "duration_s": dur,
                "num_samples": n_samples,
                "fir_median_ms": fir_med,
                "fir_std_ms": fir_timing["std_ms"],
                "iir_median_ms": iir_med,
                "iir_std_ms": iir_timing["std_ms"],
                "speedup_ratio": speedup,
            }
            results[mode].append(entry)

            print(
                f"  Duration: {dur:4.1f}s ({n_samples:8,d} samples) | "
                f"FIR: {fir_med:7.2f} ms | IIR: {iir_med:6.2f} ms | "
                f"Speedup: {speedup:5.2f}x"
            )

    # Print Markdown Summary Table
    print("\n" + "=" * 78)
    print("CONSOLIDATED BENCHMARK SUMMARY (MARKDOWN)")
    print("=" * 78)
    print(
        "| Duration (s) | Samples | FIR Zero-Phase (ms) | IIR Zero-Phase (ms) | ZP Speedup | "
        "FIR Causal (ms) | IIR Causal (ms) | Causal Speedup |"
    )
    print(
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    )
    for i, dur in enumerate(durations):
        zp = results["zero_phase"][i]
        c = results["causal"][i]
        print(
            f"| **{dur:.1f} s** | `{zp['num_samples']:,}` | "
            f"`{zp['fir_median_ms']:.2f} +/- {zp['fir_std_ms']:.2f}` | "
            f"`{zp['iir_median_ms']:.2f} +/- {zp['iir_std_ms']:.2f}` | "
            f"**{zp['speedup_ratio']:.2f}x** | "
            f"`{c['fir_median_ms']:.2f} +/- {c['fir_std_ms']:.2f}` | "
            f"`{c['iir_median_ms']:.2f} +/- {c['iir_std_ms']:.2f}` | "
            f"**{c['speedup_ratio']:.2f}x** |"
        )
    print("=" * 78)

    return results


if __name__ == "__main__":
    run_benchmark_suite()
