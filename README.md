# AudioClean DSP — Digital Audio Noise Removal & Comparative Filter Performance Analysis

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.66.1-red.svg)](https://streamlit.io/)
[![Tests Passing](https://img.shields.io/badge/pytest-150%20passed-brightgreen.svg)](https://pytest.org/)

**AudioClean DSP** is a comprehensive undergraduate Digital Signal Processing (DSP) engineering application and laboratory workbench designed for digital audio noise removal and rigorous, scientifically grounded comparative performance analysis between **Finite Impulse Response (FIR)** and **Infinite Impulse Response (IIR)** digital filters.


---

## 1. Engineering Motivation & Problem Statement

Digital audio recordings in real-world environments are routinely corrupted by acoustic and electrical interference, including:
1. **Broadband thermal noise** (modeled as White Gaussian Noise).
2. **AC power-line interference** (50 Hz / 60 Hz hum and upper harmonic overtones from ground loops and electromagnetic coupling).
3. **High-frequency tonal artifacts** (sensor interference, carrier leakage, and switching transients).

While classical DSP offers both FIR and IIR filter families to address these corruptions, textbook comparisons frequently oversimplify their operational trade-offs or evaluate them under unequal, mathematically invalid boundary conditions.

### The AudioClean DSP Solution:
AudioClean DSP bridges theory and practical implementation by providing:
- **Calibrated Digital Filter Designers:** Windowed Type-I FIR and Second-Order Sections (SOS) Butterworth/Notch IIR filters.
- **Scientifically Grounded Head-to-Head Comparison:** Strict enforcement of cutoff alignment and causal evaluation intervals that account for filter startup transients and group delay.
- **Dual Analytical Workflows:** Clear separation between ground-truth reference restoration (Workflow A) and single-file real-world degraded analysis (Workflow B).
- **Interactive Engineering Dashboard:** An intuitive, full-featured Streamlit dashboard for real-time visualization, playback, and analysis with state persistence across reruns.

---

## 2. Key System Capabilities

- **Audio Ingestion & Signal Preprocessing:**
  - Robust ingestion of single-channel (mono) and multi-channel (stereo) `.wav` files in 64-bit floating-point precision.
  - Safe channel downmixing to mono and peak normalization with linked gain scaling across clean and degraded signals.
  - Signal validation catching non-finite samples (`NaN`, `Inf`), invalid sampling rates, and empty arrays.
- **Controlled Noise Generation Engine:**
  - White Gaussian Noise (WGN) calibrated to exact target Signal-to-Noise Ratio (SNR in dB) with random seed control.
  - Sinusoidal power-line hum (50 Hz or 60 Hz fundamental) with customizable harmonic series and decay rates.
  - High-frequency interference tones with strict Nyquist boundary enforcement ($f < f_s / 2$).
  - Additive composite noise combining multiple isolated degradation models with digital clipping prevention.
- **FIR Filter Design & Execution:**
  - Type-I linear-phase designs: Lowpass, Highpass, Bandpass, and Bandstop.
  - Standard classical windows: Hamming, Hann, and Blackman.
  - Dual operational modes: Causal filtering (`scipy.signal.lfilter` with exact group delay $\tau = (M-1)/2$) and Zero-Phase filtering (`scipy.signal.filtfilt`, offline two-pass forward-backward).
- **IIR Filter Design & SOS Stability Verification:**
  - Butterworth standard filters (Lowpass, Highpass, Bandpass, Bandstop) with orders from 1 to 10.
  - Second-Order Sections (SOS) representation eliminating coefficient quantization sensitivity.
  - Real-time Bounded-Input Bounded-Output (BIBO) pole stability checking ($|z_p| < 1.0$).
  - Dedicated high-Q IIR Notch filter for power-line hum rejection (> 35 dB attenuation).
  - Causal (`sosfilt`, real-time with non-linear phase) and Zero-Phase forward-backward (`sosfiltfilt`) filtering.
- **Rigorous Comparative Benchmark Engine:**
  - Automatic topology and cutoff parameter compatibility checks.
  - Causal FIR group-delay alignment using a shared valid reference interval $[\tau : N - \tau]$ ($N - 2\tau$ samples).
  - Statistically robust median execution latency measurement with speedup ratio calculation.
  - Comprehensive comparison matrix: Output SNR, $\Delta\text{SNR}$ improvement, Output RMSE, RMSE reduction percentage, and MAE.
- **Spectral Diagnostics & Time-Frequency Analysis:**
  - Single-sided calibrated FFT magnitude spectrum (dBFS or linear) with peak frequency detection.
  - Short-Time Fourier Transform (STFT) spectrogram with physically accurate units: $\text{dBFS/Hz}$ (power spectral density) or $\text{dBFS}$ (spectrum), with dynamic segment sizing preventing crashes on short audio.

---

## 3. Scientific Methodology & Integrity Standards

AudioClean DSP adheres strictly to core digital signal processing theorems:

### A. Cutoff Frequency Conventions
- **FIR Window Method:** Designs to the transition midpoint where $|H(e^{j\omega})| = 0.5$ (half-amplitude, **-6.02 dB** attenuation).
- **Butterworth IIR Design:** Designs to the half-power passband edge where $|H(e^{j\omega})| = 1/\sqrt{2}$ (**-3.01 dB** attenuation).
- *Engineering Note:* Nominal cutoff equivalence does not imply identical transition roll-offs. The comparative engine explicitly documents this convention rather than claiming identical frequency curves.

### B. Causal FIR vs. IIR Evaluation Interval
Causal FIR filtering introduces a constant group delay of $\tau = (M-1)/2$ samples. Evaluating the output from index 0 introduces severe transient startup errors. AudioClean DSP evaluates causal FIR filtering over the shared middle interval:
$$\text{Reference Interval} = x[\tau : N - \tau], \quad \text{FIR Output Interval} = y_{\text{FIR}}[2\tau : N], \quad \text{IIR Output Interval} = y_{\text{IIR}}[\tau : N - \tau]$$
All evaluated vectors have exactly $N - 2\tau$ samples, completely eliminating boundary bias without applying artificial non-causal shifts to IIR filters.

### C. Ground-Truth Policy
- **Workflow A (Clean Reference):** Objective restoration metrics ($\Delta\text{SNR}$, % RMSE reduction, MAE) are calculated only when a genuine clean reference is provided.
- **Workflow B (Real-World Noisy Audio):** In the absence of a clean reference, objective restoration metrics are strictly suppressed. The system provides waveforms, spectra, spectrograms, execution timings, and response curves without fabricating ground truth.

---

## 4. Repository Structure

```
AudioClean-DSP/
├── app.py                     # Streamlit Interactive Engineering Application
├── requirements.txt           # Project dependencies
├── pytest.ini                 # Pytest configuration
├── README.md                  # Comprehensive project engineering guide
│
├── src/                       # Audited DSP Core Engine
│   ├── __init__.py            # Package export interfaces
│   ├── audio_io.py            # Audio file loading, saving, and metadata
│   ├── preprocessing.py       # Normalization, downmixing, and validation
│   ├── noise_generator.py     # Calibrated WGN, mains hum, tone, and composite noise
│   ├── fir_filters.py         # Type-I linear-phase windowed FIR filters
│   ├── iir_filters.py         # Butterworth SOS and notch IIR filters with stability check
│   ├── fft_analysis.py        # Calibrated single-sided FFT and STFT spectrograms
│   ├── visualization.py       # Matplotlib figures for waveforms, spectra, and responses
│   ├── metrics.py             # SNR, RMSE, MAE, and timing measurement harness
│   └── comparison.py          # Rigorous FIR vs. IIR head-to-head comparison engine
│
├── tests/                     # Automated Test Suite (149 tests)
│   ├── __init__.py
│   ├── conftest.py            # Shared fixtures and synthesized test signals
│   ├── test_app.py            # Streamlit UI contracts, state, provenance & AppTest (20 tests)
│   ├── test_audio_io.py       # Audio I/O loading, saving, and info tests (9 tests)
│   ├── test_comparison.py     # Comparison engine and causal interval tests (9 tests)
│   ├── test_fft_analysis.py   # FFT calibration and spectrogram tests (15 tests)
│   ├── test_fir_filters.py    # FIR design, symmetry, and filtering tests (21 tests)
│   ├── test_iir_filters.py    # IIR design, SOS stability, and notch tests (23 tests)
│   ├── test_integration.py    # Full end-to-end DSP pipeline integration tests (2 tests)
│   ├── test_metrics.py        # Objective SNR, RMSE, MAE mathematical accuracy tests (21 tests)
│   ├── test_noise_generator.py# Noise synthesis and SNR calibration tests (7 tests)
│   ├── test_preprocessing.py  # Preprocessing, normalization, and validation tests (14 tests)
│   └── test_visualization.py  # Diagnostic plotting and figure export tests (8 tests)
│
├── docs/                      # Engineering Documentation
│   ├── architecture.md        # System architecture specification
│   ├── dsp_methodology.md     # Mathematical theory and DSP formulations
│   ├── phase4_remediation_report.md # Phase 4 remediation details
│   ├── phase5_validation_report.md # Phase 5 validation report and benchmarks
│   ├── test_coverage_matrix.md     # Traceability coverage matrix
│   └── final_delivery_report.md    # Final project closure report
│
├── scripts/                   # Verification & Benchmark Scripts
│   ├── benchmark_performance.py    # Standardized 30-trial benchmark (1s, 5s, 30s, 60s)
│   ├── verify_audit_remediation.py # 6-experiment mathematical verification pipeline
│   ├── verify_phase2.py
│   ├── verify_phase4_5.py
│   └── verify_phase6.py
│
└── results/                   # Experimental Artifacts
    ├── audio/                 # Output audio demonstration files
    └── plots/                 # Rendered diagnostic figures
```

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- Git installed.

### Step 1: Clone Repository
```powershell
git clone https://github.com/umeshpandeysh/AudioClean-DSP.git
cd AudioClean-DSP
```

### Step 2: Create & Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 6. Launching the Streamlit Application

Launch the interactive engineering dashboard locally:
```powershell
python -m streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### Navigating the Six Engineering Workspaces:
1. **Audio Workspace:** Ingest mono/stereo WAV files, toggle between Workflow A (clean ground truth) and Workflow B (real-world degraded), inspect audio metadata, listen to playback, downmix to mono, or normalize peak amplitude using linked gain scaling.
2. **Noise Generation Engine:** Synthesize calibrated White Gaussian Noise at target SNR, 50/60 Hz mains hum with harmonics, high-frequency tones, or composite noise combinations.
3. **FIR Filter Designer:** Configure lowpass, highpass, bandpass, or bandstop filters with selectable window functions and tap lengths; inspect magnitude and linear phase responses; compare causal vs. zero-phase audio.
4. **IIR Filter Designer:** Design Butterworth filters in Second-Order Sections (SOS) or dedicated 50/60 Hz hum notch filters; verify pole stability ($|z_p| < 1.0$); listen to and download filtered audio.
5. **FIR vs. IIR Comparative Evaluation:** Execute head-to-head benchmarking on identical audio; evaluate speedup ratios, output SNR improvement, and RMSE error reduction; inspect overlaid magnitude and phase curves.
6. **Signal Analysis & Diagnostics:** Analyze single-sided calibrated FFT spectra with dominant peak readouts; generate STFT spectrograms with calibrated physical units (`dBFS/Hz` or `dBFS`).

---

## 7. Running the Automated Test Suite

Run all 150 unit, integration, and contract tests:
```powershell
python -m pytest -v tests
```

Run test collection only:
```powershell
python -m pytest --collect-only -q
```

Run the standalone 6-experiment mathematical audit script:
```powershell
python scripts/verify_audit_remediation.py
```

Run the standardized 30-trial benchmark reproduction script:
```powershell
python scripts/benchmark_performance.py
```


---

## 8. Quantitative Performance Summary

Benchmark measurements performed on 64-bit Windows AMD64 using 30 trials with 5 warm-up iterations on synthetic composite test signals (440 Hz fundamental + 1200 Hz harmonic with 10 dB target WGN, $f_s = 44,100\text{ Hz}$):

| Audio Duration | Sample Count | FIR Zero-Phase (101 taps) | FIR Causal (101 taps) | IIR Zero-Phase (Order 4 SOS) | IIR Causal (Order 4 SOS) | IIR Speedup (Zero-Phase) | IIR Speedup (Causal) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0 s** | 44,100 | 3.67 ± 0.58 ms | 4.13 ± 5.48 ms | 1.13 ± 0.44 ms | 1.04 ± 0.63 ms | **3.25x** | **3.97x** |
| **5.0 s** | 220,500 | 27.08 ± 7.65 ms | 18.86 ± 11.21 ms | 7.38 ± 3.08 ms | 5.36 ± 0.50 ms | **3.67x** | **3.52x** |
| **30.0 s** | 1,323,000 | 146.15 ± 19.08 ms | 75.90 ± 36.83 ms | 43.64 ± 4.44 ms | 24.58 ± 5.99 ms | **3.35x** | **3.09x** |
| **60.0 s** | 2,646,000 | 257.47 ± 41.23 ms | 150.77 ± 29.79 ms | 89.34 ± 41.63 ms | 49.23 ± 6.57 ms | **2.88x** | **3.06x** |

*Note on Configuration Specificity:* These benchmarks reflect the specific comparison between a 101-tap FIR filter and a 4th-order Butterworth IIR filter. They demonstrate the recursion efficiency of low-order IIR sections compared to high-tap FIR convolution, but do not imply universal superiority under all filter specifications. Filter benchmarks represent direct time-domain filtering (`scipy.signal.lfilter`/`filtfilt` vs `sosfilt`/`sosfiltfilt`) and are reported separately from spectral transformations (FFT/STFT).

---

## 9. Current Limitations & Scope Boundaries
- **Supported Formats:** Uncompressed PCM WAV files (mono and stereo). Compressed formats (MP3, AAC, FLAC) are intentionally excluded to eliminate lossy perceptual codec artifacts from DSP analysis.
- **Filter Topologies:** Classic Type-I windowed FIR and Butterworth / Notch IIR filters. High-order Chebyshev, Elliptic, and adaptive filters (LMS/RLS) are outside the undergraduate scope of this project.
- **Phase Linearity Trade-Off:** Causal FIR filters provide strictly linear phase with constant group delay at the cost of higher tap counts and computational latency. Causal IIR filters provide lower computational latency at the cost of non-linear phase distortion.

---

## 10. Academic Context & Project Metadata
- **Author:** Umesh Pandey (Undergraduate Engineering Project)
- **Institution:** Department of Electronics & Communication Engineering
- **Supervision:** Digital Signal Processing Laboratory
