# AudioClean DSP

### Digital Audio Noise Removal & Comparative Filter Performance Analysis Using FIR and IIR Digital Filters

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.66.1-red.svg)](https://audioclean-dsp-e8gienueiutdbjnw2apth7.streamlit.app/)
[![Pytest Suite](https://img.shields.io/badge/pytest-150%20tests-brightgreen.svg)](https://pytest.org/)
[![CI](https://github.com/umeshpandeysh/AudioClean-DSP/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/umeshpandeysh/AudioClean-DSP/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AudioClean DSP** is a comprehensive Python-based Digital Signal Processing (DSP) engineering application and laboratory workbench designed for digital audio noise removal and rigorous, scientifically grounded comparative performance analysis between **Finite Impulse Response (FIR)** and **Infinite Impulse Response (IIR)** digital filters.

---

## 🌐 Live Application

> ### [🚀 Live Demo — Launch AudioClean DSP](https://audioclean-dsp-e8gienueiutdbjnw2apth7.streamlit.app/)
> **GitHub Repository:** [https://github.com/umeshpandeysh/AudioClean-DSP](https://github.com/umeshpandeysh/AudioClean-DSP)
> *Access the interactive engineering application directly in your web browser. Ingest test audio, inject calibrated acoustic and electrical noise models, design Type-I FIR and Butterworth/Notch IIR filters, and evaluate real-time spectral diagnostics.*

---

## 1. Project Overview

In real-world acoustic capture and communication systems, digital audio recordings are routinely contaminated by electromagnetic and physical noise sources:
1. **Broadband thermal and acoustic noise:** Modeled as White Gaussian Noise (WGN).
2. **AC electrical interference:** 50 Hz or 60 Hz power-line hum and upper harmonic overtones introduced through electrical ground loops and inductive coupling.
3. **High-frequency tonal artifacts:** Sensor carrier leakage, high-frequency hiss, and transducer switching transients.

Digital filtering is a fundamental engineering discipline required to suppress unwanted noise while preserving speech clarity, musical timbre, and signal fidelity. However, standard textbook comparisons between **FIR** and **IIR** filters often oversimplify operational trade-offs or evaluate them under unequal, mathematically invalid boundary conditions.

### Purpose and Scope:
AudioClean DSP was engineered to bridge textbook theory and practical DSP implementation:
- **Controlled Degradation & Restoration:** Generate, process, analyze, and restore contaminated audio using calibrated linear-time-invariant (LTI) filters.
- **Fair, Head-to-Head Architectural Comparison:** Evaluate FIR versus IIR filters across computational latency, numerical stability, amplitude attenuation, and phase linearity.
- **Scientifically Grounded Evaluation Intervals:** Account for FIR group delay ($\tau = (M-1)/2$) and startup transients to ensure sample-accurate alignment without artificial distortion.
- **Interactive Educational Dashboard:** Provide an accessible, visual workbench for undergraduate engineering demonstration, laboratory research, and audio DSP portfolio evaluation.

---

## 2. Key Features

- **Audio Ingestion & Signal Preprocessing:**
  - Robust ingestion of single-channel (mono) and multi-channel (stereo) uncompressed `.wav` files in 64-bit floating-point precision (`np.float64`).
  - Coherent arithmetic downmixing ($\frac{L+R}{2}$) preventing phase cancellation and dynamic range clipping.
  - Peak normalization and linked gain scaling preserving relative SNR across clean reference and degraded signals.
  - Strict input validation catching non-finite samples (`NaN`, `Inf`), invalid sampling rates, and empty arrays.
- **Controlled Noise Generation Engine:**
  - **White Gaussian Noise (WGN):** Zero-mean Gaussian noise calibrated to exact user-defined Signal-to-Noise Ratio (SNR in dB) with seed reproducibility.
  - **Mains Electrical Hum:** 50 Hz or 60 Hz fundamental sinusoidal hum with customizable harmonic series ($2f_0, 3f_0, \dots$) and exponential decay factors.
  - **High-Frequency Interference:** Narrowband sinusoids with strict Nyquist boundary enforcement ($f < f_s/2$).
  - **Composite Noise Injection:** Additive multi-source contamination with digital clipping prevention.
- **FIR Filter Designer:**
  - Standard Type-I linear-phase designs: Low-pass, High-pass, Band-pass, and Band-stop.
  - Windowing methods: Hamming, Hann, and Blackman windows with odd tap enforcement.
  - Dual execution modes: Causal filtering (`scipy.signal.lfilter` with constant group delay $\tau = (M-1)/2$) and Zero-Phase forward-backward filtering (`scipy.signal.filtfilt`).
- **IIR Filter Designer & Stability Engine:**
  - Butterworth standard filters (Low-pass, High-pass, Band-pass, Band-stop) with orders from 1 to 10.
  - Second-Order Sections (SOS) representation eliminating coefficient quantization sensitivity and pole clustering.
  - Dedicated high-Q IIR Notch filter for power-line hum rejection (> 35 dB attenuation).
  - Real-time Bounded-Input Bounded-Output (BIBO) pole stability verification ($|z_p| < 1.0$).
  - Dual modes: Causal recursive filtering (`scipy.signal.sosfilt`) and Zero-Phase filtering (`scipy.signal.sosfiltfilt`).
- **Spectral Diagnostics & Time-Frequency Analysis:**
  - Single-sided Fast Fourier Transform (FFT) magnitude spectrum calibrated to peak amplitude ($\text{dBFS}$ or linear) with dominant peak frequency detection.
  - Short-Time Fourier Transform (STFT) spectrogram with physically calibrated units ($\text{dBFS/Hz}$ or $\text{dBFS}$) and dynamic window adaptation for short signals.
- **Objective Quality Metrics:**
  - Signal-to-Noise Ratio (SNR in dB) and $\Delta\text{SNR}$ improvement.
  - Root Mean Square Error (RMSE) and percentage RMSE error reduction.
  - Mean Absolute Error (MAE).
  - High-resolution median execution timing harness with speedup ratio calculation.
- **Dual Analytical Workflows:**
  - **Workflow A (Clean Ground Truth Available):** Ingest clean audio, apply synthetic noise, restore via filters, and calculate objective restoration metrics.
  - **Workflow B (Real-World Degraded Audio):** Ingest real-world noisy audio without ground truth. Objective metrics are strictly suppressed; diagnostic waveforms, spectra, spectrograms, and independent timing are presented.

---

## 3. Application Workspaces

The Streamlit engineering dashboard organizes the end-to-end signal processing pipeline into six distinct interactive workspaces:

| # | Workspace Name | Primary Capabilities & Engineering Responsibilities |
| :---: | :--- | :--- |
| **1** | **Audio Workspace** | Ingest mono/stereo `.wav` files or load built-in audio signals. Toggle between Workflow A (synthetic degradation) and Workflow B (real-world degraded). View waveform plots, inspect sampling parameters, downmix channels, and apply linked peak normalization. |
| **2** | **Noise Generation** | Synthesize calibrated White Gaussian Noise at target SNR (dB), power-line mains hum (50/60 Hz) with harmonic decay, or high-frequency interference tones. Mix composite noise models with clipping protection. |
| **3** | **FIR Filter Designer** | Configure Type-I windowed FIR filters (low-pass, high-pass, band-pass, band-stop) with selectable window functions (Hamming, Hann, Blackman) and odd tap counts. Inspect magnitude response (dB) and linear phase curve ($\tau$ delay). Listen to and download filtered audio. |
| **4** | **IIR Filter Designer** | Design Butterworth filters in Second-Order Sections (SOS) or dedicated 50/60 Hz power-line notch filters. Inspect magnitude and non-linear phase response. Perform automated BIBO pole stability verification ($|z_p| < 1.0$) with pole-zero visualization. |
| **5** | **FIR vs IIR Comparative Evaluation** | Execute head-to-head comparative filtering on identical audio buffers. Enforce topology and cutoff compatibility. Evaluate execution latency, speedup ratio, output SNR improvement, and RMSE reduction across zero-phase and causal modes. |
| **6** | **Signal Analysis & Diagnostics** | Inspect single-sided calibrated FFT spectra with dominant frequency peak readouts. Generate STFT spectrograms with calibrated physical units ($\text{dBFS/Hz}$ PSD or $\text{dBFS}$ spectrum) and dynamic segment scaling. |

---

## UI/UX Release Notes (November 2026)

The Streamlit interface has received a presentation-layer redesign while preserving the DSP processing modules and scientific calculations.

- Added a light-blue/white engineering dashboard theme and native Streamlit theme configuration.
- Refined the application header, workspace titles, session indicators, status cards, and sidebar navigation.
- Organized all six workspaces into bordered control, inspection, execution, and result panels.
- Standardized waveform, FFT, spectrogram, filter magnitude, phase, and FIR/IIR comparison plot styling.
- Added a compact four-stage workflow navigator and a capability ribbon.
- Added restrained motion effects: animated header accent, page entrance, session pulse, brand shimmer, hover feedback, and button interaction feedback.
- Added reduced-motion handling and responsive layout rules for narrow/mobile screens.
- Corrected the external-reference removal flow's malformed statement and repaired indentation in the time-domain and STFT display sections.
- Corrected FIR/IIR metric presentation to use the actual `AudioEvaluationResult.input_mae` and `output_mae` fields instead of the nonexistent `mae` attribute.

These UI changes are presentation and application-flow corrections; they do not alter FIR/IIR filter design, filtering algorithms, FFT/STFT mathematics, or quality-metric formulas. The GitHub Actions workflow verifies compilation and automated tests, but does not replace manual browser/device acceptance testing.

## 4. Technical Architecture

AudioClean DSP enforces strict modularity and separation of concerns between core mathematical signal processing and user-facing presentation:

```
+-------------------------------------------------------------------------+
|                      Streamlit Engineering Dashboard                    |
|                                 (app.py)                                |
+-------------------+-----------------+-------------------+---------------+
                    |                 |                   |
                    v                 v                   v
            +---------------+ +---------------+ +-------------------+
            | Audio I/O &   | | Noise Engine  | | Filter Engines    |
            | Preprocessing | | (Controlled   | | (FIR & IIR)       |
            | (Module A)    | |  Module B)    | | (Modules C & D)   |
            +-------+-------+ +-------+-------+ +---------+---------+
                    |                 |                   |
                    +--------+--------+                   |
                             |                            |
                             v                            v
                    +-----------------+         +-------------------+
                    | Reference &     |         | Cleaned Signal    |
                    | Degraded Signal |         | Output            |
                    +--------+--------+         +---------+---------+
                             |                            |
                             +-------------+--------------+
                                           |
                                           v
                             +----------------------------+
                             | Metrics & FFT Analysis     |
                             | (Modules E, F & G)         |
                             +----------------------------+
```

### Module Responsibilities:
- [`src/audio_io.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/src/audio_io.py): Uncompressed WAV file reading and writing via SoundFile, audio metadata extraction (duration, channels, bit-depth, RMS, crest factor), and safe overwrite validation.
- [`src/preprocessing.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/src/preprocessing.py): Channel downmixing, peak normalization, signal length alignment (truncation/padding), and float64 array validation (rejecting NaN, Inf, empty, non-numeric, and complex inputs).
- [`src/noise_generator.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/src/noise_generator.py): Calibrated White Gaussian Noise (WGN) synthesis, harmonic mains hum generation, high-frequency tone injection with Nyquist enforcement, and composite noise mixing.
- [`src/fir_filters.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/src/fir_filters.py): Type-I windowed sinc FIR filter design (`firwin`), frequency response generation (`freqz`), and causal (`lfilter`) / zero-phase (`filtfilt`) filtering execution.
- [`src/iir_filters.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/src/iir_filters.py): Butterworth and Notch IIR filter design in Second-Order Sections (`butter`, `iirnotch`), BIBO pole stability checking, frequency response evaluation (`sosfreqz`), and causal (`sosfilt`) / zero-phase (`sosfiltfilt`) filtering.
- [`src/fft_analysis.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/src/fft_analysis.py): Single-sided calibrated FFT spectrum calculation (`rfft`) with coherent gain normalization, peak frequency detection, and STFT spectrogram generation with dynamic `nperseg` scaling.
- [`src/visualization.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/src/visualization.py): Diagnostic Matplotlib plotting for waveforms, FFT spectra, STFT spectrograms, filter magnitude responses, phase curves, and time-domain signal overlays.
- [`src/metrics.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/src/metrics.py): Mathematical formulations for SNR, RMSE, MAE, RMSE reduction %, and statistically robust execution time measurement (warmup runs, median, percentiles).
- [`src/comparison.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/src/comparison.py): Head-to-head FIR vs. IIR comparative benchmarking engine with cutoff compatibility checks, speedup ratio calculation, and causal group delay evaluation interval slicing.
- [`app.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/app.py): Full interactive Streamlit engineering application featuring state persistence, workflow provenance tracking, dynamic uploader key lifecycle management, and linked gain normalization.

---

## 5. DSP Methodology & Mathematical Foundations

AudioClean DSP adheres strictly to core digital signal processing principles:

### A. Cutoff Frequency Conventions
- **FIR Window Method:** Cutoff frequency $f_c$ specifies the transition midpoint where the frequency response magnitude $|H(e^{j\omega})| = 0.5$ (half-amplitude, **-6.02 dB** attenuation).
- **Butterworth IIR Design:** Cutoff frequency $f_c$ specifies the half-power passband edge where $|H(e^{j\omega})| = 1/\sqrt{2}$ (**-3.01 dB** attenuation).
- *Engineering Note:* Nominal cutoff equivalence does not imply identical transition roll-offs. AudioClean DSP explicitly documents this convention rather than claiming identical frequency curves.

### B. Causal Filtering vs. Zero-Phase Forward-Backward Filtering
- **Causal FIR Filtering (`lfilter`):** Introduces a strictly linear phase shift corresponding to an exact, constant group delay:
  $$\tau = \frac{M - 1}{2} \text{ samples}$$
  Where $M$ is the number of filter taps.
- **Causal IIR Filtering (`sosfilt`):** Features non-linear phase response, resulting in frequency-dependent phase delay and transient dispersion.
- **Zero-Phase Filtering (`filtfilt` / `sosfiltfilt`):** Employs two-pass bidirectional filtering ($y_1[n] = x[n] * h[n]$, $y[n] = y_1[-n] * h[n]$). This completely eliminates phase distortion ($|H_{\text{total}}(e^{j\omega})| = |H(e^{j\omega})|^2$, $\theta(\omega) = 0$).
- *Important Distinction:* Forward-backward zero-phase filtering requires access to the entire signal buffer; it is an **offline non-causal technique** and is not a real-time zero-latency method.

### C. Causal FIR vs. IIR Evaluation Interval
Evaluating a causal FIR output from sample index $0$ introduces severe boundary bias due to the $M-1$ transient startup samples. AudioClean DSP evaluates causal FIR filtering over the shared valid middle interval:
$$\text{Reference Interval} = x[\tau : N - \tau]$$
$$\text{FIR Output Interval} = y_{\text{FIR}}[2\tau : N]$$
$$\text{IIR Output Interval} = y_{\text{IIR}}[\tau : N - \tau]$$
Where all evaluated arrays have exactly $N - 2\tau$ samples. This ensures sample-exact correspondence against the uncorrupted ground truth without applying artificial non-causal shifts to the IIR filter.

### D. Objective Metrics Formulation
- **Signal-to-Noise Ratio (SNR):**
  $$\text{SNR}_{\text{dB}} = 10 \log_{10} \left( \frac{\sum_{n=0}^{N-1} s^2[n]}{\sum_{n=0}^{N-1} (s[n] - \hat{s}[n])^2} \right)$$
- **Root Mean Square Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} (s[n] - \hat{s}[n])^2}$$
- **RMSE Reduction Percentage:**
  $$\text{Reduction } \% = \left( 1 - \frac{\text{RMSE}_{\text{output}}}{\text{RMSE}_{\text{input}}} \right) \times 100\%$$
  *(If $\text{RMSE}_{\text{input}} = 0$ and $\text{RMSE}_{\text{output}} > 0$, the system reports $-\infty\%$, reflecting an infinite percentage degradation).*

---

## 6. Performance Benchmarks

The computational performance of FIR and IIR digital filters was evaluated under standardized, reproducible benchmark conditions using [`scripts/benchmark_performance.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/scripts/benchmark_performance.py).

### Benchmark Methodology:
- **Test Signal:** Synthetic composite multi-tone harmonic signal (440 Hz fundamental, 880 Hz, 1760 Hz, 3520 Hz overtones with additive 10 dB SNR White Gaussian Noise, $f_s = 44,100\text{ Hz}$).
- **FIR Specification:** 101-tap Type-I Low-pass filter, Hamming window, $f_c = 1500\text{ Hz}$.
- **IIR Specification:** 4th-order (2-section SOS) Butterworth Low-pass filter, $f_c = 1500\text{ Hz}$.
- **Protocol:** 5 warm-up iterations, 30 timed trials, reporting median execution time and standard deviation on 64-bit Windows AMD64.

### Previously Reported Historical Benchmark Results:

| Signal Duration | Sample Count | FIR Zero-Phase (101 taps) | FIR Causal (101 taps) | IIR Zero-Phase (Order 4 SOS) | IIR Causal (Order 4 SOS) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0 s** | 44,100 | 3.67 ± 0.58 ms | 4.13 ± 5.48 ms | 1.13 ± 0.44 ms | 1.04 ± 0.63 ms |
| **5.0 s** | 220,500 | 27.08 ± 7.65 ms | 18.86 ± 11.21 ms | 7.38 ± 3.08 ms | 5.36 ± 0.50 ms |
| **30.0 s** | 1,323,000 | 146.15 ± 19.08 ms | 75.90 ± 36.83 ms | 43.64 ± 4.44 ms | 24.58 ± 5.99 ms |
| **60.0 s** | 2,646,000 | 257.47 ± 41.23 ms | 150.77 ± 29.79 ms | 89.34 ± 41.63 ms | 49.23 ± 6.57 ms |

### Computational Speedup Ratios (FIR Time / IIR Time):

| Signal Duration | Sample Count | Zero-Phase Speedup | Causal Speedup |
| :---: | :---: | :---: | :---: |
| **1.0 s** | 44,100 | **3.25x** | **3.97x** |
| **5.0 s** | 220,500 | **3.67x** | **3.52x** |
| **30.0 s** | 1,323,000 | **3.35x** | **3.09x** |
| **60.0 s** | 2,646,000 | **2.88x** | **3.06x** |

### Engineering Observations & Scientific Clarification:
- **Configuration Specificity:** These benchmark results reflect a specific architectural comparison between a 101-tap FIR filter and a 4th-order (2-section) Butterworth IIR filter. They demonstrate the recursion efficiency of low-order IIR sections compared to high-tap FIR convolution, but **do not represent universal performance guarantees** across all possible filter orders, cutoff frequencies, or hardware platforms.
- **Complexity Scaling:** Execution latency scales in close proportion to signal length $N$ across both filter types.
- **Reporting Boundary:** Filter benchmarks isolate direct time-domain filtering (`scipy.signal.lfilter`/`filtfilt` vs `sosfilt`/`sosfiltfilt`) and are reported separately from spectral transformations (FFT/STFT).

---

## 7. Verification & Automated Test Suite

AudioClean DSP was developed following rigorous test-driven engineering practices.

> **Automated test suite:** **150 tests passed** on each of Python 3.10, 3.12, and 3.13 in GitHub Actions (450 successful test executions across the three-version matrix; 0 failures). The same CI run also completed Python bytecode compilation for `app.py`, `src`, `tests`, and `scripts`.

### Continuous Integration (GitHub Actions)

The repository now runs an automated validation workflow on every push to `main`, pull request targeting `main`, and manual workflow dispatch.

| Matrix environment | Python compilation | Pytest result |
| :--- | :---: | :---: |
| Ubuntu / Python 3.10 | **PASS** | **150 passed** |
| Ubuntu / Python 3.12 | **PASS** | **150 passed** |
| Ubuntu / Python 3.13 | **PASS** | **150 passed** |

- **Workflow:** [AudioClean DSP CI](https://github.com/umeshpandeysh/AudioClean-DSP/actions/workflows/ci.yml)
- **Successful run:** [Run 41579919481](https://github.com/umeshpandeysh/AudioClean-DSP/actions/runs/41579919481)
- **Validated commit:** `2c07e4e2221bbac59305510f0de2a30a600ddb66`
- **Overall result:** 3/3 matrix jobs successful; 450 test executions passed (the same 150-test suite executed once per Python version).

The matrix currently tests Python 3.10, 3.12, and 3.13. Python 3.11 remains listed as an installation-compatible version but is not part of this CI matrix.

### Test Suite Breakdown:

| Test Module | Tests | Verified Subsystems & Behavioral Contracts |
| :--- | :---: | :--- |
| [`tests/test_app.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/tests/test_app.py) | **21** | UI contracts, PCM16 clipping safety, SHA-256 fingerprinting, linked normalization, speedup formatting, workflow provenance, dynamic uploader key lifecycle invalidation, and interactive multi-workspace `AppTest` navigation. |
| [`tests/test_audio_io.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/tests/test_audio_io.py) | **9** | Mono/stereo WAV loading, float64 representation, PCM16 round-trip, clipping protection, non-finite rejection, metadata inspection, and safe overwrite protection. |
| [`tests/test_comparison.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/tests/test_comparison.py) | **9** | Head-to-head comparison in zero-phase and causal modes, topology compatibility enforcement, cutoff alignment checks, and shared valid reference interval $[\tau : N - \tau]$ ($N - 2\tau$ samples). |
| [`tests/test_fft_analysis.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/tests/test_fft_analysis.py) | **15** | Single-sided calibrated FFT spectrum, coherent gain handling, odd and even sample length support, DC bin preservation, STFT spectrogram units (`dBFS/Hz`), and truncation protection. |
| [`tests/test_fir_filters.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/tests/test_fir_filters.py) | **21** | Type-I linear-phase designs (lowpass, highpass, bandpass, bandstop), Hamming/Hann/Blackman windowing, odd tap enforcement, frequency response calculations, and causal group delay ($\tau = (M-1)/2$). |
| [`tests/test_iir_filters.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/tests/test_iir_filters.py) | **23** | Butterworth SOS designs (orders 1–10), dedicated 50/60 Hz notch filter with configurable $Q$, real-time BIBO pole stability check ($|z_p| < 1.0$), and SciPy SOS padding matching. |
| [`tests/test_integration.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/tests/test_integration.py) | **2** | Complete end-to-end DSP pipeline restoration, quantitative SNR improvement, and RMSE error reduction on compound noise. |
| [`tests/test_metrics.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/tests/test_metrics.py) | **21** | Reference-based SNR, RMSE, MAE, analytical vector verification, RMSE reduction %, `evaluate_audio_quality` dataclass verification, and execution timing harness. |
| [`tests/test_noise_generator.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/tests/test_noise_generator.py) | **7** | Calibrated White Gaussian Noise (WGN), seed reproducibility, 50/60 Hz mains hum with harmonic decay, high-frequency tone Nyquist enforcement, and composite noise mixing. |
| [`tests/test_preprocessing.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/tests/test_preprocessing.py) | **14** | Stereo-to-mono downmixing, peak normalization, duration calculations, input validation (NaN/Inf, empty, non-numeric, complex rejection), and length alignment. |
| [`tests/test_visualization.py`](file:///C:/Users/UMESH%20PANDEY/Downloads/ceenew/AudioClean-DSP/tests/test_visualization.py) | **8** | Waveform plotting (mono/stereo), calibrated FFT spectrum visualization, STFT spectrogram plotting, filter magnitude/phase response overlays, and time-domain signal comparison. |

---

## 8. Technology Stack

- **Core Programming Language:** Python 3.13 (compatible with Python 3.10, 3.11, 3.12, 3.13)
- **Numerical Processing:** NumPy 2.5.4
- **Digital Signal Processing:** SciPy 1.18.1 (`scipy.signal`, `scipy.fft`)
- **Audio I/O & Serialization:** SoundFile 0.14.0 (libsndfile backend)
- **Data Visualization:** Matplotlib 3.11.2
- **Interactive Application Framework:** Streamlit 1.66.1
- **Testing & Verification Harness:** pytest 9.2.0

---

## 9. Installation & Local Execution

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed on your machine.
- Git installed.

### Quickstart Setup:

```powershell
# 1. Clone the repository
git clone https://github.com/umeshpandeysh/AudioClean-DSP.git
cd AudioClean-DSP

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the Streamlit application
streamlit run app.py
```

Open your web browser to `http://localhost:8501` to access the application dashboard.

---

## 10. Repository File Structure

```
AudioClean-DSP/
├── app.py                          # Streamlit Interactive Engineering Dashboard
├── requirements.txt                # Production and development dependencies
├── pytest.ini                      # Pytest runner configuration
├── README.md                       # Comprehensive engineering documentation
│
├── src/                            # Core Digital Signal Processing Engine
│   ├── __init__.py                 # Package interface exports
│   ├── audio_io.py                 # WAV audio loading, saving, and metadata extraction
│   ├── preprocessing.py            # Normalization, downmixing, alignment, and validation
│   ├── noise_generator.py          # Calibrated WGN, mains hum, tone, and composite noise
│   ├── fir_filters.py              # Type-I linear-phase windowed FIR filters
│   ├── iir_filters.py              # Butterworth SOS and Notch IIR filters with stability check
│   ├── fft_analysis.py             # Calibrated single-sided FFT and STFT spectrograms
│   ├── visualization.py            # Diagnostic Matplotlib figures and plotting utilities
│   ├── metrics.py                  # SNR, RMSE, MAE, and execution timing harness
│   └── comparison.py               # Rigorous FIR vs. IIR head-to-head comparison engine
│
├── tests/                          # Automated Pytest Regression Suite (150 tests)
│   ├── __init__.py
│   ├── conftest.py                 # Test fixtures and synthesized audio signals
│   ├── test_app.py                 # App contracts, state, lifecycle & AppTest navigation
│   ├── test_audio_io.py            # Audio I/O loading, saving, and info tests
│   ├── test_comparison.py          # Comparison engine and causal interval tests
│   ├── test_fft_analysis.py        # FFT calibration and spectrogram tests
│   ├── test_fir_filters.py         # FIR design, symmetry, and filtering tests
│   ├── test_iir_filters.py         # IIR design, SOS stability, and notch tests
│   ├── test_integration.py         # Full end-to-end DSP pipeline integration tests
│   ├── test_metrics.py             # Objective SNR, RMSE, MAE mathematical accuracy tests
│   ├── test_noise_generator.py     # Noise synthesis and SNR calibration tests
│   ├── test_preprocessing.py       # Preprocessing, normalization, and validation tests
│   └── test_visualization.py       # Diagnostic plotting and figure export tests
│
├── docs/                           # Formal Engineering Documentation
│   ├── architecture.md             # System architecture and data flow specification
│   ├── dsp_methodology.md          # Mathematical foundations and DSP formulations
│   ├── phase4_remediation_report.md# Phase 4 UI audit remediation report
│   ├── phase5_validation_report.md # Phase 5 testing and performance validation report
│   ├── test_coverage_matrix.md     # Complete test traceability matrix
│   └── final_delivery_report.md    # Final project closure and delivery report
│
├── scripts/                        # Standalone Verification & Reproduction Scripts
│   ├── benchmark_performance.py    # Standardized 30-trial benchmark reproduction script
│   ├── verify_audit_remediation.py # 6-experiment mathematical verification pipeline
│   ├── verify_phase2.py            # Phase 2 core engine verification script
│   ├── verify_phase4_5.py          # Phase 4-5 UI and pipeline verification script
│   └── verify_phase6.py            # Phase 6 delivery verification script
│
└── results/                        # Experimental Output Artifacts
    ├── audio/                      # Audio demonstration samples
    └── plots/                      # Rendered diagnostic frequency and phase curves
```

---

## 11. Scientific Integrity & Documented Limitations

1. **Audio Codec Scope:** Exclusively supports uncompressed linear PCM WAV files (mono and stereo, 16-bit or 32-bit floating point). Compressed lossy formats (MP3, AAC, OGG) are intentionally excluded to eliminate non-linear perceptual quantization artifacts from DSP analysis.
2. **Filter Topologies:** Classic Type-I windowed FIR (Hamming, Hann, Blackman) and Butterworth / Notch IIR filters. Higher-order Elliptic, Chebyshev, and adaptive filtering topologies (LMS, RLS) are outside the undergraduate scope of this project.
3. **Phase Linearity vs. Computational Latency:** Causal FIR filters provide strictly linear phase response with constant group delay at the expense of higher tap counts and convolution latency. Causal IIR filters provide lower latency and sharper roll-off per pole at the expense of non-linear phase distortion.
4. **Offline Processing Constraint:** Bidirectional forward-backward zero-phase filtering (`filtfilt` / `sosfiltfilt`) requires complete signal buffers and is restricted to offline post-processing.
5. **No Blind Ground-Truth Inference:** Objective restoration metrics ($\Delta\text{SNR}$, % RMSE reduction) require a pre-aligned, uncorrupted reference signal. In Workflow B (single-file degraded audio), objective metrics are suppressed to uphold mathematical integrity.
6. **Benchmark Hardware Sensitivity:** Numerical execution times depend on underlying CPU architecture, operating system scheduling, memory bandwidth, and Python interpreter version. Reported benchmark figures reflect standardized measurements under the specified test environment.

---

## 12. Future Research & Development

The following potential extensions represent natural avenues for future academic investigation:
- **Adaptive Filtering Engines:** Implementation of Least Mean Squares (LMS) and Normalized LMS (NLMS) adaptive filters for time-varying acoustic noise cancellation.
- **Parametric Equalization (Biquads):** Peaking, low-shelf, and high-shelf Second-Order Sections with interactive Q and gain controls.
- **Multirate Digital Signal Processing:** Decimation and interpolation polyphase filter banks for multirate audio sample rate conversion.
- **Spectrogram Masking & Denoising:** Spectral subtraction and Wiener filtering implementations operating directly in the STFT frequency domain.

---

## 13. Academic Context & Project Metadata

- **Project Title:** AudioClean DSP — Digital Audio Noise Removal and Comparative Filter Performance Analysis
- **Author:** Umesh Pandey (Undergraduate Engineering Project)
- **Discipline:** Digital Signal Processing (DSP) / Electronics & Communication Engineering
- **Laboratory:** Digital Signal Processing Laboratory
- **Roadmap Architecture:** Six-Phase Engineering Lifecycle (Architecture, Core Engine, Analysis, Application UI, Validation, and Final Delivery).
