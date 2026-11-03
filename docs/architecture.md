# AudioClean DSP — System Architecture & Data Flow

## 1. Architectural Overview

AudioClean DSP is structured according to a strict separation of concerns between core mathematical/DSP processing and user-facing presentation layers. The architecture is modular, deterministic, and test-driven, ensuring all signal transformations can be executed and validated in headless script environments (such as automated Pytest runs) as well as within the interactive Streamlit dashboard.

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

---

## 2. Core Modules & Responsibilities

### Module A: Audio Input and Preprocessing (`src/audio_io.py`, `src/preprocessing.py`)
- **Ingestion:** Reads WAV audio streams using `soundfile`, extracting exact integer sample rates ($f_s$), channel count, and bit-depth.
- **Channel Conversion:** Converts multi-channel (stereo) signals to mono for spectral analysis ($\frac{L + R}{2}$) while retaining raw channel buffers for filtered audio export.
- **Amplitude Normalization:** Scales audio floats to peak unity $[-1.0, 1.0]$ or specified target RMS without introducing hard clipping.
- **Integrity Preservation:** Prevents implicit or uncontrolled sample-rate modification; maintains original $f_s$ throughout all processing stages.

### Module B: Controlled Noise Generator (`src/noise_generator.py`)
- **White Gaussian Noise (WGN):** Generates zero-mean normal distribution noise calibrated to a precise user-defined Signal-to-Noise Ratio (SNR in dB):
  $$\sigma_{\text{noise}}^2 = \frac{P_{\text{signal}}}{10^{\text{SNR}_{\text{dB}}/10}}$$
- **Powerline Hum (50 Hz):** Synthesizes deterministic sinusoidal interference $A \sin(2\pi \cdot 50 \cdot t)$ with optional harmonic series ($100\text{ Hz}, 150\text{ Hz}$).
- **High-Frequency Tone / Hiss:** Injects narrowband or colored high-frequency components to evaluate low-pass filter rolloff and attenuation.
- **Reference Management:** Retains the uncorrupted input as an explicit ground-truth reference array for quantitative evaluation.

### Module C: FIR Digital Filter Engine (`src/fir_filters.py`)
- **Filter Topologies:** Low-pass, High-pass, Band-pass, and Band-stop.
- **Design Methodology:** Windowed sinc design using `scipy.signal.firwin`. Supported windows: Hamming, Hann, Blackman, Rectangular, Kaiser.
- **Characteristics:** Strictly linear phase (symmetric coefficients, Type I/II), guaranteed unconditional stability (all poles at $z = 0$), finite duration impulse response.
- **Implementation:** Direct-form discrete convolution / tapped delay line via `scipy.signal.lfilter`.

### Module D: IIR Digital Filter Engine (`src/iir_filters.py`)
- **Filter Topologies:** Butterworth Low-pass, High-pass, Band-pass, Band-stop, and IIR Notch (50 Hz / 60 Hz mains hum rejection).
- **Design Methodology:** Analog prototype bilinear transformation (`scipy.signal.butter`, `scipy.signal.iirnotch`).
- **Order Semantics:** Lowpass/highpass filters maintain $N$ poles (`order = N`), while bandpass/bandstop double the pole count to $2N$ (`order = 2N`, `prototype_order = N`).
- **Numerical Stability:** Implementation via Second-Order Sections (SOS) representations (`scipy.signal.sosfilt`, `scipy.signal.sosfiltfilt`). BIBO stability is programmatically guaranteed through `check_sos_stability` ensuring all section poles satisfy $|p_k| < 1.0$.
- **Filtering Modes:** Zero-phase noncausal bidirectional filtering (`sosfiltfilt`) for offline phase-distortion-free processing, and causal recursive filtering (`sosfilt`) for real-time simulation.

### Module E: Spectral & Frequency-Domain Analysis (`src/fft_analysis.py`)
- **Discrete Fourier Analysis:** Calibrated single-sided Fast Fourier Transform (FFT) using `scipy.fft.rfft` with Hann, Hamming, Blackman, or rectangular windowing.
- **Magnitude Scaling:** Peak-amplitude calibrated using coherent window gain $S_1 = \sum w[n]$, ensuring a sinusoid of amplitude $A$ maps to a peak of exactly $A$ ($20 \log_{10}(A)\text{ dBFS}$). DC ($k=0$) and Nyquist ($k=N/2$) bins are preserved without doubling.
- **Multichannel & Truncation Safety:** Explicit channel indexing for stereo files and `allow_truncation=False` validation against accidental data loss when $n_{\text{fft}} < N$.
- **Time-Frequency Representation:** Short-Time Fourier Transform (STFT) power spectrogram computation via `scipy.signal.spectrogram`, reporting Power Spectral Density in $\text{dBFS/Hz}$ ($10 \log_{10}$ scaling).
- **Transfer Function Response:** Complex frequency response evaluation ($H(e^{j\omega})$) using `scipy.signal.freqz` (for FIR) and `scipy.signal.sosfreqz` (for IIR SOS), supporting both integer point resolution and arbitrary frequency evaluation vectors.

### Module F: Objective Performance Evaluation (`src/metrics.py`)
- **Signal-to-Noise Ratio (SNR):**
  $$\text{SNR} = 10 \log_{10} \left( \frac{\sum_{n=0}^{N-1} s^2[n]}{\sum_{n=0}^{N-1} (s[n] - \hat{s}[n])^2} \right)$$
- **SNR Improvement ($\Delta\text{SNR}$):**
  $$\Delta\text{SNR} = \text{SNR}_{\text{filtered}} - \text{SNR}_{\text{noisy}}$$
- **Root Mean Square Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} (s[n] - \hat{s}[n])^2}$$
- **Computational Benchmark:** Latency benchmarking (wall-clock time and algorithmic throughput in samples/second).
- **Alignment Compensation:** Group delay compensation between filtered output $\hat{s}[n]$ and ground truth $s[n]$ to prevent artificial RMSE penalty caused purely by linear phase lag ($\tau_g = \frac{M-1}{2}$).

### Module G: Comparative Evaluation Engine (`src/comparison.py`)
- Orchestrates multi-filter matrix evaluations across identical test signals.
- Generates comparative data structures comparing FIR vs. IIR across order, cutoff frequency, window type, SNR improvement, RMSE, and processing duration.

---

## 3. Data Flow Diagram

```
[Raw Audio File (.wav)]
          │
          ▼
[Module A: audio_io.load_audio] ───► returns: (audio_data, sample_rate)
          │
          ├───────────────────────────────────────────┐
          │ (Ground Truth: s[n])                      │
          ▼                                           ▼
[Module B: noise_generator]                 [Uncorrupted Reference]
          │ (adds noise e.g., WGN, 50Hz)              │
          ▼                                           │
   (Degraded: x[n])                                   │
          │                                           │
          ├──────────────────────────┐                │
          ▼                          ▼                │
[Module C: FIR Engine]      [Module D: IIR Engine]    │
   h_fir * x[n]                SOS filtering          │
          │                          │                │
          ▼                          ▼                │
   (Output: y_fir[n])         (Output: y_iir[n])      │
          │                          │                │
          ├──────────────────────────┴────────────────┤
          │                                           │
          ▼                                           ▼
[Module E: Spectral Analysis]             [Module F: Metrics]
 - FFT Magnitude Spectrum                  - SNR Before / After
 - STFT Spectrogram                        - SNR Gain
 - Filter Frequency Response H(f)          - RMSE & Processing Time
          │                                           │
          └─────────────────────┬─────────────────────┘
                                │
                                ▼
                   [Module H: UI / Streamlit]
                   - Waveform & Spectrum Plots
                   - Comparative Metric Tables
                   - Audio Player & WAV Exporter
```

---

## 4. Key Engineering Constraints & Academic Standards

1. **Deterministic Filter Design:** All filter specifications require explicit validation of the Nyquist bound ($f_{\text{cutoff}} < f_s / 2$).
2. **Numerical Precision:** 64-bit floating point arrays (`float64`) are used internally for filter convolutions and spectral calculations; converted to normalized 16-bit PCM or 32-bit float upon WAV export.
3. **No Fabricated Benchmarks:** Performance metric calculations are evaluated directly against actual signal arrays. When a real recording lacks an uncorrupted reference, synthetic SNR calculations are omitted and spectral noise floor attenuation is reported instead.
