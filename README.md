# AudioClean DSP

**Digital Audio Noise Removal and Comparative Performance Analysis Using FIR and IIR Digital Filters**

## 1. Project Overview & Objectives

**AudioClean DSP** is an academic digital signal processing (DSP) application designed to demonstrate fundamental DSP concepts through interactive experimentation and quantitative evaluation. The application allows users to ingest audio recordings, inspect time- and frequency-domain characteristics, synthesize controlled acoustic and electrical interference, design and apply Finite Impulse Response (FIR) and Infinite Impulse Response (IIR) digital filters, and systematically evaluate filtering performance using objective metrics.

---

## 2. Technology Stack

- **Language:** Python 3.13+
- **Numerical Processing:** NumPy
- **Digital Signal Processing:** SciPy (`scipy.signal`, `scipy.fft`)
- **Audio I/O:** SoundFile
- **Visualizations:** Matplotlib
- **Interactive GUI:** Streamlit
- **Testing & Verification:** Pytest

---

## 3. Implemented Modules (Phase 1 & Phase 2)

### Module 1: Audio Input and Output (`src/audio_io.py`)
- `load_audio(file_path)`: Reads mono and stereo WAV audio into 64-bit floating point NumPy arrays with exact sample rate preservation.
- `save_audio(file_path, data, sample_rate, subtype, allow_clipping, overwrite)`: Exports WAV audio safely, preventing accidental overwriting of user recordings and detecting amplitude clipping.
- `get_audio_info(data, sample_rate)`: Computes duration, channel count, sample count, peak amplitude, RMS amplitude, and crest factor.

### Module 2: Audio Preprocessing (`src/preprocessing.py`)
- `convert_to_mono(data)`: Downmixes multi-channel stereo signals using arithmetic channel averaging ($0.5 \times (L + R)$) without mutating the input array.
- `normalize_audio(data, target_peak)`: Scales peak absolute amplitude to a target boundary while safeguarding against division by zero on silent signals.
- `validate_audio(data, sample_rate)`: Enforces dimension, non-emptiness, finiteness (no NaNs/Infs), and sample rate constraints.
- `calculate_duration(num_samples, sample_rate)`: Computes exact temporal duration in seconds ($T = N / f_s$).
- `align_signal_lengths(sig1, sig2, mode)`: Aligns comparative signals via truncation or zero-padding.

### Module 3: Controlled Noise Generation (`src/noise_generator.py`)
- `add_white_gaussian_noise(clean_signal, target_snr_db, seed)`: Generates zero-mean Gaussian thermal noise calibrated to an exact target SNR (within $\le 0.01$ dB accuracy).
- `add_sinusoidal_hum(signal, sample_rate, freq, amplitude, harmonics, harmonic_decay)`: Synthesizes deterministic 50 Hz power-line mains hum with optional harmonic overtones (100 Hz, 150 Hz, etc.).
- `add_high_frequency_noise(signal, sample_rate, freq, amplitude)`: Synthesizes high-frequency interference while strictly enforcing the Nyquist criterion ($f < f_s / 2$).
- `add_noise_combination(signal, noise_components, prevent_clipping)`: Combines multiple additive noise models with automatic clipping prevention.

---

## 4. Usage Examples

```python
import numpy as np
from src.audio_io import load_audio, save_audio, get_audio_info
from src.preprocessing import convert_to_mono, normalize_audio
from src.noise_generator import add_white_gaussian_noise, add_sinusoidal_hum

# 1. Load audio
audio_data, sr = load_audio("data/samples/sample.wav")

# 2. Inspect audio properties
info = get_audio_info(audio_data, sr)
print(f"Duration: {info['duration_seconds']:.2f} s | Channels: {info['num_channels']} | RMS: {info['rms_amplitude']:.4f}")

# 3. Downmix stereo to mono and normalize
mono_audio = convert_to_mono(audio_data)
normalized_audio = normalize_audio(mono_audio, target_peak=0.95)

### Module 4: FIR Digital Filter Engine (`src/fir_filters.py`)
- `design_fir_filter(sample_rate, filter_type, cutoff_hz, numtaps, window)`: Designs Type-I symmetric linear-phase FIR filters (lowpass, highpass, bandpass, bandstop) using windowed sinc designs with odd tap counts.
- `apply_fir_filter(audio, filter_design, mode)`: Applies causal filtering (`mode="causal"` with exact group delay) or offline zero-phase filtering (`mode="zero_phase"` with zero delay).
- `get_fir_frequency_response(filter_design, worN)`: Computes physical frequency response ($0$ to $f_s/2$ Hz), linear/dB magnitude, and continuous unwrapped phase.

### Module 5: IIR Digital Filter Engine (`src/iir_filters.py`)
- `design_iir_filter(sample_rate, filter_type, cutoff_hz, order)`: Designs Butterworth lowpass, highpass, bandpass, and bandstop filters in numerically stable Second-Order Sections (SOS).
- `design_iir_notch(sample_rate, notch_freq, quality_factor)`: Designs high-Q narrow-band notch rejection filters (e.g. 50 Hz mains hum).
- `apply_iir_filter(audio, filter_design, mode)`: Applies causal filtering (`mode="causal"` via `sosfilt`) or offline zero-phase filtering (`mode="zero_phase"` via `sosfiltfilt`).
- `get_iir_frequency_response(filter_design, worN)`: Computes complex frequency response using `sosfreqz` over exact physical frequencies in Hz.

### Module 6: FFT & Spectral Analysis (`src/fft_analysis.py`)
- `compute_fft(audio, sample_rate, window, n_fft)`: Computes calibrated single-sided Fast Fourier Transform magnitude spectra with coherent gain normalization and DC preservation.
- `compute_spectrogram(audio, sample_rate, nperseg, noverlap, window, scaling)`: Computes STFT time-frequency spectrogram matrices in power dB.

### Module 7: Signal Visualization (`src/visualization.py`)
- Reusable Matplotlib figures: `plot_waveform`, `plot_fft_spectrum`, `plot_spectrogram`, `plot_filter_response`, `plot_phase_response`, `plot_signal_comparison`.

---

## 4. Usage Examples

```python
import numpy as np
from src.audio_io import load_audio, save_audio
from src.noise_generator import add_sinusoidal_hum
from src.iir_filters import design_iir_notch, apply_iir_filter
from src.fft_analysis import compute_fft, compute_spectrogram
from src.visualization import plot_fft_spectrum, plot_signal_comparison

# 1. Load audio
audio_data, sr = load_audio("data/samples/fidelity_test.wav")

# 2. Add 50 Hz powerline hum with harmonics
noisy_audio, hum = add_sinusoidal_hum(audio_data, sample_rate=sr, freq=50.0, amplitude=0.2, harmonics=[2, 3])

# 3. Design an IIR Notch filter to reject 50 Hz interference (Q=30)
notch_filter = design_iir_notch(sample_rate=sr, notch_freq=50.0, quality_factor=30.0)

# 4. Filter audio using zero-phase forward-backward filtering
cleaned_audio = apply_iir_filter(noisy_audio, notch_filter, mode="zero_phase")

# 5. Compute FFT spectrum before and after filtering
freqs, mag_before, _ = compute_fft(noisy_audio, sample_rate=sr)
_, mag_after, _ = compute_fft(cleaned_audio, sample_rate=sr)

# 6. Generate and save verification plot
fig = plot_signal_comparison(
    audio_data, noisy_audio, cleaned_audio, sample_rate=sr,
    save_path="results/plots/notch_cleaning_comparison.png", overwrite=True
)
```

---

## 5. Automated Testing & Verification

Run the comprehensive test suite (all 98 tests across 8 test modules):
```powershell
.\.venv\Scripts\python.exe -m pytest -v tests
```

Run focused IIR unit tests:
```powershell
.\.venv\Scripts\python.exe -m pytest -v tests/test_iir_filters.py
```

Run focused FFT, visualization, and integration tests:
```powershell
.\.venv\Scripts\python.exe -m pytest -v tests/test_fft_analysis.py tests/test_visualization.py tests/test_integration.py
```

Run Consolidated Audit & Remediation reproducible numerical experiments (Experiments 1–6):
```powershell
.\.venv\Scripts\python.exe scripts/verify_audit_remediation.py
```

Run Phase 4 & 5 verification experiments:
```powershell
.\.venv\Scripts\python.exe scripts/verify_phase4_5.py
```

---

## 6. Development Roadmap

- [x] **Phase 1:** Project architecture, directory structure, environment setup, and dependency verification.
- [x] **Phase 2:** Audio input/output, preprocessing, and calibrated noise generation engine.
- [x] **Phase 3:** FIR filter implementation (window methods, frequency responses, impulse responses).
- [x] **Phase 4:** IIR filter implementation (Butterworth, notch filter, second-order sections, stability checks).
- [x] **Phase 5:** Frequency-domain analysis (FFT magnitude spectra, STFT spectrograms, frequency response curves, and visualizations).
- [x] **Phase 6:** Performance evaluation metrics (SNR, RMSE, MAE, SNR improvement, processing latency).
- [ ] **Phase 7:** Comparative evaluation engine (FIR vs. IIR trade-off matrices).
- [ ] **Phase 8:** Interactive Streamlit engineering dashboard.
- [ ] **Phase 9:** Academic documentation, mathematical derivations, experiment logs, and final report.
