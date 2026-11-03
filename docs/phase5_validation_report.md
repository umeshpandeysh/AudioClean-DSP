# AudioClean DSP — Phase 5: Comprehensive Testing and Validation Report

## 1. Executive Summary & Validation Objective
AudioClean DSP is an undergraduate digital signal processing engineering application designed for audio noise removal and rigorous comparative performance analysis between Finite Impulse Response (FIR) and Infinite Impulse Response (IIR) digital filters.

The objective of Phase 5 is to execute a complete, independent engineering validation of the entire system:
1. Core DSP mathematical correctness and numerical stability.
2. End-to-end user workflows in the Streamlit application (`app.py`).
3. Regression test suite execution across all modules.
4. Stress, performance, and latency scaling from short signals up to 60-second recordings under controlled experimental conditions.
5. Strict enforcement of scientific integrity (no fabricated ground-truth metrics, no artificial alignment, and mathematically grounded filter comparison).

---

## 2. Test Environment & System Configuration
- **Operating System:** Windows 11 Enterprise (64-bit AMD64)
- **Python Version:** 3.13.15 (MSC v.1944 64-bit)
- **Virtual Environment:** Dedicated local `.venv`
- **Installed Core Dependencies:**
  - `numpy`: 2.5.4
  - `scipy`: 1.18.1
  - `soundfile`: 0.14.0
  - `matplotlib`: 3.11.2
  - `pytest`: 9.2.0
  - `streamlit`: 1.66.1

---

## 3. Automated Test Suite Execution Results
The test suite was executed using:
```powershell
.\.venv\Scripts\python.exe -m pytest -v tests
```

### Summary of Results:
- **Total Tests Collected:** 150
- **Tests Passed:** 150 (100%)
- **Tests Failed:** 0
- **Tests Skipped:** 0
- **Exit Code:** 0 (Clean Pass)

### Exact Breakdown by Test Module:
1. `tests/test_app.py`: 21 passed
   - In-memory WAV byte conversion (`audio_to_bytes`), PCM16 clipping safety, SHA-256 file fingerprinting, strict external reference upload validation (`validate_reference_match`), linked gain normalization preserving relative SNR, safe `format_speedup` formatting, Workflow A/B provenance tracking, reference lifecycle detachment/invalidation on primary change/removal/same-shape replacements, full widget uploader dynamic key lifecycle reset helpers, and interactive `streamlit.testing.v1.AppTest` session state persistence across app reruns, workflow toggles, and workspace navigation.
2. `tests/test_audio_io.py`: 9 passed
   - Mono/stereo WAV loading, float64 representation, PCM16 round-trip, clipping protection, non-finite rejection, metadata inspection, safe overwrite protection.
3. `tests/test_comparison.py`: 9 passed
   - Head-to-head `compare_fir_and_iir` execution in zero-phase and causal modes, topology compatibility enforcement, cutoff alignment verification, shared valid reference interval $[\tau : N - \tau]$ ($N - 2\tau$ samples) for causal mode, speedup ratio formatting, and multi-channel handling.
4. `tests/test_fft_analysis.py`: 15 passed
   - Single-sided calibrated FFT magnitude spectrum, coherent gain handling, odd and even sample length handling, DC bin preservation, STFT spectrogram calculation (`dBFS/Hz` and `dBFS` units), channel selection, and truncation protection.
5. `tests/test_fir_filters.py`: 21 passed
   - Type-I linear-phase designs (lowpass, highpass, bandpass, bandstop), Hamming/Hann/Blackman windowing, odd tap enforcement, frequency response calculations, causal constant group delay ($\tau = (M-1)/2$), zero-phase forward-backward filtering. Note: Includes 4 parametrized topology cases and 3 parametrized window cases.
6. `tests/test_iir_filters.py`: 23 passed
   - Butterworth SOS designs (orders 1-10), dedicated 50/60 Hz IIR notch filter with configurable $Q$, real-time BIBO pole stability check ($|z_p| < 1$), causal `sosfilt` vs zero-phase `sosfiltfilt`, SciPy SOS padding matching. Note: Includes 4 parametrized topology cases.
7. `tests/test_integration.py`: 2 passed
   - Complete end-to-end DSP pipeline restoration, quantitative SNR improvement, and RMSE error reduction on compound noise.
8. `tests/test_metrics.py`: 21 passed
   - Ground-truth SNR calculation, RMSE and MAE hand-calculated vectors, RMSE reduction %, `evaluate_audio_quality` dataclass verification, zero-energy reference rejection, shape mismatch rejection, `measure_processing_time` timing harness with type validation and boolean rejection.
9. `tests/test_noise_generator.py`: 7 passed
   - Calibrated White Gaussian Noise (WGN), deterministic seed reproducibility, 50/60 Hz mains hum with harmonic decay, high-frequency tone Nyquist enforcement, composite noise combination with clipping protection.
10. `tests/test_preprocessing.py`: 14 passed
    - Stereo-to-mono downmixing, peak normalization, duration calculations, input validation (NaN/Inf, empty, non-numeric, complex rejection), length alignment (truncation and zero-padding).
11. `tests/test_visualization.py`: 8 passed
    - Waveform plotting (mono/stereo), calibrated FFT spectrum visualization, STFT spectrogram plotting, filter magnitude/phase response overlays, time-domain signal comparison, figure saving, and overwrite safety.

---

## 4. DSP Mathematical Validation

| Module | Verification Target | Tested Behavior | Result |
| :--- | :--- | :--- | :--- |
| **Audio I/O** | `src.audio_io` | Mono/Stereo WAV ingestion, PCM16 conversion, safe overwrite, float64 fidelity | **VERIFIED** |
| **Preprocessing** | `src.preprocessing` | Safe downmixing, peak normalization to target 1.0, length alignment, invalid input rejection | **VERIFIED** |
| **Noise Generator** | `src.noise_generator` | WGN SNR calibrated within statistical tolerance, 50/60 Hz hum harmonics, tone Nyquist bounds | **VERIFIED** |
| **FIR Filters** | `src.fir_filters` | Type-I linear phase, constant delay $\tau = (M-1)/2$, -6.02 dB midpoint convention, odd tap constraint | **VERIFIED** |
| **IIR Filters** | `src.iir_filters` | Butterworth SOS representations, -3.01 dB cutoff convention, notch rejection > 35 dB, pole $|z_p| < 1$ | **VERIFIED** |
| **FFT Analysis** | `src.fft_analysis` | Calibrated single-sided magnitude, DC bin not doubled, STFT spectral density units (`dBFS/Hz`) | **VERIFIED** |
| **Metrics** | `src.metrics` | Reference-based SNR, RMSE, MAE, non-finite input protection, accurate median timing harness | **VERIFIED** |
| **Comparison** | `src.comparison` | Causal shared interval $[\tau : N - \tau]$ ($N - 2\tau$ samples), cutoff equality check, speedup ratio | **VERIFIED** |

---

## 5. Streamlit Application Workflow Validation

The application (`app.py`) was verified using interactive `streamlit.testing.v1.AppTest`, programmatic verification scripts, and headless launch across four core operational workflows:

### Workflow A: Clean Ground-Truth Audio (Reference Available)
- Clean audio ingested and verified.
- Noise synthesis (WGN, mains hum, HF tone, composite) applied successfully.
- FIR and IIR filters designed and executed in both zero-phase and causal modes.
- Head-to-head comparison executed using `compare_fir_and_iir()`:
  - Input vs. Output SNR accurately calculated.
  - $\Delta\text{SNR}$ improvement and RMSE reduction % computed against aligned ground truth.
- Filtered audio rendered to in-memory player and downloaded as compliant 16-bit PCM WAV.

### Workflow B: Real-World Degraded Audio (No Ground Truth)
- Degraded audio ingested without reference.
- Ground-truth objective error metrics (SNR, RMSE, MAE, restoration %) strictly **suppressed**.
- No-reference comparative analysis performed:
  - Waveform time-domain plots.
  - Calibrated single-sided FFT spectrum.
  - STFT Spectrogram with calibrated units (`dBFS/Hz` or `dBFS`).
  - Independent filter execution timing via `measure_processing_time()` with speedup ratio (displays N/A on invalid latency).
  - Filter magnitude and phase response overlays.

### Workflow C: Separate External Clean Reference Upload
- Uploaded external clean reference validated against noisy audio via `validate_reference_match()`.
- Successfully rejected mismatched sample rates (e.g., 44.1 kHz vs 48 kHz).
- Successfully rejected mismatched sample lengths/channels.
- Validated clean references attached to enable Workflow A objective metric evaluation.
- UI explicitly explains that matching sample rate and shape does not guarantee synchronization; the user must supply pre-aligned references.

### Workflow D: Robust Error Handling
- Invalid filter cutoffs exceeding Nyquist boundary rejected with clear `ValueError`.
- Even filter tap counts for Type-I FIR rejected with helpful error guidance.
- Signals shorter than minimum zero-phase padding rejected safely.
- Incompatible filter topologies (e.g., lowpass vs highpass) rejected by comparison engine.
- STFT segment length dynamically adapts to input signal length, preventing crashes on short signals.

---

## 6. Performance & Resource Benchmarking

Performance was benchmarked under strictly controlled reproducible conditions using a synthetic composite test signal (440 Hz fundamental + 1200 Hz harmonic with 10 dB target WGN, $f_s = 44,100\text{ Hz}$). Benchmarks were conducted over **30 repetitions** with **5 warm-up iterations**, reporting the exact median execution latency and standard deviation from `scripts/benchmark_performance.py`:

| Signal Duration | Sample Count | FIR Zero-Phase (101 taps) | FIR Causal (101 taps) | IIR Zero-Phase (Order 4 SOS) | IIR Causal (Order 4 SOS) | IIR Speedup (Zero-Phase) | IIR Speedup (Causal) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0 s** | 44,100 | 3.67 ± 0.58 ms | 4.13 ± 5.48 ms | 1.13 ± 0.44 ms | 1.04 ± 0.63 ms | **3.25x** | **3.97x** |
| **5.0 s** | 220,500 | 27.08 ± 7.65 ms | 18.86 ± 11.21 ms | 7.38 ± 3.08 ms | 5.36 ± 0.50 ms | **3.67x** | **3.52x** |
| **30.0 s** | 1,323,000 | 146.15 ± 19.08 ms | 75.90 ± 36.83 ms | 43.64 ± 4.44 ms | 24.58 ± 5.99 ms | **3.35x** | **3.09x** |
| **60.0 s** | 2,646,000 | 257.47 ± 41.23 ms | 150.77 ± 29.79 ms | 89.34 ± 41.63 ms | 49.23 ± 6.57 ms | **2.88x** | **3.06x** |

### Engineering Observations & Complexity Analysis:
- **Configuration Specificity:** These results evaluate a specific architectural pairing: a 101-tap Type-I windowed FIR filter versus a 4th-order (2-section) Butterworth IIR filter. They demonstrate the computational efficiency of low-order IIR recursion compared to high-tap FIR convolution, but do not imply that every IIR filter is universally faster than every FIR filter under all design specifications. Filter benchmarks isolate direct convolution (`scipy.signal.lfilter`/`filtfilt`) versus SOS recursion (`sosfilt`/`sosfiltfilt`).
- **Observed Scaling:** Across the four tested sample sizes ($N = 44.1\text{k}$ to $2.65\text{M}$ samples), execution time scales in close proportion to signal length $N$.
- **Real-Time Factor:** For all tested durations, processing latency is orders of magnitude lower than audio duration (e.g., 60 seconds of audio is filtered in ~151 ms causally by FIR and ~49 ms by IIR).

---

## 7. Remediation Summary
- **Defect 1 (Upload Rerun State):** Upload state now preserved across reruns via SHA-256 content hashing. Working audio and noise components no longer reset on UI interaction.
- **Defect 2 (Linked Gain Normalization):** Added `apply_linked_normalization()`, scaling clean reference and noisy audio by the exact identical scalar gain, preserving relative SNR and RMSE reduction %.
- **Defect 3 (Technical Descriptions):** Clarified offline non-causal nature of `filtfilt`, exact linear-phase group delay of causal FIR, and distinct character of narrowband IIR notch filters.
- **Defect 4 (Short Audio in STFT):** Segment length (`nperseg`) dynamically constrained to signal length, preventing crashes on short audio.
- **Defect 5 (Workflow Provenance):** Ensured degraded audio uploaded in Workflow B is never treated as clean ground truth when switching to Workflow A.
- **Defect 6 (External Reference Lifecycle & Widget Invalidation):** References strictly bound to primary recording hash (`ref_primary_hash`), detached on primary change/removal, and cleared upon invalid upload or uploader removal. Uploader widget state invalidation enforced via dynamic keys (`ref_uploader_v{version}`) and state popping upon primary replacement, removal, and workflow switching, preventing silent reattachment of previously uploaded references.
- **Defect 7 (Safe Speedup Formatting):** Handled None/non-finite/non-positive speedup ratio safely without TypeError.

---

## 8. Scientific Integrity & Documented Limitations
1. **Cutoff Conventions:** Windowed FIR specifies the $-6.02\text{ dB}$ (half-amplitude $|H|=0.5$) transition midpoint, whereas Butterworth IIR specifies $-3.01\text{ dB}$ (half-power $|H|=1/\sqrt{2}$).
2. **Causal Evaluation Interval:** In causal FIR mode, evaluation is strictly constrained to the valid middle interval $[\tau : N - \tau]$ ($N - 2\tau$ samples) to guarantee identical sample correspondence between FIR and IIR without invalid startup boundary bias.
3. **No Fabricated Metrics:** When real-world noisy audio is processed without a clean reference (Workflow B), objective restoration metrics are omitted entirely.

---

## 9. Phase 5 Exit Gate Verdict
**STATUS: PASSED (COMPLETE)**
- All 150 tests passing cleanly at 100%.
- Streamlit application launches and operates without exceptions.
- Mathematical validation, reproducible 30-trial performance benchmarking (`scripts/benchmark_performance.py`), and defect remediation verified.
