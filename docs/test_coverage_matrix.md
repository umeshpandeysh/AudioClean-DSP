# AudioClean DSP — Comprehensive Test Coverage Traceability Matrix

## Matrix Overview
This matrix provides complete traceability between system components, test types, test scenarios, expected engineering behaviors, actual test outputs, and validation status across all 139 tests in the test suite.

| Component | Test File | Test Type | Test Scenario | Expected Behavior | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Audio I/O** | `test_audio_io.py` | Unit | Load mono/stereo WAV | Preserves sample rate, channels, float64 representation | Identical shapes & rates | **PASS** |
| **Audio I/O** | `test_audio_io.py` | Unit | Save WAV round-trip | Writes PCM_16 WAV without clipping or sample drift | Accurate round-trip | **PASS** |
| **Audio I/O** | `test_audio_io.py` | Unit | Overwrite protection | Rejects existing file if `overwrite=False` | Raises `FileExistsError` | **PASS** |
| **Audio I/O** | `test_audio_io.py` | Unit | Metadata extraction | Accurately computes duration, RMS, peak, crest factor | Matches theoretical values | **PASS** |
| **Preprocessing** | `test_preprocessing.py`| Unit | Downmix stereo to mono | Averages channels without phase cancellation | Matches $(L+R)/2$ | **PASS** |
| **Preprocessing** | `test_preprocessing.py`| Unit | Peak normalization | Scales signal peak to exact target (e.g. 1.0) | Absolute peak equals target | **PASS** |
| **Preprocessing** | `test_preprocessing.py`| Unit | Signal validation | Rejects NaN, Inf, non-numeric, and complex inputs | Raises `ValueError`/`TypeError`| **PASS** |
| **Preprocessing** | `test_preprocessing.py`| Unit | Length alignment | Correctly truncates or zero-pads arrays | Exact target length reached | **PASS** |
| **Noise Generator** | `test_noise_generator.py` | Unit | Calibrated WGN | Synthesizes noise at specified target SNR | Achieved SNR within tolerance | **PASS** |
| **Noise Generator** | `test_noise_generator.py` | Unit | Random seed reproducibility | Produces bit-exact noise with fixed seed | Identical arrays | **PASS** |
| **Noise Generator** | `test_noise_generator.py` | Unit | Mains hum synthesis | 50/60 Hz hum with selectable harmonic decay | Spectral peaks at multiples | **PASS** |
| **Noise Generator** | `test_noise_generator.py` | Unit | HF tone & Nyquist check | Rejects frequencies at or above Nyquist ($f \ge f_s/2$) | Raises `ValueError` | **PASS** |
| **Noise Generator** | `test_noise_generator.py` | Unit | Composite noise addition | Combines multiple noises with clipping prevention | No values exceed $[-1, 1]$ | **PASS** |
| **FIR Filters** | `test_fir_filters.py` | Unit | Type-I filter design | Designs lowpass, highpass, bandpass, bandstop | Correct tap count & symmetry | **PASS** |
| **FIR Filters** | `test_fir_filters.py` | Unit | Window selection | Supports Hamming, Hann, Blackman windows | Proper side-lobe suppression | **PASS** |
| **FIR Filters** | `test_fir_filters.py` | Unit | Causal group delay | Causal mode exhibits constant delay $\tau = (M-1)/2$ | Peak aligned with delay | **PASS** |
| **FIR Filters** | `test_fir_filters.py` | Unit | Zero-phase filtering | Forward-backward filtering eliminates phase shift | Zero phase lag observed | **PASS** |
| **FIR Filters** | `test_fir_filters.py` | Unit | Cutoff convention | Cutoff defined at -6.02 dB half-amplitude point | Verified at transition point | **PASS** |
| **FIR Filters** | `test_fir_filters.py` | Unit | Even tap rejection | Type-I design strictly rejects even tap counts | Raises `ValueError` | **PASS** |
| **IIR Filters** | `test_iir_filters.py` | Unit | Butterworth SOS design | Designs stable SOS matrices for all 4 topologies | All poles inside unit circle | **PASS** |
| **IIR Filters** | `test_iir_filters.py` | Unit | 50/60 Hz notch filter | Sharp rejection (> 35 dB) at fundamental hum | Deep null at notch frequency | **PASS** |
| **IIR Filters** | `test_iir_filters.py` | Unit | Pole stability check | Verifies $|z_p| < 1.0$ for all second-order sections | Flags stability accurately | **PASS** |
| **IIR Filters** | `test_iir_filters.py` | Unit | SciPy SOSfiltfilt padding | Matches SciPy default padding length calculation | No premature signal rejection| **PASS** |
| **IIR Filters** | `test_iir_filters.py` | Unit | Causal vs Zero-phase | Causal introduces non-linear phase, zero-phase eliminates | Phase curvature confirmed | **PASS** |
| **FFT Analysis** | `test_fft_analysis.py` | Unit | Calibrated magnitude | Single-sided spectrum calibrated to peak amplitude | Pure tone amplitude exact | **PASS** |
| **FFT Analysis** | `test_fft_analysis.py` | Unit | DC bin preservation | DC bin ($0\text{ Hz}$) is not doubled | DC level mathematically exact | **PASS** |
| **FFT Analysis** | `test_fft_analysis.py` | Unit | Odd/even sample lengths | Accurately generates $(N/2)+1$ or $(N+1)/2$ bins | Correct bin counts | **PASS** |
| **FFT Analysis** | `test_fft_analysis.py` | Unit | STFT Spectrogram | Calibrated units in $\text{dBFS/Hz}$ (density) and $\text{dBFS}$| Verified frequency/time axes | **PASS** |
| **Metrics** | `test_metrics.py` | Unit | SNR calculation | Mathematically exact ratio against known ground truth | Verified against analytical | **PASS** |
| **Metrics** | `test_metrics.py` | Unit | RMSE & MAE calculation | Hand-calculated vector verification | Matches analytical formulas | **PASS** |
| **Metrics** | `test_metrics.py` | Unit | Zero-energy rejection | Rejects silent reference signal for SNR | Raises `ValueError` | **PASS** |
| **Metrics** | `test_metrics.py` | Unit | Timing harness | Accurately measures median and percentiles | Returns timing dictionary | **PASS** |
| **Comparison** | `test_comparison.py` | Integration | Matched filter comparison | Runs FIR and IIR comparison on identical signal | Produces comparative report | **PASS** |
| **Comparison** | `test_comparison.py` | Integration | Causal evaluation interval | Compares shared valid interval $[\tau : N - \tau]$ | Exactly $N - 2\tau$ samples | **PASS** |
| **Comparison** | `test_comparison.py` | Integration | Cutoff equality check | Rejects comparison if cutoffs mismatch beyond tolerance | Raises `ValueError` | **PASS** |
| **Comparison** | `test_comparison.py` | Integration | Speedup calculation | Computes FIR execution time / IIR execution time | Returns accurate speedup | **PASS** |
| **Integration** | `test_integration.py` | Integration | End-to-end DSP pipeline | Ingests, degrades, restores, and evaluates audio | Positive SNR gain confirmed | **PASS** |
| **Application UI** | `test_app.py` | App Contract | In-memory WAV bytes | Converts floating-point array to 16-bit PCM WAV bytes | Decodable by soundfile | **PASS** |
| **Application UI** | `test_app.py` | App Contract | PCM16 clipping safety | Clamps overshoot beyond $[-1.0, 1.0]$ before write | No integer wraparound | **PASS** |
| **Application UI** | `test_app.py` | App Contract | Content hashing | Generates stable SHA-256 fingerprint for uploads | Bit-exact consistency | **PASS** |
| **Application UI** | `test_app.py` | App Contract | Reference validation | Strictly validates sample rate, channels, and length | Blocks invalid reference | **PASS** |
| **Application UI** | `test_app.py` | App Contract | Linked Normalization | Scales reference and noisy audio by identical gain factor | Preserves relative SNR | **PASS** |
| **Application UI** | `test_app.py` | AppTest Interaction| Session state persistence | Preserves processed audio across simulated script reruns | Audio buffer not reset | **PASS** |
| **Application UI** | Headless smoke run | System / Manual | Streamlit app launch | Launches cleanly without import/syntax errors | Zero startup errors | **PASS** |
| **Application UI** | Workflow A | System / Manual | Clean ground truth flow | Noise synthesis, filtering, and objective metrics | Verified end-to-end | **PASS** |
| **Application UI** | Workflow B | System / Manual | No-reference flow | Degraded audio filtered without fabricated metrics | Suppresses false SNR/RMSE | **PASS** |
