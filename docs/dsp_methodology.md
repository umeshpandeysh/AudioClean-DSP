# Digital Signal Processing (DSP) Methodology & Mathematical Foundations

## 1. Discrete-Time Audio Signals & Sampling

A continuous acoustic sound pressure wave $s_a(t)$ is digitized through uniform periodic sampling according to the **Nyquist-Shannon Sampling Theorem**:
$$s[n] = s_a(n T_s) = s_a\left(\frac{n}{f_s}\right), \quad n \in \mathbb{Z}$$

Where:
- $T_s$ is the sampling interval (seconds).
- $f_s = \frac{1}{T_s}$ is the sampling frequency (Hertz).
- $n$ is the discrete time sample index.

To prevent irreversible spectral aliasing, the continuous signal must be strictly band-limited to below the **Nyquist frequency**:
$$f_{\text{max}} < f_{\text{Nyquist}} = \frac{f_s}{2}$$

If frequency components exceeding $f_s / 2$ are sampled, they alias (mirror) symmetrically into the baseband spectrum:
$$f_{\text{alias}} = |f - k \cdot f_s|$$

---

## 2. Multi-Channel Representation & Downmixing

Audio signals in AudioClean DSP are handled as single-channel (mono) or multi-channel (stereo) discrete time sequences:
- **Mono:** $s[n] \in \mathbb{R}^N$, a 1D vector of length $N$.
- **Stereo:** $\mathbf{S}[n] = [s_L[n], s_R[n]] \in \mathbb{R}^{N \times 2}$.

### Arithmetic Downmixing (Stereo to Mono)
To perform single-channel spectral and filtering analysis without introducing bias toward one spatial channel, stereo signals are converted to mono using arithmetic channel averaging:
$$s_{\text{mono}}[n] = \frac{1}{2} \left(s_L[n] + s_R[n]\right)$$

This linear combination preserves coherent in-phase acoustic content, eliminates phase-cancellation-induced amplitude inflation, and maintains unity dynamic range.

---

## 3. Numerical Representation: Floating-Point Audio

In fixed-point integer audio formats (e.g., 16-bit PCM), signals are discretized into integer levels:
$$x_{\text{int16}} \in [-32768, 32767]$$

Integer representation causes non-negligible quantization noise and catastrophic overflow during intermediate filter convolutions. AudioClean DSP enforces **64-bit double-precision floating point (`np.float64`)** across all intermediate DSP algorithms:
$$s[n] \in [-1.0, 1.0] \subset \mathbb{R}$$

- **Dynamic Range:** Exceeds 1500 dB (eliminating intermediate digital clipping).
- **Quantization Noise Floor:** Around $-300\text{ dB}$, ensuring mathematical filter behaviors reflect true continuous transfer functions.
- **Conversion on Export:** Normalized floating-point values are linearly mapped to integer PCM formats only at final file serialization (`soundfile.write`).

---

## 4. Signal Amplitude Metrics: Peak and RMS

### Peak Amplitude
Peak amplitude measures the maximum instantaneous excursion of the audio waveform:
$$A_{\text{peak}} = \max_{0 \le n < N} |s[n]|$$

Peak amplitude determines the headroom before digital clipping ($A_{\text{peak}} \le 1.0$).

### Root Mean Square (RMS) Amplitude & Power
RMS amplitude characterizes the effective average acoustic power and perceived loudness of the signal:
$$A_{\text{RMS}} = \sqrt{P_{\text{signal}}} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} s^2[n]}$$

Where $P_{\text{signal}}$ is the average signal power.

### Crest Factor
The Crest Factor ($C$) quantifies waveform peakiness and dynamic range:
$$C_{\text{dB}} = 20 \log_{10}\left(\frac{A_{\text{peak}}}{A_{\text{RMS}}}\right)$$

---

## 5. Controlled Noise Models

### White Gaussian Noise (WGN)
White Gaussian Noise represents wideband thermal and acoustic noise with a constant power spectral density across all frequencies from $0$ to $f_s / 2$. Its instantaneous amplitudes follow a normal (Gaussian) probability distribution:
$$w[n] \sim \mathcal{N}(0, \sigma_w^2)$$
$$\mathbb{E}[w[n]] = 0, \quad \mathbb{E}[w[n]^2] = \sigma_w^2 = P_{\text{noise}}$$

Because samples are statistically uncorrelated:
$$\mathbb{E}[w[n] w[n-k]] = \sigma_w^2 \delta[k]$$

### Sinusoidal Electrical Interference (Powerline Hum)
Powerline hum originates from stray electromagnetic induction from the AC power grid (50 Hz standard in Europe/Asia, 60 Hz in North America). It is modeled as a deterministic fundamental sinusoid with optional harmonic overtones:
$$h[n] = A_0 \sin\left(2\pi \frac{f_0}{f_s} n\right) + \sum_{k=2}^{K} A_0 \cdot \gamma^{k-1} \sin\left(2\pi \frac{k f_0}{f_s} n\right)$$

Where:
- $f_0 = 50\text{ Hz}$ (fundamental interference).
- $k \in \{2, 3, \dots\}$ represents harmonic multiples (100 Hz, 150 Hz, etc.).
- $\gamma \in (0, 1)$ is the harmonic decay rate (typically $\gamma = 0.5$).

---

## 6. Signal-to-Noise Ratio (SNR) Formulation & Numerical Example

### Definition
Signal-to-Noise Ratio (SNR) expresses the ratio of clean signal power to additive noise power:
$$\text{SNR} = \frac{P_{\text{signal}}}{P_{\text{noise}}}$$

In decibels (dB):
$$\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right)$$

### Derivation for Calibrated Noise Generation
Given a clean reference signal $s[n]$ and a desired target $\text{SNR}_{\text{target}}$:
1. Compute the actual clean signal power:
   $$P_{\text{signal}} = \frac{1}{N} \sum_{n=0}^{N-1} s^2[n]$$
2. Invert the decibel equation to solve for required noise power:
   $$P_{\text{noise}} = \frac{P_{\text{signal}}}{10^{\text{SNR}_{\text{target}} / 10}}$$
3. Determine the required noise standard deviation:
   $$\sigma_{\text{noise}} = \sqrt{P_{\text{noise}}}$$
4. Sample independent Gaussian random variables $w[n] \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)$.

### Worked Numerical Example
Consider a 1-second pure tone sampled at $f_s = 48000\text{ Hz}$ with amplitude $A = 0.5$:
$$s[n] = 0.5 \sin\left(2\pi \frac{440}{48000} n\right)$$

1. **Clean Signal Power:**
   $$P_{\text{signal}} = \frac{A^2}{2} = \frac{0.5^2}{2} = \frac{0.25}{2} = 0.125000$$
2. **Target SNR:**
   Let $\text{SNR}_{\text{target}} = 10.0\text{ dB}$.
3. **Required Noise Power:**
   $$P_{\text{noise}} = \frac{P_{\text{signal}}}{10^{10.0 / 10}} = \frac{0.125000}{10^1} = 0.012500$$
4. **Noise Standard Deviation:**
   $$\sigma_{\text{noise}} = \sqrt{0.012500} \approx 0.111803$$
5. **Validation:**
   $$\text{SNR}_{\text{actual}} = 10 \log_{10}\left(\frac{0.125000}{0.012500}\right) = 10 \log_{10}(10) = 10.000\text{ dB}$$

---

## 7. Finite Impulse Response (FIR) Digital Filters

### 1. Conceptual Definition of an FIR Filter
A Finite Impulse Response (FIR) digital filter is a linear time-invariant (LTI) discrete system whose unit impulse response $h[n]$ settles to exactly zero after a finite number of sampling intervals $M$. Unlike recursive analog or IIR systems, an FIR filter has no continuous feedback loops: each output sample depends exclusively on current and prior input samples.

### 2. Fundamental Differences: FIR vs. IIR Filters
| Feature | FIR Filter | IIR Filter |
| :--- | :--- | :--- |
| **Impulse Response Length** | Strictly finite ($M$ samples) | Infinitely long (exponential decay) |
| **System Poles** | All poles restricted to the origin $z = 0$ | Poles located throughout the $z$-plane inside $|z| < 1$ |
| **Stability** | **Unconditionally stable** (no runaway feedback) | Conditionally stable (poles must remain inside unit circle) |
| **Phase Response** | **Exactly linear phase** (zero phase distortion with symmetric taps) | Non-linear phase (dispersive group delay) |
| **Computational Complexity**| Higher order required for steep roll-off ($M \approx 50 - 300$) | Much lower order required ($N \approx 4 - 8$) |
| **Feedback Structure** | Non-recursive feedforward tapped delay line | Recursive (contains feedback from past outputs $y[n-k]$) |

### 3. Difference Equation and Transfer Function
The discrete-time convolution difference equation governing an $M$-tap FIR filter is:
$$y[n] = \sum_{k=0}^{M-1} b[k] x[n-k] = b[0]x[n] + b[1]x[n-1] + \dots + b[M-1]x[n-(M-1)]$$

In the complex $z$-domain, the transfer function $H(z)$ is an all-zero polynomial:
$$H(z) = \sum_{k=0}^{M-1} b[k] z^{-k} = \frac{\sum_{k=0}^{M-1} b[k] z^{M-1-k}}{z^{M-1}}$$

Because the denominator is strictly $z^{M-1}$, all $M-1$ system poles reside at the origin $z = 0$, guaranteeing absolute Bounded-Input Bounded-Output (BIBO) stability under all conditions.

### 4. Meaning of Filter Coefficients
The coefficients $b[k]$ represent the discrete values of the filter's unit impulse response $h[k]$:
$$h[k] = b[k] \quad \text{for } 0 \le k \le M-1$$
When an ideal unit impulse $\delta[n] = [1, 0, 0, \dots]$ is presented to the filter input, the resulting output sequence is identical to the coefficient vector $[b[0], b[1], \dots, b[M-1]]$.

### 5. Filter Order vs. Number of Taps
- **Number of Taps ($M = \text{numtaps}$):** The total count of coefficients (feedforward multipliers) in the impulse response.
- **Filter Order ($N$):** The number of unit delay elements ($z^{-1}$) in the tapped delay line:
  $$N = M - 1$$
For example, an FIR filter with $M = 101$ taps has a filter order of $N = 100$.

### 6. The Window Method of FIR Filter Design
An ideal frequency-selective filter (e.g., ideal lowpass with brick-wall cutoff $\omega_c$) possesses an infinitely long, noncausal sinc impulse response:
$$h_d[n] = \frac{\sin(\omega_c n)}{\pi n} = \frac{\omega_c}{\pi} \operatorname{sinc}\left(\frac{\omega_c n}{\pi}\right), \quad -\infty < n < \infty$$

Direct truncation of $h_d[n]$ to a finite window of length $M$ causes severe ripples near the band edges due to **Gibbs phenomenon** (overshoot of $\approx 8.9\%$). To suppress spectral leakage, the ideal impulse response is multiplied point-by-point by a tapered continuous window function $w[n]$ centered at $\tau = \frac{M-1}{2}$:
$$h[n] = h_d\left(n - \frac{M-1}{2}\right) \cdot w[n], \quad 0 \le n \le M-1$$

### 7. Window Function Characteristics
AudioClean DSP supports three classic tapered windows:

1. **Hamming Window:**
   $$w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{M-1}\right)$$
   - *Peak Sidelobe Attenuation:* $\approx -43\text{ dB}$
   - *Mainlobe Width:* $\approx \frac{8\pi}{M}$
   - *Use Case:* General-purpose audio filtering with a balanced trade-off between roll-off steepness and stopband rejection.

2. **Hann Window:**
   $$w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{M-1}\right) = \sin^2\left(\frac{\pi n}{M-1}\right)$$
   - *Peak Sidelobe Attenuation:* $\approx -31.5\text{ dB}$
   - *Sidelobe Fall-off Rate:* $-18\text{ dB/octave}$ (rapid asymptotic roll-off)
   - *Use Case:* Smooth spectral transitions with minimal high-frequency leakage.

3. **Blackman Window:**
   $$w[n] = 0.42 - 0.5 \cos\left(\frac{2\pi n}{M-1}\right) + 0.08 \cos\left(\frac{4\pi n}{M-1}\right)$$
   - *Peak Sidelobe Attenuation:* $\approx -58\text{ dB}$ (exceptionally quiet stopband)
   - *Mainlobe Width:* $\approx \frac{12\pi}{M}$ (wider transition band)
   - *Use Case:* High-attenuation noise removal (e.g., mains hum and out-of-band hiss rejection).

### 8. Cutoff Frequency and the Nyquist Bound
The cutoff frequency $f_c$ designates the half-power ($-3\text{ dB}$ / $70.7\%$ amplitude) boundary separating the passband from the transition band:
$$|H(e^{j 2\pi f_c / f_s})| \approx \frac{1}{\sqrt{2}} \approx 0.7071$$
By Shannon's theorem, all filter cutoffs must be strictly positive and strictly below the Nyquist frequency:
$$0 < f_c < f_{\text{Nyquist}} = \frac{f_s}{2}$$

### 9. Linear Phase and Symmetric Coefficients (Type-I FIR)
An FIR filter exhibits **generalized linear phase** if its phase response satisfies $\theta(\omega) = -\alpha \omega$. Linear phase ensures that all frequency components experience the exact same temporal delay, preventing waveform dispersion (phase distortion) which would smear transient speech consonants and musical attacks.

Linear phase requires the coefficients to be strictly symmetric about their midpoint:
$$b[n] = b[M - 1 - n], \quad 0 \le n \le M-1$$

### 9. FIR Cutoff Frequency Convention vs. IIR Butterworth
An essential theoretical and practical distinction exists between the cutoff conventions of windowed FIR filters and IIR Butterworth filters:
- **FIR Window Method (`firwin`):** Cutoff frequency specifies the **-6.02 dB (half-amplitude, $|H(f)| = 0.5$)** point of the unwindowed ideal sinc transition. Windowing preserves this transition midpoint symmetry.
- **IIR Butterworth Filters:** Cutoff frequency specifies the **-3.01 dB (half-power, $|H(f)| = 1/\sqrt{2} \approx 0.7071$)** point where signal power is halved:
  $$|H(j\Omega_c)|^2 = \frac{1}{1 + 1} = \frac{1}{2} \implies 10 \log_{10}(0.5) = -3.0103\text{ dB}$$
When comparing FIR and IIR filters side-by-side, this 3 dB discrepancy at $f_c$ is mathematically expected and reflects the underlying synthesis criteria.

AudioClean DSP enforces **odd $M$ (odd tap count)** to generate **Type-I linear-phase FIR filters**:
- Integer delay $\tau = \frac{M-1}{2}$ samples.
- Does not force zero at $\omega = \pi$ ($z = -1$) or $\omega = 0$ ($z = 1$), allowing arbitrary lowpass, highpass, bandpass, and bandstop designs. (By contrast, even-tap Type-II filters mathematically force $H(e^{j\pi}) = 0$, making highpass and bandstop designs impossible).

### 10. Group Delay
Group delay $\tau_g(\omega)$ is the negative derivative of phase with respect to angular frequency:
$$\tau_g(\omega) = -\frac{d\theta(\omega)}{d\omega}$$
For a Type-I linear phase FIR filter, group delay is an absolute constant across all frequencies:
$$\tau_g = \frac{M - 1}{2} \text{ samples} = \frac{M - 1}{2 f_s} \text{ seconds}$$

### 11. Causal vs. Zero-Phase Filtering
1. **Causal Filtering (`mode="causal"`, `scipy.signal.lfilter`):**
   - Real-time computable: computes $y[n]$ strictly from current and past samples $x[n], x[n-1], \dots$.
   - Output signal is delayed by exactly $\tau_g = \frac{M-1}{2}$ samples.
2. **Zero-Phase Filtering (`mode="zero_phase"`, `scipy.signal.filtfilt`):**
   - Noncausal offline algorithm: passes the signal forward through the filter, reverses the sequence in time, and filters backwards.
   - Resulting transfer function: $|H(e^{j\omega})|^2$.
   - **Phase response:** Identically zero radians across all frequencies ($\theta(\omega) \equiv 0$).
   - Output signal is perfectly aligned with the input with **zero sample delay**.

### 12. Engineering Advantages and Limitations of FIR Filters
- **Advantages:**
  1. Guaranteed unconditional numerical stability (no poles outside $z=0$).
  2. Perfect linear phase (constant group delay; no phase distortion).
  3. Simple implementation using hardware/DSP Multiply-Accumulate (MAC) units.
- **Limitations:**
  1. Higher computational complexity and latency than IIR filters for equivalent transition sharpness.
  2. Inability to remove noise that overlaps in frequency with desired speech harmonics without simultaneously attenuating the speech content.

---

## 8. Infinite Impulse Response (IIR) Digital Filters

### 1. Conceptual Definition of an IIR Filter
An Infinite Impulse Response (IIR) digital filter is a recursive discrete-time system whose unit impulse response $h[n]$ theoretically persists indefinitely, decaying asymptotically toward zero. This infinite memory is enabled by internal **feedback**: the current output $y[n]$ is computed not only from current and past inputs $x[n-k]$, but also from weighted previous outputs $y[n-k]$.

### 2. Difference Equation and Transfer Function
The governing autoregressive moving-average (ARMA) difference equation is:
$$y[n] = \frac{1}{a_0} \left( \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k] \right)$$

Applying the $z$-transform yields the rational transfer function:
$$H(z) = \frac{Y(z)}{X(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{\sum_{k=0}^{N} a_k z^{-k}} = \frac{B(z)}{A(z)}$$

Where:
- Roots of numerator polynomial $B(z)$ represent system **zeros**.
- Roots of denominator polynomial $A(z)$ represent system **poles**.

### 3. Butterworth Filter Characteristics and Order Semantics
AudioClean DSP employs classical analog-prototype Butterworth filters mapped into the digital domain via the **Bilinear Transform**:
$$s = \frac{2}{T_s} \frac{1 - z^{-1}}{1 + z^{-1}}$$

Key properties of the Butterworth response:
- **Maximally Flat Passband:** The magnitude response has zero derivatives at $\omega = 0$ ($d^k |H(\omega)| / d\omega^k = 0$ for $k = 1, \dots, 2N-1$). It exhibits no passband ripple (unlike Chebyshev or Elliptic filters).
- **Monotonic Fall-off:** The magnitude response $|H(j\Omega)|^2 = \frac{1}{1 + (\Omega/\Omega_c)^{2N}}$ decreases monotonically with frequency.
- **Roll-off Rate:** The asymptotic stopband attenuation steepness is $20N\text{ dB/decade}$ ($6N\text{ dB/octave}$), where $N$ is the prototype filter order.
- **Order Semantics (Prototype vs. Resulting Order):**
  - **Lowpass / Highpass:** A prototype filter of order $N$ yields a digital filter of order $N$ ($K = \lceil N/2 \rceil$ second-order sections).
  - **Bandpass / Bandstop:** The frequency transformation $s \to \frac{s^2 + \omega_0^2}{B s}$ doubles the number of poles. A prototype filter of order $N$ produces a digital filter of resulting order $2N$ ($N$ second-order sections). Both `prototype_order` ($N$) and `resulting_order` ($2N$) are explicitly exposed in `IIRFilterDesign`.

### 4. Second-Order Sections (SOS) Representation
In high-order direct-form IIR filters ($N \ge 4$), the coefficients of high-degree polynomials $A(z) = \prod_{k=1}^{N} (1 - p_k z^{-1})$ become extremely sensitive to finite numerical precision. Minute floating-point roundoff errors can shift poles outside the unit circle ($|p_k| \ge 1$), causing catastrophic instability and limit-cycle oscillations.

To ensure numerical stability, AudioClean DSP factors the transfer function into a cascade of bi-quadratic **Second-Order Sections (SOS)**:
$$H(z) = g \prod_{k=1}^{K} \frac{b_{0k} + b_{1k} z^{-1} + b_{2k} z^{-2}}{a_{0k} + a_{1k} z^{-1} + a_{2k} z^{-2}}$$

Each biquad section pairs complex-conjugate poles with adjacent zeros, drastically reducing coefficient quantization sensitivity and internal dynamic range expansion.

### 5. BIBO Stability Criteria & Automated Verification
A discrete-time LTI IIR filter is strictly Bounded-Input Bounded-Output (BIBO) stable if and only if all system poles $p_k$ lie strictly inside the open unit circle of the complex $z$-plane:
$$|p_k| < 1.0 \quad \forall k \in \{1, \dots, N_{\text{poles}}\}$$

In AudioClean DSP, every SOS filter design is verified programmatically via `check_sos_stability(sos)`. For each section $k$, the quadratic roots of the denominator $a_{0k} z^2 + a_{1k} z + a_{2k} = 0$ are computed, and the maximum pole radius is constrained:
$$\max_{k} |p_k| < 1.0 - \epsilon_{\text{tol}}, \quad \epsilon_{\text{tol}} = 10^{-7}$$
Filters with poles on or outside the unit circle are rejected immediately.

### 6. Digital Notch Filters and Quality Factor (Q)
A digital notch (band-reject) filter eliminates a single discrete frequency $f_0$ (such as 50 Hz power-line mains hum) while leaving surrounding frequencies intact.
Its transfer function places a pair of complex-conjugate zeros directly on the unit circle at $\omega_0 = 2\pi f_0 / f_s$, and a pair of poles inside the unit circle at radius $r = 1 - \frac{\pi \Delta f}{f_s}$:
$$H(z) = b_0 \frac{1 - 2\cos(\omega_0) z^{-1} + z^{-2}}{1 - 2 r \cos(\omega_0) z^{-1} + r^2 z^{-2}}$$

The **Quality Factor ($Q$)** governs the sharpness and $-3\text{ dB}$ rejection bandwidth $\Delta f$:
$$Q = \frac{f_0}{\Delta f} \implies \Delta f = \frac{f_0}{Q}$$
- A high $Q$ ($Q \approx 30$) produces an exceptionally narrow notch ($\Delta f \approx 1.67\text{ Hz}$ at $50\text{ Hz}$), preventing attenuation of nearby speech or musical harmonics.
- *Trade-off:* High $Q$ poles are positioned very close to the unit circle ($r \to 1$), resulting in longer transient settling times ($\tau \approx \frac{1}{\pi \Delta f}$).

### 7. Causal vs. Zero-Phase IIR Filtering
- **Causal Mode (`sosfilt`):** Real-time recursive processing. Introduces nonlinear phase delay and dispersive group delay $\tau_g(\omega)$, which varies with frequency.
- **Zero-Phase Mode (`sosfiltfilt`):** Forward-backward filtering. Completely cancels phase distortion ($\theta(\omega) \equiv 0$) and squares the magnitude response ($|H(e^{j\omega})|^2$), doubling stopband attenuation.

---

## 9. Frequency-Domain Analysis: Discrete Fourier Transform & Spectrograms

### 1. Discrete Fourier Transform (DFT)
The discrete Fourier representation of a finite discrete audio sequence $x[n]$ ($0 \le n < N$) is:
$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2\pi k n / N}, \quad k = 0, 1, \dots, N-1$$

AudioClean DSP utilizes the Fast Fourier Transform (FFT) algorithm via `scipy.fft.rfft` for real-valued signals, computing $N_{\text{rfft}} = \lfloor N/2 \rfloor + 1$ discrete bins corresponding to physical frequencies:
$$f_k = \frac{k \cdot f_s}{N}, \quad 0 \le k \le \lfloor N/2 \rfloor$$

### 2. Frequency Resolution vs. Zero-Padding
The physical frequency resolution (Rayleigh resolution) of an audio recording is fundamentally bounded by its temporal duration $T = N / f_s$:
$$\Delta f_{\text{physical}} = \frac{f_s}{N} = \frac{1}{T}$$
- **Zero-Padding ($n_{\text{fft}} > N$):** Appending zeros to a signal before computing the FFT evaluates the continuous Discrete-Time Fourier Transform (DTFT) at finer frequency intervals $\Delta f_{\text{grid}} = f_s / n_{\text{fft}}$.
- *Crucial Physical Fact:* Zero-padding acts as sinc-interpolation in the frequency domain. It smooths spectral curves and improves visual peak interpolation, but **does not resolve closely spaced frequencies** that were merged in the original observation window $T$.

### 3. Windowing and Spectral Leakage
Computing the FFT of a finite block implicitly multiplies the infinite signal by a rectangular window. The sharp temporal discontinuities at the block edges convolve the true spectrum with a sinc function, causing **spectral leakage** (energy from discrete tones spills into distant sidelobes).
Tapered windows ($w[n]$) smoothly attenuate edge samples to zero:
- **Hann Window:** Replaces rectangular sidelobes ($-13\text{ dB}$) with attenuated sidelobes ($-31.5\text{ dB}$, falling at $-18\text{ dB/octave}$).
- **Coherent Gain Scaling:** To prevent windowing from artificially reducing measured tone amplitudes, the spectrum is divided by the window coherent sum $S_1 = \sum w[n]$.

### 4. Single-Sided Spectrum Scaling
For real-valued time-domain signals, the two-sided spectrum is conjugate-symmetric: $X[-k] = X^*[k]$.
AudioClean DSP folds the negative-frequency power into the positive half:
- Interior bins ($0 < k < \text{Nyquist}$) are scaled by $2.0 / S_1$.
- The DC bin ($k = 0$) and exact Nyquist bin ($k = N/2$) represent unique real-valued components and are scaled strictly by $1.0 / S_1$.
This calibration guarantees that an input sinusoid $A \sin(2\pi f t)$ displays a peak magnitude of exactly $A$ ($20 \log_{10}(A)\text{ dBFS}$).

### 5. Short-Time Fourier Transform (STFT) & Spectrograms
Audio and speech signals are inherently non-stationary: their spectral content evolves continuously over time.
The STFT slides a short analysis window $w[n]$ of length $M = \text{nperseg}$ across the signal with hop size $R = \text{nperseg} - \text{noverlap}$:
$$X[m, k] = \sum_{n=0}^{M-1} x[n + m R] w[n] e^{-j 2\pi k n / M}$$

The **Spectrogram** represents the square magnitude $|X[m, k]|^2$, expressed as Power Spectral Density (PSD in $\text{dB/Hz}$):
$$\text{PSD}_{\text{dB}}[m, k] = 10 \log_{10} \left( \max(|X[m, k]|^2, 10^{-15}) \right)$$

- **Time-Frequency Uncertainty (Gabor Limit):**
  $$\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$$
  Choosing a shorter segment length (`nperseg`) improves temporal resolution (precise onset timing) at the expense of frequency resolution (broader spectral peaks). Conversely, larger `nperseg` resolves close harmonic overtones but smears temporal transients.
---

## 10. Quantitative Performance Evaluation (Phase 6)

### 1. Objective Metrics Formulations

Objective digital filter evaluation requires comparing the degraded (input) signal and restored (output) signal against an uncorrupted ground-truth clean reference $s_{\text{clean}}[n]$.

#### A. Signal-to-Noise Ratio (SNR)
For an evaluated discrete signal $s_{\text{eval}}[n]$ (either noisy input $s_{\text{noisy}}[n]$ or filtered output $s_{\text{filtered}}[n]$) and reference $s_{\text{clean}}[n]$:
$$e[n] = s_{\text{eval}}[n] - s_{\text{clean}}[n]$$

The average reference signal power and residual error/noise power are:
$$P_{\text{sig}} = \frac{1}{N} \sum_{n=0}^{N-1} s_{\text{clean}}^2[n]$$
$$P_{\text{noise}} = \frac{1}{N} \sum_{n=0}^{N-1} e^2[n]$$

$$\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{P_{\text{sig}}}{P_{\text{noise}}}\right)$$

#### B. SNR Improvement ($\Delta\text{SNR}$)
$$\Delta\text{SNR}_{\text{dB}} = \text{SNR}_{\text{output, dB}} - \text{SNR}_{\text{input, dB}}$$

- **$\Delta\text{SNR} > 0$:** Objective reduction in overall noise/error power.
- **$\Delta\text{SNR} \le 0$:** Filtering has distorted in-band signal components or introduced more error energy than it suppressed (e.g. ringing, aggressive passband attenuation, or phase misalignment).

#### C. Root Mean Square Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} (s_{\text{clean}}[n] - s_{\text{eval}}[n])^2} = \sqrt{P_{\text{noise}}}$$

$$\text{RMSE Reduction (\%)} = \left(\frac{\text{RMSE}_{\text{input}} - \text{RMSE}_{\text{output}}}{\text{RMSE}_{\text{input}}}\right) \times 100\%$$

#### D. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{N} \sum_{n=0}^{N-1} |s_{\text{clean}}[n] - s_{\text{eval}}[n]|$$

---

### 2. Methodological Requirements & Scientific Limitations

#### 1. Absolute Requirement of a Clean Reference
Objective metrics like SNR, RMSE, and MAE **cannot** be computed from a single noisy or filtered audio file alone. Without a sample-exact clean reference, one cannot distinguish between desired audio harmonics and unwanted stationary/non-stationary noise. Blind (reference-free) quality estimation requires statistical psychoacoustic models (e.g., ITU-T P.862 PESQ, P.563, or DNSMOS), which fall outside linear discrete LTI filter evaluation.

#### 2. Temporal Alignment & Phase Distortion
Reference-based error metrics are exquisitely sensitive to phase alignment:
- An unfiltered clean signal delayed by even a few milliseconds relative to itself can yield an SNR below $0\text{ dB}$ due to destructive waveform cancellation:
  $$e[n] = s[n] - s[n - \tau]$$
- Therefore, objective comparative benchmarks in AudioClean DSP employ **zero-phase forward-backward filtering** (`filtfilt` / `sosfiltfilt`) to preserve exact sample alignment ($t=0$ group delay) when calculating SNR, RMSE, and MAE.

#### 3. Multichannel Energy Aggregation vs. Decibel Averaging
For stereo signals ($C = 2$):
- **Energy Pooling (Implemented):** Total signal power and total error power are aggregated across all channels and samples:
  $$P_{\text{total}} = \frac{1}{N \cdot C} \sum_{n=0}^{N-1} \sum_{c=0}^{C-1} s^2[n, c]$$
  $$\text{SNR}_{\text{overall}} = 10 \log_{10}\left(\frac{P_{\text{sig, total}}}{P_{\text{noise, total}}}\right)$$
- **Pitfall of Arithmetic dB Averaging:** Averaging decibel values directly ($\frac{\text{SNR}_1 + \text{SNR}_2}{2}$) violates logarithmic mathematics: the decibel scale is logarithmic, so an arithmetic mean of dB numbers does not equal the logarithm of average power ratios.

#### 4. Edge Cases: Perfect Reconstruction & Silence
- **Perfect Reconstruction ($P_{\text{noise}} = 0$):** When the evaluated signal exactly equals the clean reference, error is identically zero. $\text{SNR} = 10 \log_{10}(P_{\text{sig}} / 0) = +\infty\text{ dB}$. AudioClean DSP safely returns `float('inf')`.
- **Zero-Energy Reference ($P_{\text{sig}} = 0$):** When the reference signal is completely silent, the denominator of SNR is zero and clean signal power is undefined. AudioClean DSP raises an explicit `ValueError`.

#### 5. Filtering Processing Time Benchmarking
- **Isolated Timing:** Timing routines benchmark the core filtering call exclusively, completely excluding file I/O, spectrogram generation, Matplotlib plotting, and metric calculation.
- **Warm-Up Runs:** Prior to timed trials, several unrecorded warm-up runs (default: 5) are executed to allow CPU caches, NumPy thread pools, and interpreter states to stabilize.
- **Repeated Trials & Median Reporting:** The system records 30 independent measurement trials and reports the **median** processing time along with the range $[\min, \max]$ and standard deviation. The median provides robust resistance to operating-system background scheduling jitter.
- **Scientific Caveat:** Timing benchmarks depend on the host CPU architecture, RAM speed, operating system scheduling, buffer lengths, and NumPy BLAS/LAPACK thread counts. A benchmark on a local desktop demonstrates relative algorithmic efficiency (e.g. 6th-order IIR SOS vs 101-tap FIR convolution) but is not a universal hardware constant.
