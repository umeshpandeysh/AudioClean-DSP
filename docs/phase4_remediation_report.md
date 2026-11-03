# AudioClean DSP — Phase 4 Remediation Report: Streamlit Application

## 1. Executive Summary
During the Phase 4 audit of `app.py`, two critical operational defects and several misleading technical descriptions were identified and remediated. The application architecture and its six core workspaces were preserved in their entirety. All fixes were validated via automated contract and unit tests in `tests/test_app.py`, interactive `streamlit.testing.v1.AppTest` executions, and headless server verification.

---

## 2. Identified Defects & Implemented Corrections

### Defect 1: Uploaded Audio Reloaded on Every Streamlit Rerun
- **Identified Defect:** Streamlit reruns scripts from top to bottom on every user interaction (e.g., parameter slider adjustment, button click, workspace navigation). The previous file uploader implementation repeatedly read the uploaded file buffer on every rerun, resetting `audio_data`, `reference_audio`, `noisy_audio`, and `source_type`. This destroyed synthesized noise and filter outputs whenever the user interacted with the UI.
- **Implemented Remediation:**
  - Implemented content-based SHA-256 fingerprinting via `compute_file_hash(raw_bytes)`.
  - Stored `primary_file_hash`, `ref_file_hash`, and `active_ingest_mode` in `st.session_state`.
  - Reinitialized audio state **only** when a genuinely new file is uploaded or when switching between Workflow A and Workflow B.
  - Cleared stale external references when primary recordings change, preventing a reference from improperly attaching to an unrelated signal.
  - Preserved working signal buffers (`noisy_audio` and `noise_component`) across normal reruns and navigation.

### Defect 2: Independent Normalization Breaking Reference-Based Evaluation
- **Identified Defect:** The previous normalization button modified `noisy_audio` directly to peak 1.0 without applying the same gain factor to `reference_audio`. This altered the relative power between clean ground truth and degraded audio, falsifying SNR, RMSE, and MAE calculations.
- **Implemented Remediation:**
  - Added `apply_linked_normalization(noisy, reference, audio_data, target_peak=1.0)`.
  - Measured the maximum absolute peak of `noisy_audio` and scaled all linked arrays by the exact identical scalar gain factor: $\text{gain} = 1.0 / \max(|y_{\text{noisy}}|)$.
  - Guaranteed that the Signal-to-Noise Ratio (SNR) and relative distortion metrics before and after normalization remain mathematically invariant.
  - Updated button label to `"Normalize Signal (Linked Gain)"` to accurately describe the operation.

### Defect 3: Misleading Filter & Timing Technical Descriptions
- **Identified Defect:** Causal FIR was ambiguously labeled in comparison tables, `filtfilt` was labeled "zero latency" (ignoring that it is an offline, non-causal two-pass operation), and IIR notch filters were incorrectly characterized using the Butterworth −3.01 dB half-power cutoff convention.
- **Implemented Remediation:**
  - Updated descriptions to clarify that `filtfilt` is an *"offline non-causal forward-backward filtering operation with zero phase shift"*.
  - Updated causal FIR descriptions to explicitly highlight *"real-time causal, strictly linear phase, group delay $\tau = (M-1)/2$"*.
  - Updated IIR notch descriptions to state *"Narrowband notch rejection null at $f_0$ with Quality Factor $Q$"*, distinguishing it from Butterworth lowpass/highpass -3.01 dB conventions.
  - In Workflow B no-reference comparison, displayed speedup as `"N/A"` if measured latency is invalid or zero, rather than displaying an arbitrary `1.0x`.
  - Clarified in Workspace 1 that matching sample rate and shape does not prove audio synchronization, instructing the user that external references must already be time-aligned.

### Defect 4: Short Signal Crashing in STFT Spectrogram
- **Identified Defect:** When input recordings were shorter than the standard STFT segment length (`nperseg = 1024` or `4096`), `compute_spectrogram` threw a SciPy `ValueError`, crashing Workspace 6.
- **Implemented Remediation:**
  - Inspected input audio length and dynamically adapted the selectable `nperseg` options to `[s for s in [256, 512, 1024, 2048, 4096] if s <= sig_len]`.
  - If a signal is shorter than 256 samples, displayed a graceful warning rather than crashing.

### Defect 5: Workflow A / Workflow B Provenance Leakage
- **Identified Defect:** When switching the ingestion radio button from Workflow B to Workflow A, the existing loaded recording (`audio_data`) was unconditionally copied into `reference_audio` and treated as clean ground truth. This allowed degraded recordings to masquerade as clean references.
- **Implemented Remediation:**
  - Tracked source recording provenance in `st.session_state.source_type` (`"synthetic"`, `"clean_upload"`, or `"noisy_upload"`).
  - Enforced that switching to Workflow A only restores `reference_audio` if the recording originated from the synthetic demo tone or a verified clean upload.
  - If a degraded recording was ingested under Workflow B, switching to Workflow A keeps `reference_audio = None`, renders an explicit warning banner, and provides a `"Reset to Synthetic Clean Demo Tone"` button.

### Defect 6: Stale External Reference Lifecycle & Invalidation
- **Identified Defect:** When a primary recording was replaced with another recording having the same shape and sample rate, an existing external reference remained attached. Additionally, clearing the reference file uploader left stale reference state, and uploading an invalid reference did not clear prior state.
- **Implemented Remediation:**
  - Explicitly bound attached references to the primary recording SHA-256 fingerprint (`ref_primary_hash`).
  - Immediately detached any reference (`reference_audio = None`, `ref_file_hash = None`, `ref_primary_hash = None`) when primary audio is replaced or cleared.
  - Reset reference state when the user clears the reference uploader widget (`ref_file is None`).
  - Guaranteed that an invalid reference upload resets reference state to `None` rather than retaining a previous reference.

### Defect 7: Unsafe Speedup Ratio Formatting
- **Identified Defect:** In Workspace 5 comparative benchmark rendering, `speedup_str = f"{report.speedup_ratio:.2f}x" if report.speedup_ratio > 0 else "N/A"` evaluated `report.speedup_ratio > 0`. When `speedup_ratio` was `None`, Python raised `TypeError: '>' not supported between instances of 'NoneType' and 'int'`.
- **Implemented Remediation:**
  - Implemented `format_speedup(val: Optional[Union[float, int]]) -> str` helper.
  - Safely verified non-None, finite, and positive values before numeric comparisons, returning `"N/A"` for non-positive, non-finite, or None values without raising exceptions.
  - Replaced repetitive metric references in the benchmark markdown table to correctly display `iir_metrics` for RMSE and MAE.

---

## 3. Automated & Interactive Test Verification

### 3.1 Test Suite Results
All 20 tests in `tests/test_app.py` passed:
- `test_audio_to_bytes_valid_wav`: Verified valid 16-bit PCM WAV in-memory encoding.
- `test_audio_to_bytes_clipping_protection`: Verified clamping of amplitudes exceeding $[-1.0, 1.0]$.
- `test_compute_file_hash_deterministic`: Verified SHA-256 fingerprint consistency.
- `test_validate_reference_match_compatible`: Verified strict validation on compatible pairs.
- `test_validate_reference_match_sample_rate_mismatch`: Verified rejection of mismatched sample rates.
- `test_validate_reference_match_shape_mismatch`: Verified rejection of mismatched array shapes.
- `test_validate_reference_match_non_finite`: Verified rejection of NaN/Inf values.
- `test_apply_linked_normalization_preserves_relative_gain`: Verified mathematical invariance of SNR before and after linked normalization.
- `test_apply_linked_normalization_silent_signal`: Verified safe handling of zero-energy signals.
- `test_streamlit_app_interactive_session_state_and_navigation`: Executed interactive `streamlit.testing.v1.AppTest`, verifying clean session state initialization and signal preservation across app reruns.
- `test_format_speedup_valid`: Verified speedup ratio formatting for standard positive values.
- `test_format_speedup_none`: Verified `None` input safely returns `"N/A"`.
- `test_format_speedup_non_finite_and_negative`: Verified non-finite, negative, and zero inputs return `"N/A"`.
- `test_workflow_transition_provenance_preservation`: Verified strict provenance enforcement when transitioning between workflows.
- `test_reference_lifecycle_primary_replacement`: Verified external reference detaches on primary recording change.
- `test_reference_lifecycle_uploader_cleared`: Verified reference clears when file uploader is cleared.
- `test_reference_lifecycle_same_shape_hash_invalidation`: Verified hash-based invalidation prevents stale reference carryover between same-shape signals.
- `test_reference_lifecycle_rejected_reference_clears_prior_state`: Verified invalid reference upload resets prior reference state.
- `test_streamlit_apptest_workflow_toggle`: Interactive AppTest verification of workflow radio toggle and reference state.
- `test_streamlit_apptest_workspace_switching`: Interactive AppTest verification of smooth navigation across all six workspaces.

### 3.2 Headless Streamlit Launch
Executed:
```powershell
python -m streamlit run app.py --server.headless=true
```
- App started cleanly with Uvicorn server on port 8503 without startup warnings or exceptions.

---

## 4. Phase 4 Exit Gate Verdict
**STATUS: PASSED (APPROVED)**
- All state, provenance, lifecycle, and normalization defects resolved.
- Technical descriptions corrected across all workspaces.
- Automated and interactive AppTest suites (20 tests) passing cleanly at 100%.
