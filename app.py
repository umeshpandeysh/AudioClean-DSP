"""AudioClean DSP — Interactive Digital Filter and Noise Removal Engineering Application.

Phase 4 Implementation for AudioClean DSP.
"""

import hashlib
import io
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

import src.audio_io as aio
import src.comparison as comp
import src.fft_analysis as fft_mod
import src.fir_filters as fir_mod
import src.iir_filters as iir_mod
import src.metrics as metrics_mod
import src.noise_generator as noise_mod
import src.preprocessing as pre_mod
import src.visualization as vis_mod

st.set_page_config(
    page_title="AudioClean DSP — Engineering Dashboard",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --ac-bg:#f5f7fb; --ac-surface:#ffffff; --ac-ink:#17263c;
      --ac-muted:#68788e; --ac-blue:#2563eb; --ac-cyan:#0ea5e9;
      --ac-line:#e5eaf2; --ac-navy:#101e33; --ac-shadow:0 8px 26px rgba(19,38,65,.055);
    }
    html,body,[class*="css"] {font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
    .stApp {background:var(--ac-bg);color:var(--ac-ink);}
    [data-testid="stHeader"] {background:rgba(245,247,251,.92);}
    [data-testid="stToolbar"] {right:1rem;}
    .block-container {max-width:1480px;padding-top:1.35rem;padding-bottom:3rem;}
    h1,h2,h3,h4,p,label {color:var(--ac-ink);}
    h1,h2,h3 {letter-spacing:-.035em;}
    h2 {font-size:1.38rem;font-weight:750;}
    h3 {font-size:1.12rem;font-weight:720;}
    .ac-topbar {display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.2rem 0 1.05rem;margin-bottom:1.15rem;border-bottom:1px solid var(--ac-line);}
    .ac-brand {display:flex;align-items:center;gap:.75rem;}
    .ac-brand-mark {display:flex;align-items:center;justify-content:center;width:2.45rem;height:2.45rem;border-radius:12px;background:linear-gradient(145deg,#2563eb,#0891b2);color:white;font-size:1.2rem;font-weight:800;box-shadow:0 5px 13px rgba(37,99,235,.2);}
    .ac-brand-name {font-size:1.08rem;font-weight:800;letter-spacing:-.035em;color:#14243b;line-height:1.2;}
    .ac-brand-sub {font-size:.7rem;color:#718096;margin-top:.12rem;letter-spacing:.035em;}
    .ac-top-meta {display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;justify-content:flex-end;}
    .ac-top-chip {font-size:.72rem;font-weight:650;color:#42546d;background:#fff;border:1px solid var(--ac-line);border-radius:999px;padding:.38rem .68rem;}
    .ac-live-dot {display:inline-block;width:7px;height:7px;border-radius:50%;background:#16a34a;margin-right:.38rem;}
    .ac-page-head {margin:.25rem 0 1.05rem;}
    .ac-eyebrow {font-size:.68rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;color:#2563eb;margin-bottom:.38rem;}
    .ac-page-title {font-size:clamp(1.55rem,2.5vw,2.05rem);font-weight:800;line-height:1.15;letter-spacing:-.05em;color:#14243b;margin:0;}
    .ac-page-copy {font-size:.91rem;color:var(--ac-muted);line-height:1.55;margin:.42rem 0 0;max-width:850px;}
    .ac-status-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem;margin:.9rem 0 1.4rem;}
    .ac-status-card {background:#fff;border:1px solid var(--ac-line);border-radius:13px;padding:.8rem .95rem;box-shadow:0 3px 12px rgba(20,40,70,.025);min-width:0;}
    .ac-status-label {font-size:.66rem;font-weight:750;letter-spacing:.08em;text-transform:uppercase;color:#78879b;}
    .ac-status-value {font-size:1.04rem;font-weight:780;letter-spacing:-.025em;color:#17263c;margin-top:.22rem;overflow-wrap:anywhere;}
    .ac-status-note {font-size:.7rem;color:#8491a3;margin-top:.12rem;}
    .main-title {font-size:clamp(1.45rem,2.2vw,1.85rem);line-height:1.2;font-weight:800;letter-spacing:-.045em;color:#14243b;margin:1.15rem 0 .28rem;}
    .sub-title {font-size:.91rem;line-height:1.6;color:var(--ac-muted);margin:0 0 1.3rem;max-width:900px;}
    [data-testid="stVerticalBlockBorderWrapper"] {
      background:#ffffff;
      border:1px solid #e3eaf3;
      border-radius:15px;
      box-shadow:0 5px 18px rgba(20,40,70,.035);
      padding:1rem 1.05rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {gap:.8rem;}
    [data-testid="stSelectbox"],[data-testid="stSlider"],[data-testid="stNumberInput"],
    [data-testid="stRadio"],[data-testid="stCheckbox"] {margin-bottom:.18rem;}
    [data-testid="stSelectbox"] label,[data-testid="stSlider"] label,
    [data-testid="stNumberInput"] label,[data-testid="stRadio"] label {
      font-weight:650;color:#34445c;
    }
    [data-testid="stPlotlyChart"],[data-testid="stImage"] {border-radius:11px;}
        [data-testid="stMetric"] {background:#fff;border:1px solid var(--ac-line);padding:.8rem .95rem;border-radius:13px;box-shadow:0 3px 12px rgba(20,40,70,.025);}
    [data-testid="stMetricLabel"] {color:#718096;font-size:.72rem;font-weight:700;}
    [data-testid="stMetricValue"] {color:#17263c;font-weight:800;letter-spacing:-.035em;}
    section[data-testid="stSidebar"] {background:linear-gradient(180deg,#101e33 0%,#172d49 100%);border-right:1px solid rgba(255,255,255,.08);}
    section[data-testid="stSidebar"] * {color:#eaf2fb;}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#c0cede;}
    section[data-testid="stSidebar"] hr {border-color:rgba(219,234,254,.14);margin:1rem 0;}
    section[data-testid="stSidebar"] div[role="radiogroup"] {gap:.42rem;}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {background:rgba(255,255,255,.045);border:1px solid rgba(219,234,254,.09);border-radius:10px;padding:.58rem .68rem;transition:background .15s,border-color .15s;}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {background:rgba(255,255,255,.11);border-color:rgba(125,211,252,.38);}
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {font-size:.84rem;font-weight:650;}
    section[data-testid="stSidebar"] [data-testid="stAlert"] {background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);}
    .stButton>button,.stDownloadButton>button {min-height:2.55rem;border-radius:10px;border:1px solid #d9e2ef;font-weight:700;transition:all .15s ease;}
    .stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"] {background:linear-gradient(105deg,#2563eb,#087eaa);border:0;color:#fff;box-shadow:0 5px 14px rgba(37,99,235,.16);}
    .stButton>button:hover,.stDownloadButton>button:hover {border-color:#8bb8f3;transform:translateY(-1px);box-shadow:0 6px 16px rgba(27,62,110,.1);}
    [data-testid="stFileUploader"] section {background:#fbfdff;border:1.5px dashed #c3d2e5;border-radius:13px;}
    [data-testid="stFileUploader"] section:hover {border-color:#4f8fe8;background:#f5f9ff;}
    [data-testid="stAudio"] {border:1px solid var(--ac-line);border-radius:11px;padding:.25rem;background:#fff;}
    [data-testid="stExpander"],[data-testid="stVerticalBlockBorderWrapper"] {border-color:var(--ac-line);border-radius:12px;}
    .stAlert {border-radius:11px;border:1px solid rgba(148,163,184,.2);}
    [data-testid="stDataFrame"],[data-testid="stTable"] {border:1px solid var(--ac-line);border-radius:11px;overflow:hidden;}
    .stTabs [data-baseweb="tab"] {font-weight:700;padding:.68rem .9rem;}
    .stTabs [aria-selected="true"] {color:#2563eb;}
    div[data-testid="stMarkdownContainer"] hr {border-color:var(--ac-line);margin:1.2rem 0;}
    @media(max-width:900px) {.ac-status-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
    @media(max-width:640px) {
      .block-container{padding:1rem .9rem 2rem;}
      .ac-topbar{align-items:flex-start;flex-direction:column;}
      .ac-top-meta{justify-content:flex-start;}
      .ac-status-grid{gap:.5rem;}
      .ac-status-card{padding:.68rem .72rem;}
      .main-title{margin-top:.8rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Helper Functions: Audio In-Memory Encoding, Fingerprinting and Normalization
# -----------------------------------------------------------------------------
def audio_to_bytes(data: np.ndarray, sample_rate: int) -> bytes:
    """Encode floating-point audio data into in-memory 16-bit PCM WAV bytes without normalization."""
    buf = io.BytesIO()
    clipped = np.clip(data, -1.0, 1.0)
    sf.write(buf, clipped, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def render_audio_player(data: np.ndarray, sample_rate: int, label: str = "Audio Player") -> None:
    """Render an in-memory st.audio player."""
    audio_bytes = audio_to_bytes(data, sample_rate)
    st.caption(label)
    st.audio(audio_bytes, format="audio/wav")


def render_download_button(
    data: np.ndarray, sample_rate: int, filename: str, label: str = "Download Processed WAV"
) -> None:
    """Render a download button for processed audio without altering signal gain."""
    audio_bytes = audio_to_bytes(data, sample_rate)
    st.download_button(
        label=label,
        data=audio_bytes,
        file_name=filename,
        mime="audio/wav",
        use_container_width=True,
    )


def compute_file_hash(raw_bytes: bytes) -> str:
    """Compute stable SHA-256 fingerprint for uploaded file data."""
    return hashlib.sha256(raw_bytes).hexdigest()


def validate_reference_match(
    reference: np.ndarray, ref_sr: int, noisy: np.ndarray, noisy_sr: int
) -> Tuple[bool, str]:
    """Strict validation for reference audio upload matching sample rate, length, and channels."""
    if ref_sr != noisy_sr:
        return (
            False,
            f"Sample rate mismatch: Reference has {ref_sr} Hz, but noisy signal has {noisy_sr} Hz.",
        )
    if reference.shape != noisy.shape:
        return (
            False,
            f"Array shape mismatch: Reference has shape {reference.shape}, but noisy signal has {noisy.shape}. "
            "Channel count and sample length must match exactly.",
        )
    if not np.all(np.isfinite(reference)):
        return False, "Reference signal contains non-finite (NaN or Inf) values."
    return True, ""


def apply_linked_normalization(
    noisy_audio: np.ndarray,
    reference_audio: Optional[np.ndarray],
    audio_data: Optional[np.ndarray],
    target_peak: float = 1.0,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float]:
    """Normalize noisy audio to target_peak and scale linked reference audio by the EXACT same gain factor.

    This strictly preserves relative gain, SNR, RMSE reduction %, and MAE relationships.
    """
    max_val = float(np.max(np.abs(noisy_audio)))
    if max_val == 0.0 or not np.isfinite(max_val):
        return noisy_audio, reference_audio, audio_data, 1.0

    gain = float(target_peak / max_val)
    norm_noisy = noisy_audio * gain
    norm_ref = reference_audio * gain if reference_audio is not None else None
    norm_data = audio_data * gain if audio_data is not None else None
    return norm_noisy, norm_ref, norm_data, gain


def format_speedup(val: Optional[Union[float, int]]) -> str:
    """Format speedup ratio safely or return 'N/A' if None, non-finite, or non-positive.

    Guarantees no invalid None comparisons or crashing on non-finite values.
    """
    if val is None:
        return "N/A"
    try:
        f_val = float(val)
        if math.isfinite(f_val) and f_val > 0.0:
            return f"{f_val:.2f}x"
        return "N/A"
    except (TypeError, ValueError):
        return "N/A"


def reset_reference_uploader() -> None:
    """Explicitly invalidate and reset the external reference uploader widget lifecycle.

    Pops the existing widget key from session state and increments the version counter,
    ensuring Streamlit destroys the internal file buffer and cannot silently retain
    or reattach a previously uploaded reference file.
    """
    st.session_state.reference_audio = None
    st.session_state.ref_file_hash = None
    st.session_state.ref_primary_hash = None
    curr_v = st.session_state.get("ref_uploader_version", 0)
    old_key = f"ref_uploader_v{curr_v}"
    st.session_state.pop(old_key, None)
    st.session_state.ref_uploader_version = curr_v + 1


def reset_primary_uploader() -> None:
    """Explicitly reset the primary audio uploader and all associated reference state."""
    curr_v = st.session_state.get("primary_uploader_version", 0)
    old_key = f"primary_audio_uploader_v{curr_v}"
    st.session_state.pop(old_key, None)
    st.session_state.primary_uploader_version = curr_v + 1
    st.session_state.primary_file_hash = None
    reset_reference_uploader()

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "sample_rate" not in st.session_state:
    st.session_state.sample_rate = 44100
if "source_type" not in st.session_state:
    st.session_state.source_type = "synthetic"
if "reference_audio" not in st.session_state:
    st.session_state.reference_audio = None
if "noisy_audio" not in st.session_state:
    st.session_state.noisy_audio = None
if "noise_component" not in st.session_state:
    st.session_state.noise_component = None
if "primary_file_hash" not in st.session_state:
    st.session_state.primary_file_hash = None
if "ref_file_hash" not in st.session_state:
    st.session_state.ref_file_hash = None
if "ref_primary_hash" not in st.session_state:
    st.session_state.ref_primary_hash = None
if "active_ingest_mode" not in st.session_state:
    st.session_state.active_ingest_mode = "Workflow A"
if "ref_uploader_version" not in st.session_state:
    st.session_state.ref_uploader_version = 0
if "primary_uploader_version" not in st.session_state:
    st.session_state.primary_uploader_version = 0

# Synthesize default demo tone if no audio loaded
if st.session_state.audio_data is None:
    sr_demo = 44100
    t_demo = np.linspace(0.0, 1.5, int(sr_demo * 1.5), endpoint=False)
    demo_signal = 0.5 * np.sin(2 * np.pi * 440 * t_demo) + 0.25 * np.sin(2 * np.pi * 880 * t_demo)
    demo_signal = pre_mod.normalize_audio(demo_signal, target_peak=0.8)
    st.session_state.audio_data = demo_signal
    st.session_state.sample_rate = sr_demo
    st.session_state.source_type = "synthetic"
    st.session_state.reference_audio = demo_signal.copy()
    st.session_state.noisy_audio = demo_signal.copy()


# -----------------------------------------------------------------------------
# Sidebar Navigation and System Status
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding:0.45rem 0 0.8rem;">
      <div style="font-size:0.7rem;font-weight:800;letter-spacing:0.16em;color:#7dd3fc;text-transform:uppercase;">DSP ENGINEERING LAB</div>
      <div style="font-size:1.55rem;font-weight:850;letter-spacing:-0.05em;color:#ffffff;margin-top:0.22rem;">◉ AudioClean</div>
      <div style="font-size:0.88rem;font-weight:650;color:#d4e5f7;">Signal Processing Studio</div>
      <div style="font-size:0.72rem;color:#9fb6cf;margin-top:0.55rem;">Explore · Design · Compare · Analyze</div>
    </div>
    """,
    unsafe_allow_html=True,
)

workspace = st.sidebar.radio(
    "WORKSPACES",
    [
        "1  ·  Audio Workspace",
        "2  ·  Noise Generation",
        "3  ·  FIR Filter Designer",
        "4  ·  IIR Filter Designer",
        "5  ·  FIR vs. IIR Comparison",
        "6  ·  Signal Analysis",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Active Audio Status")

active_sig = (
    st.session_state.noisy_audio
    if st.session_state.noisy_audio is not None
    else st.session_state.audio_data
)
sr = st.session_state.sample_rate

if active_sig is not None:
    dur = len(active_sig) / sr
    channels = 1 if active_sig.ndim == 1 else active_sig.shape[1]
    has_ref = st.session_state.reference_audio is not None
    mode_text = "Workflow A (Clean Reference Ground-Truth)" if has_ref else "Workflow B (No Reference / Real-World)"

    st.sidebar.info(
        f"**Mode:** {mode_text}\n\n"
        f"**Sample Rate:** {sr:,} Hz\n\n"
        f"**Duration:** {dur:.2f} s ({len(active_sig):,} samples)\n\n"
        f"**Channels:** {'Mono' if channels == 1 else 'Stereo'}\n\n"
        f"**Peak Level:** {np.max(np.abs(active_sig)):.3f}"
    )
else:
    st.sidebar.warning("No audio signal loaded.")

st.sidebar.markdown("---")
st.sidebar.caption("AUDIOCLEAN DSP  ·  VERSION 1.4")
st.sidebar.caption("Academic digital signal processing project")

workspace_names = {
    "1": ("Audio ingestion", "Load, inspect and prepare your recording."),
    "2": ("Noise laboratory", "Create controlled noise for repeatable experiments."),
    "3": ("FIR filter design", "Explore windowed finite impulse response filters."),
    "4": ("IIR filter design", "Configure Butterworth and notch filters."),
    "5": ("Comparative evaluation", "Compare FIR and IIR under matched conditions."),
    "6": ("Signal diagnostics", "Inspect spectra, dominant peaks and spectrograms."),
}
workspace_key = workspace.split()[0]
workspace_name, workspace_copy = workspace_names.get(workspace_key, ("DSP workspace", "Explore the signal processing workflow."))

st.markdown(
    """
    <div class="ac-topbar">
      <div class="ac-brand">
        <div class="ac-brand-mark">A</div>
        <div>
          <div class="ac-brand-name">AudioClean <span style="color:#2563eb;">DSP</span></div>
          <div class="ac-brand-sub">DIGITAL SIGNAL PROCESSING WORKBENCH</div>
        </div>
      </div>
      <div class="ac-top-meta">
        <span class="ac-top-chip"><span class="ac-live-dot"></span>Session active</span>
        <span class="ac-top-chip">Python · SciPy</span>
        <span class="ac-top-chip">Build 1.4</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="ac-page-head">
      <div class="ac-eyebrow">Engineering workspace / {workspace_key} of 6</div>
      <div class="ac-page-title">{workspace_name}</div>
      <p class="ac-page-copy">{workspace_copy} Configure the signal-processing workflow and inspect the resulting evidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if active_sig is not None:
    channels_label = "Mono" if channels == 1 else "Stereo"
    reference_label = "Attached" if has_ref else "Not attached"
    st.markdown(
        f"""
        <div class="ac-status-grid">
          <div class="ac-status-card"><div class="ac-status-label">Signal sample rate</div><div class="ac-status-value">{sr:,} Hz</div><div class="ac-status-note">Current working signal</div></div>
          <div class="ac-status-card"><div class="ac-status-label">Recording duration</div><div class="ac-status-value">{dur:.2f} s</div><div class="ac-status-note">{len(active_sig):,} samples</div></div>
          <div class="ac-status-card"><div class="ac-status-label">Channel layout</div><div class="ac-status-value">{channels_label}</div><div class="ac-status-note">Loaded audio format</div></div>
          <div class="ac-status-card"><div class="ac-status-label">Clean reference</div><div class="ac-status-value">{reference_label}</div><div class="ac-status-note">Objective metrics availability</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# WORKSPACE 1: Audio Workspace
# =============================================================================
if workspace.startswith("1"):
    st.markdown('<div class="main-title">1. Audio Workspace and Ingestion</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Ingest, inspect, pre-process, and configure ground-truth audio workflows.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        col_up1, col_up2 = st.columns([1.2, 1.0])

        with col_up1:
            st.subheader("Audio Ingestion")
            ingest_mode = st.radio(
                "Select Ingestion Workflow:",
                [
                    "Workflow A: Clean Audio (Ground-truth available; enable noise synthesis and objective metrics)",
                    "Workflow B: Real-World Degraded Audio (No ground-truth; no-reference evaluation)",
                ],
                index=0 if st.session_state.active_ingest_mode == "Workflow A" else 1,
            )

            selected_mode_key = "Workflow A" if "Workflow A" in ingest_mode else "Workflow B"

            # Detect ingest mode change
            if selected_mode_key != st.session_state.active_ingest_mode:
                st.session_state.active_ingest_mode = selected_mode_key
                reset_reference_uploader()
                if selected_mode_key == "Workflow A":
                    if st.session_state.source_type in ("synthetic", "clean_upload"):
                        st.session_state.reference_audio = (
                            st.session_state.audio_data.copy() if st.session_state.audio_data is not None else None
                        )
                    else:
                        # Degraded audio from Workflow B must NOT be treated as clean ground truth
                        st.session_state.reference_audio = None
                else:
                    st.session_state.reference_audio = None
                st.rerun()

            primary_key = f"primary_audio_uploader_v{st.session_state.primary_uploader_version}"
            uploaded_file = st.file_uploader("Upload Audio Recording (.wav)", type=["wav"], key=primary_key)
            if uploaded_file is not None:
                raw_bytes = uploaded_file.getvalue()
                f_hash = compute_file_hash(raw_bytes)

                # Reinitialize ONLY when a new file is uploaded
                if f_hash != st.session_state.primary_file_hash:
                    try:
                        buf = io.BytesIO(raw_bytes)
                        data, in_sr = sf.read(buf, dtype="float64")
                        pre_mod.validate_audio(data, in_sr)
                        st.session_state.sample_rate = in_sr
                        st.session_state.primary_file_hash = f_hash
                        # Invalidate reference uploader lifecycle and state immediately
                        reset_reference_uploader()
                        st.session_state.noise_component = None

                        if selected_mode_key == "Workflow A":
                            st.session_state.audio_data = data
                            st.session_state.reference_audio = data.copy()
                            st.session_state.noisy_audio = data.copy()
                            st.session_state.source_type = "clean_upload"
                            st.success(f"Loaded new Clean Reference audio ({in_sr:,} Hz, shape: {data.shape})")
                        else:
                            st.session_state.audio_data = data
                            st.session_state.reference_audio = None
                            st.session_state.noisy_audio = data.copy()
                            st.session_state.source_type = "noisy_upload"
                            st.success(f"Loaded new Real-World Degraded audio ({in_sr:,} Hz, shape: {data.shape})")
                    except Exception as e:
                        st.error(f"Failed to load audio file: {e}")
            elif uploaded_file is None and st.session_state.primary_file_hash is not None:
                # Primary upload removed: reset uploader lifecycle and restore default synthetic demo
                reset_primary_uploader()
                st.session_state.noise_component = None
                sr_demo = 44100
                t_demo = np.linspace(0.0, 1.5, int(sr_demo * 1.5), endpoint=False)
                demo_signal = 0.5 * np.sin(2 * np.pi * 440 * t_demo) + 0.25 * np.sin(2 * np.pi * 880 * t_demo)
                demo_signal = pre_mod.normalize_audio(demo_signal, target_peak=0.8)
                st.session_state.audio_data = demo_signal
                st.session_state.sample_rate = sr_demo
                st.session_state.source_type = "synthetic"
                if selected_mode_key == "Workflow A":
                    st.session_state.reference_audio = demo_signal.copy()
                    st.session_state.noisy_audio = demo_signal.copy()
                else:
                    st.session_state.reference_audio = None
                    st.session_state.noisy_audio = demo_signal.copy()
                st.rerun()

            if selected_mode_key == "Workflow A" and st.session_state.source_type == "noisy_upload":
                st.warning(
                    "⚠️ **Clean Ground Truth Required:** The currently loaded recording was ingested as "
                    "real-world degraded audio (Workflow B). Workflow A requires a verified clean recording. "
                    "Please upload a clean audio recording above or reset to the synthetic clean demo tone."
                )
                if st.button("Reset to Synthetic Clean Demo Tone", key="btn_reset_demo_workflow_a"):
                    reset_primary_uploader()
                    sr_demo = 44100
                    t_demo = np.linspace(0.0, 1.5, int(sr_demo * 1.5), endpoint=False)
                    demo_signal = 0.5 * np.sin(2 * np.pi * 440 * t_demo) + 0.25 * np.sin(2 * np.pi * 880 * t_demo)
                    demo_signal = pre_mod.normalize_audio(demo_signal, target_peak=0.8)
                    st.session_state.audio_data = demo_signal
                    st.session_state.sample_rate = sr_demo
                    st.session_state.source_type = "synthetic"
                    st.session_state.reference_audio = demo_signal.copy()
                    st.session_state.noisy_audio = demo_signal.copy()
                    st.session_state.noise_component = None
                    st.success("Reset to synthetic clean demo signal.")
                    st.rerun()

            # Optional separate clean reference for Workflow B
            if selected_mode_key == "Workflow B":
                st.markdown("#### Optional Ground-Truth Reference Upload")
                st.caption(
                    "If you have an independent, synchronized clean reference for this degraded recording, "
                    "upload it below. Note: Matching sample rate and shape does not guarantee synchronization; "
                    "the user must ensure the reference is aligned."
                )
                ref_uploader_key = f"ref_uploader_v{st.session_state.ref_uploader_version}"
                ref_file = st.file_uploader(
                    "Upload Reference Audio (.wav)",
                    type=["wav"],
                    key=ref_uploader_key,
                )
                if ref_file is None:
                    if st.session_state.ref_file_hash is not None or st.session_state.reference_audio is not None:
                        reset_reference_uploader()
                        st.info("External reference removed. Switched to no-reference evaluation.")                    st.rerun()
                else:
                    current_primary_hash = st.session_state.primary_file_hash or "synthetic_demo"
                    r_bytes = ref_file.getvalue()
                    r_hash = compute_file_hash(r_bytes)
                    if r_hash != st.session_state.ref_file_hash or st.session_state.ref_primary_hash != current_primary_hash:
                        try:
                            r_buf = io.BytesIO(r_bytes)
                            ref_data, ref_sr = sf.read(r_buf, dtype="float64")
                            target_noisy = (
                                st.session_state.noisy_audio
                                if st.session_state.noisy_audio is not None
                                else st.session_state.audio_data
                            )
                            valid, msg = validate_reference_match(
                                ref_data, ref_sr, target_noisy, st.session_state.sample_rate
                            )
                            if valid:
                                st.session_state.reference_audio = ref_data
                                st.session_state.ref_file_hash = r_hash
                                st.session_state.ref_primary_hash = current_primary_hash
                                st.success("Clean reference attached! Workflow A objective metrics enabled.")
                            else:
                                reset_reference_uploader()
                                st.error(f"Reference rejected: {msg}")
                                st.rerun()
                        except Exception as e:
                            reset_reference_uploader()
                            st.error(f"Failed to load reference: {e}")
                            st.rerun()

        with col_up2:
            st.subheader("Preprocessing and Controls")
            cur_sig = (
                st.session_state.noisy_audio
                if st.session_state.noisy_audio is not None
                else st.session_state.audio_data
            )

            if cur_sig is not None:
                info = aio.get_audio_info(cur_sig, st.session_state.sample_rate)
                st.markdown(
                    f"- **Sample Rate:** `{info['sample_rate']:,} Hz`\n"
                    f"- **Channels:** `{info['num_channels']}` ({'Mono' if info['num_channels'] == 1 else 'Stereo'})\n"
                    f"- **Samples:** `{info['num_samples']:,}` ({info['duration_seconds']:.3f} s)\n"
                    f"- **Peak Amplitude:** `{info['peak_amplitude']:.4f}`\n"
                    f"- **RMS Level:** `{info['rms_amplitude']:.4f}`\n"
                    f"- **Crest Factor:** `{info['crest_factor_db']:.2f} dB`"
                )

                c1, c2 = st.columns(2)
                with c1:
                    if cur_sig.ndim > 1 and cur_sig.shape[1] > 1:
                        if st.button("Convert to Mono", use_container_width=True):
                            st.session_state.noisy_audio = pre_mod.convert_to_mono(st.session_state.noisy_audio)
                            if st.session_state.reference_audio is not None:
                                st.session_state.reference_audio = pre_mod.convert_to_mono(
                                    st.session_state.reference_audio
                                )
                            st.session_state.audio_data = pre_mod.convert_to_mono(st.session_state.audio_data)
                            st.success("Downmixed to mono.")
                            st.rerun()
                    else:
                        st.button("Mono Signal (Downmix Inactive)", disabled=True, use_container_width=True)

                with c2:
                    if st.button("Normalize Signal (Linked Gain)", use_container_width=True):
                        norm_n, norm_r, norm_d, g = apply_linked_normalization(
                            st.session_state.noisy_audio,
                            st.session_state.reference_audio,
                            st.session_state.audio_data,
                            target_peak=1.0,
                        )
                        st.session_state.noisy_audio = norm_n
                        st.session_state.reference_audio = norm_r
                        st.session_state.audio_data = norm_d
                        st.success(f"Signal normalized (gain: {g:.3f}x). Relative reference gain preserved.")
                        st.rerun()

                st.markdown("#### Audio Player")
                render_audio_player(cur_sig, st.session_state.sample_rate, "Active Signal Playback")
                render_download_button(
                    cur_sig, st.session_state.sample_rate, "audio_workspace_active.wav", "Download Active Audio"
                )

        st.markdown("---")
        with st.container(border=True):
        st.subheader("Time-Domain Waveform")
            if cur_sig is not None:
                fig_wave = vis_mod.plot_waveform(
                    cur_sig, st.session_state.sample_rate, title="Active Signal Time-Domain Waveform"
                )
                st.pyplot(fig_wave)
                plt.close(fig_wave)

        # =============================================================================
# WORKSPACE 2: Controlled Noise Generation
# =============================================================================
elif workspace.startswith("2"):
    st.markdown('<div class="main-title">2. Controlled Noise Generation Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Synthesize calibrated noise models on clean reference signals for objective evaluation.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.reference_audio is None:
        st.warning(
            "Ground-truth clean reference is not active. Please load a clean audio signal in "
            "Workspace 1 (Workflow A) to synthesize controlled noise models with a known ground-truth."
        )
    else:
        ref = st.session_state.reference_audio
        sr = st.session_state.sample_rate

        with st.container(border=True):
            col_ctrl, col_view = st.columns([1.1, 1.2])

            with col_ctrl:
                st.subheader("Noise Model Configuration")
                noise_type = st.selectbox(
                    "Select Noise Model:",
                    [
                        "White Gaussian Noise (WGN)",
                        "Sinusoidal Mains Hum (50/60 Hz)",
                        "High-Frequency Interference Tone",
                        "Composite Noise (Combined Hum + WGN + Tone)",
                    ],
                )

                if noise_type == "White Gaussian Noise (WGN)":
                    target_snr = st.slider("Target SNR (dB):", min_value=-10.0, max_value=40.0, value=10.0, step=0.5)
                    seed_val = st.number_input("Random Seed (optional):", min_value=0, max_value=999999, value=42)

                    if st.button("Synthesize and Apply WGN", type="primary", use_container_width=True):
                        noisy, noise = noise_mod.add_white_gaussian_noise(ref, target_snr_db=target_snr, seed=int(seed_val))
                        st.session_state.noisy_audio = noisy
                        st.session_state.noise_component = noise
                        st.success(f"Synthesized WGN at {target_snr:.1f} dB SNR.")

                elif noise_type == "Sinusoidal Mains Hum (50/60 Hz)":
                    hum_freq = st.selectbox("Fundamental Hum Frequency:", [50.0, 60.0], index=0)
                    hum_amp = st.slider("Hum Amplitude (relative):", min_value=0.01, max_value=0.5, value=0.15, step=0.01)
                    include_harmonics = st.checkbox("Include Harmonics (2nd, 3rd, 4th)", value=True)
                    harmonics = [2, 3, 4] if include_harmonics else []
                    decay = st.slider("Harmonic Decay Factor:", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

                    if st.button("Synthesize and Apply Mains Hum", type="primary", use_container_width=True):
                        noisy, noise = noise_mod.add_sinusoidal_hum(
                            ref, sample_rate=sr, freq=hum_freq, amplitude=hum_amp, harmonics=harmonics, harmonic_decay=decay
                        )
                        st.session_state.noisy_audio = noisy
                        st.session_state.noise_component = noise
                        st.success(f"Synthesized {hum_freq:.0f} Hz hum with amplitude {hum_amp:.2f}.")

                elif noise_type == "High-Frequency Interference Tone":
                    max_allowable_f = float(sr // 2 - 100)
                    tone_freq = st.slider(
                        "Interference Frequency (Hz):",
                        min_value=1000.0,
                        max_value=max_allowable_f,
                        value=min(6000.0, max_allowable_f * 0.75),
                        step=100.0,
                    )
                    tone_amp = st.slider("Tone Amplitude:", min_value=0.01, max_value=0.5, value=0.12, step=0.01)

                    if st.button("Synthesize and Apply HF Tone", type="primary", use_container_width=True):
                        noisy, noise = noise_mod.add_high_frequency_noise(ref, sample_rate=sr, freq=tone_freq, amplitude=tone_amp)
                        st.session_state.noisy_audio = noisy
                        st.session_state.noise_component = noise
                        st.success(f"Synthesized {tone_freq:.0f} Hz interference tone.")

                elif noise_type == "Composite Noise (Combined Hum + WGN + Tone)":
                    st.caption("Combine multiple isolated degradation components via add_noise_combination().")
                    wgn_snr = st.slider("WGN Component SNR (dB):", -5.0, 30.0, 15.0, 0.5)
                    hum_f = st.selectbox("Hum Component Frequency:", [50.0, 60.0], 0)
                    hum_a = st.slider("Hum Component Amplitude:", 0.01, 0.3, 0.1, 0.01)
                    tone_f = st.slider("HF Tone Frequency (Hz):", 1000.0, float(sr // 2 - 200), 5000.0, 100.0)
                    tone_a = st.slider("HF Tone Amplitude:", 0.01, 0.3, 0.08, 0.01)

                    if st.button("Synthesize and Apply Composite Noise", type="primary", use_container_width=True):
                        _, n_wgn = noise_mod.add_white_gaussian_noise(ref, target_snr_db=wgn_snr, seed=123)
                        _, n_hum = noise_mod.add_sinusoidal_hum(ref, sr, freq=hum_f, amplitude=hum_a, harmonics=[2, 3])
                        _, n_tone = noise_mod.add_high_frequency_noise(ref, sr, freq=tone_f, amplitude=tone_a)

                        noisy, comp_noise = noise_mod.add_noise_combination(ref, [n_wgn, n_hum, n_tone], prevent_clipping=True)
                        st.session_state.noisy_audio = noisy
                        st.session_state.noise_component = comp_noise
                        st.success("Synthesized and combined 3 noise components safely.")

                st.markdown("---")
                if st.button("Reset Working Signal to Clean Reference", use_container_width=True):
                    st.session_state.noisy_audio = ref.copy()
                    st.session_state.noise_component = None
                    st.info("Working signal reset to clean reference.")
                    st.rerun()

            with col_view:
                st.subheader("Spectral and Auditory Inspection")
                if st.session_state.noisy_audio is not None:
                    noisy_sig = st.session_state.noisy_audio
                    cur_snr = metrics_mod.calculate_snr(ref, noisy_sig)
                    cur_rmse = metrics_mod.calculate_rmse(ref, noisy_sig)

                    st.markdown(
                        f"<div class='metric-badge'>"
                        f"<b>Baseline SNR:</b> {cur_snr:.2f} dB &nbsp;|&nbsp; "
                        f"<b>Baseline RMSE:</b> {cur_rmse:.4f}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown("**Clean Audio:**")
                    render_audio_player(ref, sr, "Clean Reference")

                    st.markdown("**Degraded Working Audio:**")
                    render_audio_player(noisy_sig, sr, "Noisy Signal")

                    if st.session_state.noise_component is not None:
                        st.markdown("**Isolated Noise Component:**")
                        render_audio_player(st.session_state.noise_component, sr, "Noise Only")

                    # FFT comparison
                    f_ref, mag_ref, _ = fft_mod.compute_fft(ref, sr)
                    f_noi, mag_noi, _ = fft_mod.compute_fft(noisy_sig, sr)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(f_noi, 20 * np.log10(np.maximum(mag_noi, 1e-12)), label="Noisy Signal", color="#e74c3c", alpha=0.7)
                    ax.plot(f_ref, 20 * np.log10(np.maximum(mag_ref, 1e-12)), label="Clean Reference", color="#2ecc71", alpha=0.85)
                    ax.set_title("FFT Magnitude Comparison: Clean vs. Noisy")
                    ax.set_xlabel("Frequency (Hz)")
                    ax.set_ylabel("Magnitude (dBFS)")
                    ax.set_xlim(0, min(sr // 2, 10000))
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

# =============================================================================
# WORKSPACE 3: FIR Filter Designer
# =============================================================================
elif workspace.startswith("3"):
    st.markdown('<div class="main-title">3. FIR Digital Filter Designer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Design linear-phase windowed FIR filters with causal and zero-phase filtering modes.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.noisy_audio is None:
        st.error("No working audio signal available. Ingest audio in Workspace 1.")
    else:
        sig = st.session_state.noisy_audio
        ref = st.session_state.reference_audio
        sr = st.session_state.sample_rate
        nyq = sr / 2.0

        with st.container(border=True):
            col_cfg, col_res = st.columns([1.1, 1.2])

            with col_cfg:
                st.subheader("FIR Design Parameters")
                f_type = st.selectbox("Filter Topology:", ["lowpass", "highpass", "bandpass", "bandstop"], index=0)

                if f_type in ["lowpass", "highpass"]:
                    cutoff = st.slider(
                        "Cutoff Frequency fc (Hz):",
                        min_value=20.0,
                        max_value=float(nyq - 50.0),
                        value=min(1500.0, nyq * 0.4),
                        step=20.0,
                    )
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        f_low = st.number_input("Lower Cutoff (Hz):", min_value=20.0, max_value=float(nyq - 100.0), value=300.0, step=20.0)
                    with c2:
                        f_high = st.number_input("Upper Cutoff (Hz):", min_value=float(f_low + 20.0), max_value=float(nyq - 50.0), value=3000.0, step=20.0)
                    cutoff = [f_low, f_high]

                numtaps = st.slider(
                    "Filter Length (Taps M - odd for Type I):",
                    min_value=5,
                    max_value=501,
                    value=101,
                    step=2,
                )

                window_name = st.selectbox("Window Function:", ["hamming", "hann", "blackman"], index=0)

                mode_fir = st.radio(
                    "Filtering Mode:",
                    [
                        "zero_phase (filtfilt: offline non-causal forward-backward, zero phase shift)",
                        "causal (lfilter: real-time causal, strictly linear phase, group delay tau = (M-1)/2)",
                    ],
                    index=0,
                )
                mode_str = "zero_phase" if "zero_phase" in mode_fir else "causal"

                st.info(
                    "ℹ️ **FIR Cutoff Convention:** Windowed FIR via `firwin` designs to the **-6.02 dB** "
                    "(half-amplitude |H|=0.5) transition midpoint, whereas Butterworth IIR designs to **-3.01 dB** "
                    "(half-power |H|=1/sqrt(2))."
                )

                try:
                    fir_design = fir_mod.design_fir_filter(
                        sample_rate=sr,
                        filter_type=f_type,
                        cutoff_hz=cutoff,
                        numtaps=numtaps,
                        window=window_name,
                    )
                    design_ok = True
                except Exception as e:
                    st.error(f"FIR Design Error: {e}")
                    design_ok = False

                if design_ok:
                    tau_samples = (fir_design.numtaps - 1) // 2
                    tau_ms = (tau_samples / sr) * 1000.0
                    st.markdown(
                        f"- **Filter Order N:** `{fir_design.order}` (Taps: `{fir_design.numtaps}`)\n"
                        f"- **Group Delay tau:** `{tau_samples}` samples (`{tau_ms:.2f} ms`)\n"
                        f"- **Transition Window:** `{fir_design.window}`"
                    )

            with col_res:
                st.subheader("Frequency and Phase Response")
                if design_ok:
                    w_f, h_mag, h_db, phi = fir_mod.get_fir_frequency_response(fir_design, worN=2048)

                    fig_resp = vis_mod.plot_filter_response(
                        w_f, h_db, title=f"FIR {f_type.capitalize()} Magnitude Response ({window_name.capitalize()})", cutoff_hz=cutoff
                    )
                    st.pyplot(fig_resp)
                    plt.close(fig_resp)

                    fig_phase = vis_mod.plot_phase_response(
                        w_f, phi, title=f"FIR {f_type.capitalize()} Phase Response (Linear Phase)"
                    )
                    st.pyplot(fig_phase)
                    plt.close(fig_phase)

        with st.container(border=True):
            if design_ok:
                st.markdown("---")
                st.subheader("Filtering Execution and Verification")

                try:
                    filtered_fir = fir_mod.apply_fir_filter(sig, fir_design, mode=mode_str)
                    apply_ok = True
                except Exception as e:
                    st.error(f"Filtering execution failed: {e}")
                    apply_ok = False

                if apply_ok:
                    col_aud, col_met = st.columns([1.0, 1.2])

                    with col_aud:
                        st.markdown("**FIR Filtered Audio:**")
                        render_audio_player(filtered_fir, sr, f"FIR Filtered ({mode_str})")
                        render_download_button(
                            filtered_fir, sr, f"fir_filtered_{f_type}_{mode_str}.wav", "Download Filtered WAV"
                        )

                    with col_met:
                        if ref is not None:
                            st.markdown("#### Objective Performance Metrics (Workflow A)")
                            if mode_str == "causal":
                                M = fir_design.numtaps
                                tau = (M - 1) // 2
                                N = len(ref)
                                if N > 2 * tau:
                                    eval_ref = ref[tau : N - tau]
                                    eval_noi = sig[tau : N - tau]
                                    eval_fil = filtered_fir[2 * tau : N]
                                    eval_res = metrics_mod.evaluate_audio_quality(eval_ref, eval_noi, eval_fil)
                                    st.markdown(
                                        f"- **Evaluation Interval:** Valid middle interval [tau : N - tau] (`{len(eval_ref):,}` samples)\n"
                                        f"- **Input SNR:** `{eval_res.input_snr_db:.2f} dB` &nbsp;→&nbsp; **Output SNR:** `{eval_res.output_snr_db:.2f} dB`\n"
                                        f"- **SNR Improvement (Delta SNR):** `{eval_res.snr_improvement_db:+.2f} dB`\n"
                                        f"- **Input RMSE:** `{eval_res.input_rmse:.4f}` &nbsp;→&nbsp; **Output RMSE:** `{eval_res.output_rmse:.4f}`\n"
                                        f"- **RMSE Reduction:** `{eval_res.rmse_reduction_percent:.2f}%`\n"
                                        f"- **MAE:** `{eval_res.mae:.4f}`"
                                    )
                                else:
                                    st.warning("Signal length is too short to evaluate causal FIR valid interval.")
                            else:
                                eval_res = metrics_mod.evaluate_audio_quality(ref, sig, filtered_fir)
                                st.markdown(
                                    f"- **Evaluation Mode:** Full signal zero-phase alignment (`{len(ref):,}` samples)\n"
                                    f"- **Input SNR:** `{eval_res.input_snr_db:.2f} dB` &nbsp;→&nbsp; **Output SNR:** `{eval_res.output_snr_db:.2f} dB`\n"
                                    f"- **SNR Improvement (Delta SNR):** `{eval_res.snr_improvement_db:+.2f} dB`\n"
                                    f"- **Input RMSE:** `{eval_res.input_rmse:.4f}` &nbsp;→&nbsp; **Output RMSE:** `{eval_res.output_rmse:.4f}`\n"
                                    f"- **RMSE Reduction:** `{eval_res.rmse_reduction_percent:.2f}%`\n"
                                    f"- **MAE:** `{eval_res.mae:.4f}`"
                                )
                        else:
                            st.info(
                                "ℹ️ **No-Reference Mode Active (Workflow B):** Objective ground-truth error metrics "
                                "(SNR / RMSE / MAE) are not computed because no clean reference was provided. "
                                "Inspect frequency spectrum and spectrogram below."
                            )
                    if ref is not None and mode_str == "zero_phase":
                        fig_comp = vis_mod.plot_signal_comparison(
                            ref, sig, filtered_fir, sr, title="Time-Domain Comparison: Clean vs. Noisy vs. FIR Filtered"
                        )
                        st.pyplot(fig_comp)
                        plt.close(fig_comp)

# =============================================================================
# WORKSPACE 4: IIR Filter Designer
# =============================================================================
elif workspace.startswith("4"):
    st.markdown('<div class="main-title">4. IIR Digital Filter Designer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Design high-efficiency Butterworth & Notch IIR filters in Second-Order Sections (SOS).</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.noisy_audio is None:
        st.error("No working audio signal available. Ingest audio in Workspace 1.")
    else:
        sig = st.session_state.noisy_audio
        ref = st.session_state.reference_audio
        sr = st.session_state.sample_rate
        nyq = sr / 2.0

        with st.container(border=True):
            col_cfg, col_res = st.columns([1.1, 1.2])

            with col_cfg:
                st.subheader("IIR Design Parameters")
                design_category = st.radio("Select IIR Family:", ["Butterworth Standard Filter", "IIR Notch Filter (Mains Hum Rejection)"])

                if design_category.startswith("Butterworth"):
                    f_type = st.selectbox("Filter Topology:", ["lowpass", "highpass", "bandpass", "bandstop"], index=0)
                    order = st.slider("Filter Order N:", min_value=1, max_value=10, value=4, step=1)

                    if f_type in ["lowpass", "highpass"]:
                        cutoff = st.slider(
                            "Cutoff Frequency fc (Hz):",
                            min_value=20.0,
                            max_value=float(nyq - 50.0),
                            value=min(1500.0, nyq * 0.4),
                            step=20.0,
                        )
                    else:
                        c1, c2 = st.columns(2)
                        with c1:
                            f_low = st.number_input("Lower Cutoff (Hz):", min_value=20.0, max_value=float(nyq - 100.0), value=300.0, step=20.0)
                        with c2:
                            f_high = st.number_input("Upper Cutoff (Hz):", min_value=float(f_low + 20.0), max_value=float(nyq - 50.0), value=3000.0, step=20.0)
                        cutoff = [f_low, f_high]

                    mode_iir = st.radio(
                        "Filtering Mode:",
                        [
                            "zero_phase (sosfiltfilt: offline non-causal forward-backward, zero phase shift)",
                            "causal (sosfilt: real-time causal, minimum latency, non-linear phase response)",
                        ],
                        index=0,
                    )
                    mode_str = "zero_phase" if "zero_phase" in mode_iir else "causal"

                    try:
                        iir_design = iir_mod.design_iir_filter(sample_rate=sr, filter_type=f_type, cutoff_hz=cutoff, order=order)
                        design_ok = True
                    except Exception as e:
                        st.error(f"IIR Design Error: {e}")
                        design_ok = False

                    convention_text = "Butterworth -3.01 dB (|H|=1/sqrt(2))"

                else:
                    f_type = "notch"
                    notch_f = st.slider("Notch Center Frequency (Hz):", min_value=20.0, max_value=500.0, value=50.0, step=1.0)
                    q_factor = st.slider("Quality Factor Q:", min_value=5.0, max_value=100.0, value=30.0, step=5.0)

                    mode_iir = st.radio(
                        "Filtering Mode:",
                        [
                            "zero_phase (sosfiltfilt: offline non-causal forward-backward, zero phase shift)",
                            "causal (sosfilt: real-time causal, non-linear phase response)",
                        ],
                        index=0,
                    )
                    mode_str = "zero_phase" if "zero_phase" in mode_iir else "causal"

                    try:
                        iir_design = iir_mod.design_iir_notch(sample_rate=sr, notch_freq=notch_f, quality_factor=q_factor)
                        cutoff = notch_f
                        design_ok = True
                    except Exception as e:
                        st.error(f"Notch Design Error: {e}")
                        design_ok = False

                    convention_text = f"Narrowband notch rejection null at f0 = {notch_f:.1f} Hz (Q = {q_factor:.1f})"

                if design_ok:
                    is_stable, max_pole = iir_mod.check_sos_stability(iir_design.sos)
                    st.markdown(
                        f"- **SOS Sections:** `{iir_design.sos.shape[0]}`\n"
                        f"- **BIBO Stability:** `{'STABLE' if is_stable else 'UNSTABLE'}` (Max pole |z| = `{max_pole:.5f}`)\n"
                        f"- **Design Target:** `{convention_text}`"
                    )

            with col_res:
                st.subheader("Frequency and Phase Response")
                if design_ok:
                    w_f, h_mag, h_db, phi = iir_mod.get_iir_frequency_response(iir_design, worN=2048)

                    fig_resp = vis_mod.plot_filter_response(
                        w_f, h_db, title=f"IIR {f_type.capitalize()} Magnitude Response (SOS)", cutoff_hz=cutoff
                    )
                    st.pyplot(fig_resp)
                    plt.close(fig_resp)

                    fig_phase = vis_mod.plot_phase_response(
                        w_f, phi, title=f"IIR {f_type.capitalize()} Phase Response (Non-linear Phase)"
                    )
                    st.pyplot(fig_phase)
                    plt.close(fig_phase)

        with st.container(border=True):
            if design_ok:
                st.markdown("---")
                st.subheader("Filtering Execution and Verification")

                try:
                    filtered_iir = iir_mod.apply_iir_filter(sig, iir_design, mode=mode_str)
                    apply_ok = True
                except Exception as e:
                    st.error(f"Filtering execution failed: {e}")
                    apply_ok = False

                if apply_ok:
                    col_aud, col_met = st.columns([1.0, 1.2])

                    with col_aud:
                        st.markdown("**IIR Filtered Audio:**")
                        render_audio_player(filtered_iir, sr, f"IIR Filtered ({mode_str})")
                        render_download_button(
                            filtered_iir, sr, f"iir_filtered_{f_type}_{mode_str}.wav", "Download Filtered WAV"
                        )

                    with col_met:
                        if ref is not None:
                            st.markdown("#### Objective Performance Metrics (Workflow A)")
                            eval_res = metrics_mod.evaluate_audio_quality(ref, sig, filtered_iir)
                            st.markdown(
                                f"- **Input SNR:** `{eval_res.input_snr_db:.2f} dB` &nbsp;→&nbsp; **Output SNR:** `{eval_res.output_snr_db:.2f} dB`\n"
                                f"- **SNR Improvement (Delta SNR):** `{eval_res.snr_improvement_db:+.2f} dB`\n"
                                f"- **Input RMSE:** `{eval_res.input_rmse:.4f}` &nbsp;→&nbsp; **Output RMSE:** `{eval_res.output_rmse:.4f}`\n"
                                f"- **RMSE Reduction:** `{eval_res.rmse_reduction_percent:.2f}%`\n"
                                f"- **MAE:** `{eval_res.mae:.4f}`"
                            )
                        else:
                            st.info(
                                "ℹ️ **No-Reference Mode Active (Workflow B):** Objective ground-truth error metrics "
                                "are suppressed because no clean reference was provided."
                            )

                    if ref is not None and mode_str == "zero_phase":
                        fig_comp = vis_mod.plot_signal_comparison(
                            ref, sig, filtered_iir, sr, title="Time-Domain Comparison: Clean vs. Noisy vs. IIR Filtered"
                        )
                        st.pyplot(fig_comp)
                        plt.close(fig_comp)

# =============================================================================
# WORKSPACE 5: FIR vs. IIR Comparative Evaluation
# =============================================================================
elif workspace.startswith("5"):
    st.markdown('<div class="main-title">5. FIR vs. IIR Comparative Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Rigorous head-to-head architectural comparison between FIR and IIR digital filter implementations.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.noisy_audio is None:
        st.error("No working audio signal available. Ingest audio in Workspace 1.")
    else:
        sig = st.session_state.noisy_audio
        ref = st.session_state.reference_audio
        sr = st.session_state.sample_rate
        nyq = sr / 2.0

        with st.container(border=True):
            st.subheader("Matched Filter Specifications")
            c_top, c_mode, c_cut = st.columns([1, 1, 1.2])

            with c_top:
                comp_type = st.selectbox("Filter Topology:", ["lowpass", "highpass", "bandpass", "bandstop"], index=0)
            with c_mode:
                comp_mode = st.radio("Comparison Mode:", ["zero_phase", "causal"], index=0)
            with c_cut:
                if comp_type in ["lowpass", "highpass"]:
                    comp_cutoff = st.slider("Cutoff Frequency fc (Hz):", 20.0, float(nyq - 50.0), min(1500.0, nyq * 0.4), 20.0)
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        cf_low = st.number_input("Low Cutoff (Hz):", 20.0, float(nyq - 100.0), 300.0, 20.0)
                    with c2:
                        cf_high = st.number_input("High Cutoff (Hz):", float(cf_low + 20.0), float(nyq - 50.0), 3000.0, 20.0)
                    comp_cutoff = [cf_low, cf_high]

            c_fir, c_iir = st.columns(2)
            with c_fir:
                st.markdown("**FIR Architecture:**")
                fir_taps = st.slider("FIR Length (Taps M - odd):", 11, 401, 101, 2)
                fir_win = st.selectbox("Window Function:", ["hamming", "hann", "blackman"], index=0)

            with c_iir:
                st.markdown("**IIR Architecture:**")
                iir_ord = st.slider("Butterworth Order N:", 1, 8, 4, 1)
                st.caption("Implemented via Second-Order Sections (SOS).")

            st.info(
                "ℹ️ **Engineering Equivalence Note:** Windowed FIR specifies the **-6.02 dB** half-amplitude point, "
                "whereas Butterworth IIR specifies the **-3.01 dB** half-power point. Both filters share identical cutoff parameters "
                "within engineering tolerances."
            )

            run_comp = st.button("Run Head-to-Head Comparative Benchmark", type="primary", use_container_width=True)

            if run_comp:
                try:
                    fir_des = fir_mod.design_fir_filter(
                        sample_rate=sr, filter_type=comp_type, cutoff_hz=comp_cutoff, numtaps=fir_taps, window=fir_win
                    )
                    iir_des = iir_mod.design_iir_filter(
                        sample_rate=sr, filter_type=comp_type, cutoff_hz=comp_cutoff, order=iir_ord
                    )
                except Exception as e:
                    st.error(f"Filter design error: {e}")
                    fir_des, iir_des = None, None

                if fir_des and iir_des:
                    # WORKFLOW A: With Ground Truth Reference
                    if ref is not None:
                        try:
                            with st.spinner("Executing rigorous comparative benchmark..."):
                                report, out_fir, out_iir = comp.compare_fir_and_iir(
                                    reference=ref,
                                    noisy_signal=sig,
                                    fir_design=fir_des,
                                    iir_design=iir_des,
                                    sample_rate=sr,
                                    mode=comp_mode,
                                    timing_trials=25,
                                    timing_warmup=5,
                                )
                            comp_success = True
                        except Exception as e:
                            st.error(f"Comparative execution failed: {e}")
                            comp_success = False

                        if comp_success:
                            st.markdown("### 📊 Benchmark Results Table")

                            speedup_str = format_speedup(report.speedup_ratio)
                            summary_md = f"""
    | Metric / Parameter | FIR ({fir_win.capitalize()}, M={fir_taps}) | IIR (Butterworth, N={iir_ord}) | Delta / Comparison |
    | :--- | :--- | :--- | :--- |
    | **Filtering Mode** | `{report.mode}` | `{report.mode}` | Matched |
    | **Evaluation Interval** | `{report.evaluation_interval}` | `{report.evaluation_interval}` | Valid Sample Alignment |
    | **Output SNR (dB)** | **{report.fir_metrics.output_snr_db:.2f} dB** | **{report.iir_metrics.output_snr_db:.2f} dB** | {report.fir_metrics.output_snr_db - report.iir_metrics.output_snr_db:+.2f} dB |
    | **SNR Improvement (Delta SNR)** | **{report.fir_metrics.snr_improvement_db:+.2f} dB** | **{report.iir_metrics.snr_improvement_db:+.2f} dB** | {report.fir_metrics.snr_improvement_db - report.iir_metrics.snr_improvement_db:+.2f} dB |
    | **Output RMSE** | `{report.fir_metrics.output_rmse:.4f}` | `{report.iir_metrics.output_rmse:.4f}` | {report.fir_metrics.output_rmse - report.iir_metrics.output_rmse:+.4f} |
    | **RMSE Reduction (%)** | **{report.fir_metrics.rmse_reduction_percent:.2f}%** | **{report.iir_metrics.rmse_reduction_percent:.2f}%** | {report.fir_metrics.rmse_reduction_percent - report.iir_metrics.rmse_reduction_percent:+.2f}% |
    | **Mean Absolute Error (MAE)** | `{report.fir_metrics.mae:.4f}` | `{report.iir_metrics.mae:.4f}` | {report.fir_metrics.mae - report.iir_metrics.mae:+.4f} |
    | **Median Execution Time** | `{report.fir_metrics.execution_time_ms:.3f} ms` | `{report.iir_metrics.execution_time_ms:.3f} ms` | **Speedup Ratio: {speedup_str}** |
    | **Filter Stability** | Unconditionally Stable | BIBO Stable (SOS) | Verified |
    """
                            st.markdown(summary_md)

                            p_col1, p_col2 = st.columns(2)
                            with p_col1:
                                st.markdown("**FIR Filtered Audio:**")
                                render_audio_player(out_fir, sr, "FIR Output")
                                render_download_button(out_fir, sr, f"fir_comp_{comp_type}.wav", "Download FIR Audio")
                            with p_col2:
                                st.markdown("**IIR Filtered Audio:**")
                                render_audio_player(out_iir, sr, "IIR Output")
                                render_download_button(out_iir, sr, f"iir_comp_{comp_type}.wav", "Download IIR Audio")

                    # WORKFLOW B: Without Ground Truth Reference
                    else:
                        st.info("🔍 **Running Independent No-Reference Comparative Analysis (Workflow B)**")
                        with st.spinner("Processing filters independently..."):
                            t_fir = metrics_mod.measure_processing_time(fir_mod.apply_fir_filter, sig, fir_des, mode=comp_mode)
                            t_iir = metrics_mod.measure_processing_time(iir_mod.apply_iir_filter, sig, iir_des, mode=comp_mode)
                            out_fir = fir_mod.apply_fir_filter(sig, fir_des, mode=comp_mode)
                            out_iir = iir_mod.apply_iir_filter(sig, iir_des, mode=comp_mode)

                            fir_ms = t_fir["median_ms"]
                            iir_ms = t_iir["median_ms"]
                            speedup_val = fir_ms / iir_ms if iir_ms > 0 else None
                            speedup_str = format_speedup(speedup_val)

                            phase_note_fir = "Linear Phase (Group delay tau = (M-1)/2)" if comp_mode == "causal" else "Zero Phase (Offline forward-backward)"
                            phase_note_iir = "Non-linear Phase (Frequency-dependent group delay)" if comp_mode == "causal" else "Zero Phase (Offline forward-backward)"

                        summary_md = f"""
    | Parameter | FIR ({fir_win.capitalize()}, M={fir_taps}) | IIR (Butterworth, N={iir_ord}) | Comparison |
    | :--- | :--- | :--- | :--- |
    | **Filtering Mode** | `{comp_mode}` | `{comp_mode}` | Matched |
    | **Ground Truth Reference** | Not Available | Not Available | Reference-based error metrics suppressed |
    | **Filter Topology** | `{comp_type.capitalize()}` | `{comp_type.capitalize()}` | Matched |
    | **Median Execution Time** | `{fir_ms:.3f} ms` | `{iir_ms:.3f} ms` | **Speedup Ratio: {speedup_str}** |
    | **Phase Characteristics** | `{phase_note_fir}` | `{phase_note_iir}` | Structural Property |
    | **Stability** | Unconditionally Stable | Stable (SOS form) | Verified |
    """
                        st.markdown(summary_md)

                        p_col1, p_col2 = st.columns(2)
                        with p_col1:
                            st.markdown("**FIR Filtered Audio:**")
                            render_audio_player(out_fir, sr, "FIR Output")
                            render_download_button(out_fir, sr, f"fir_comp_noref_{comp_type}.wav", "Download FIR Audio")
                        with p_col2:
                            st.markdown("**IIR Filtered Audio:**")
                            render_audio_player(out_iir, sr, "IIR Output")
                            render_download_button(out_iir, sr, f"iir_comp_noref_{comp_type}.wav", "Download IIR Audio")

                    # Frequency and Phase Overlay Curves (Valid for both workflows)
                    st.markdown("### 📈 Comparative Filter Response Curves")
                    wf_fir, _, h_db_fir, phi_fir = fir_mod.get_fir_frequency_response(fir_des, worN=2048)
                    wf_iir, _, h_db_iir, phi_iir = iir_mod.get_iir_frequency_response(iir_des, worN=2048)

                    fig, (ax1, ax2) = plt.subplots(
                        1, 2, figsize=(14, 4.8), facecolor="#FFFFFF",
                        gridspec_kw={"width_ratios": [1.15, 1.0]}
                    )
                    for axis in (ax1, ax2):
                        vis_mod._style_axes(axis)

                    fir_color, iir_color = "#2563EB", "#D97706"
                    display_limit = min(sr / 2.0, 8000.0)

                    ax1.plot(
                        wf_fir, h_db_fir, label=f"FIR · {fir_taps} taps",
                        color=fir_color, linewidth=2.0, zorder=3
                    )
                    ax1.plot(
                        wf_iir, h_db_iir, label=f"IIR · order {iir_ord}",
                        color=iir_color, linewidth=2.0, linestyle="--", zorder=3
                    )
                    ax1.axhline(
                        -3.0103, color="#94A3B8", linewidth=1.0,
                        linestyle=":", label="−3.01 dB reference"
                    )
                    ax1.axhline(
                        -6.0206, color="#CBD5E1", linewidth=1.0,
                        linestyle="-.", label="−6.02 dB reference"
                    )
                    cutoff_values = comp_cutoff if isinstance(comp_cutoff, (list, tuple)) else [comp_cutoff]
                    for cutoff_value in cutoff_values:
                        ax1.axvline(
                            float(cutoff_value), color="#64748B", linewidth=1.0,
                            linestyle=":", alpha=0.8
                        )
                    ax1.set_title("Magnitude response", loc="left", fontsize=12, fontweight="bold", color="#14243B", pad=12)
                    ax1.set_xlabel("Frequency (Hz)")
                    ax1.set_ylabel("Magnitude (dB)")
                    ax1.set_ylim(-80, 5)
                    ax1.set_xlim(0, display_limit)
                    ax1.legend(loc="lower right", fontsize=8, ncol=2, frameon=True,
                               facecolor="#FFFFFF", edgecolor="#DCE4EF", framealpha=0.96)

                    ax2.plot(
                        wf_fir, phi_fir, label="FIR phase",
                        color=fir_color, linewidth=1.9
                    )
                    ax2.plot(
                        wf_iir, phi_iir, label="IIR phase",
                        color=iir_color, linewidth=1.9, linestyle="--"
                    )
                    ax2.set_title("Phase response", loc="left", fontsize=12, fontweight="bold", color="#14243B", pad=12)
                    ax2.set_xlabel("Frequency (Hz)")
                    ax2.set_ylabel("Phase (radians)")
                    ax2.set_xlim(0, display_limit)
                    ax2.legend(loc="best", fontsize=8, frameon=True,
                               facecolor="#FFFFFF", edgecolor="#DCE4EF", framealpha=0.96)

                    fig.suptitle(
                        f"{comp_type.capitalize()} filter architecture comparison",
                        fontsize=13, fontweight="bold", color="#14243B",
                        x=0.06, ha="left"
                    )
                    fig.tight_layout(rect=(0, 0, 1, 0.94))
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

# =============================================================================
# WORKSPACE 6: Signal Analysis and Spectral Diagnostics
# =============================================================================
elif workspace.startswith("6"):
    st.markdown('<div class="main-title">6. Signal Analysis and Spectral Diagnostics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Inspect high-resolution FFT spectra and STFT spectrograms with calibrated units.</div>',
        unsafe_allow_html=True,
    )

    cur_sig = (
        st.session_state.noisy_audio
        if st.session_state.noisy_audio is not None
        else st.session_state.audio_data
    )
    sr = st.session_state.sample_rate

    if cur_sig is None:
        st.error("No audio signal loaded.")
    else:
        sig_len = len(cur_sig) if cur_sig.ndim == 1 else len(cur_sig[:, 0])

        with st.container(border=True):
            st.subheader("Fast Fourier Transform (FFT) Magnitude Spectrum")
            c_fft1, c_fft2 = st.columns([1, 2])

            with c_fft1:
                fft_win = st.selectbox("FFT Window Function:", ["hann", "hamming", "blackman", "boxcar"], index=0)
                db_scale = st.checkbox("Display in Decibels (dBFS)", value=True)
                max_plot_f = st.slider("Max Display Frequency (Hz):", 500, sr // 2, min(sr // 2, 10000), 500)

            with c_fft2:
                freqs, mag_fft, _ = fft_mod.compute_fft(cur_sig, sr, window=fft_win)
                fig_fft = vis_mod.plot_fft_spectrum(
                    frequencies=freqs, magnitude=mag_fft, db_scale=db_scale, max_freq=float(max_plot_f)
                )
                st.pyplot(fig_fft)
                plt.close(fig_fft)

                top_indices = np.argsort(mag_fft)[::-1][:5]
                peaks_info = [f"{freqs[idx]:.1f} Hz ({20*np.log10(max(mag_fft[idx], 1e-12)):.1f} dBFS)" for idx in top_indices if freqs[idx] > 10]
                st.caption(f"**Dominant Spectral Peaks:** {', '.join(peaks_info)}")

            st.markdown("---")
        with st.container(border=True):
            st.subheader("Short-Time Fourier Transform (STFT) Spectrogram")
            st.caption("Accurate calibrated units: dBFS/Hz for power spectral density, dBFS for spectrum.")

            # Adapt nperseg options based on recording length
            standard_segments = [256, 512, 1024, 2048, 4096]
            valid_segments = [s for s in standard_segments if s <= sig_len]

            if not valid_segments:            st.warning(
                    f"Signal length ({sig_len} samples) is too short for standard STFT analysis (minimum segment length is 256 samples). "
                    "Spectrogram computation requires longer recordings."
                )
            else:
                default_val = 1024 if 1024 in valid_segments else valid_segments[-1]

                c_stft1, c_stft2 = st.columns([1, 2])

                with c_stft1:
                    nperseg_val = st.select_slider("Segment Length (nperseg):", options=valid_segments, value=default_val)
                    scaling_mode = st.radio("Spectrogram Scaling:", ["density (dBFS/Hz)", "spectrum (dBFS)"], index=0)
                    sc_val = "density" if "density" in scaling_mode else "spectrum"
                    stft_max_f = st.slider("Spectrogram Max Frequency (Hz):", 500, sr // 2, min(sr // 2, 10000), 500)

                with c_stft2:
                    try:
                        t_spec, f_spec, sxx_db = fft_mod.compute_spectrogram(
                            cur_sig, sr, nperseg=nperseg_val, scaling=sc_val
                        )
                        unit_label = "dBFS/Hz" if sc_val == "density" else "dBFS"
                        fig_spec = vis_mod.plot_spectrogram(
                            times=t_spec,
                            frequencies=f_spec,
                            Sxx_db=sxx_db,
                            title=f"Audio Spectrogram (STFT, Scaling: {unit_label})",
                            max_freq=float(stft_max_f),
                        )
                        st.pyplot(fig_spec)
                        plt.close(fig_spec)
                    except Exception as e:
                        st.error(f"Spectrogram computation error: {e}")