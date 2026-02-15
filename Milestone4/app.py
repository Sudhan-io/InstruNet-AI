import os
import warnings
import json
from fpdf import FPDF

warnings.filterwarnings("ignore")

# ==================================================
# SILENCE TENSORFLOW NOISE
# ==================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import tensorflow as tf
import numpy as np
import tempfile
import librosa
import librosa.display
import matplotlib.pyplot as plt

from preprocess import preprocess_audio
from multidetect import predict_segments

# ==================================================
# SESSION STATE
# ==================================================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ==================================================
# LABEL DEFINITIONS
# ==================================================
LABELS = [
    "brass", "flute", "guitar", "keyboard",
    "mallet", "reed", "string", "vocal"
]

CONDITION_LABELS = ["Healthy", "Aged", "Broken"]

PRESENCE_THRESHOLD = 0.15

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="InstruNet AI",
    page_icon="🎵",
    layout="centered"
)

# ==================================================
# HEADER
# ==================================================
st.markdown(
    """
    <h1 style="text-align:center;">🎶 InstruNet AI</h1>
    <p style="text-align:center; color: gray;">
    CNN-based Music Instrument Recognition & Condition Analysis
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ==================================================
# LOAD MODELS
# ==================================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_instrument_model():
    model_path = os.path.join(BASE_DIR, "instrunet_model_v3.keras")
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_condition_model():
    model_path = os.path.join(BASE_DIR, "instrunet_condition.keras")
    return tf.keras.models.load_model(model_path)
instrument_model = load_instrument_model()
condition_model = load_condition_model()
# ==================================================
# INSTRUMENT PREDICTION
# ==================================================
def predict_instrument(audio_path):
    x = preprocess_audio(audio_path)
    preds = instrument_model.predict(x, verbose=0)[0]
    idx = np.argmax(preds)
    return LABELS[idx], float(preds[idx]), preds

# ==================================================
# CONDITION PREDICTION
# ==================================================
def predict_condition(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    if len(y) < 16000 * 4:
        y = np.pad(y, (0, 16000*4 - len(y)))
    else:
        y = y[:16000*4]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=16000,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db + 80) / 80
    mel_norm = mel_norm[..., np.newaxis]
    mel_norm = np.expand_dims(mel_norm, axis=0)

    preds = condition_model.predict(mel_norm, verbose=0)[0]
    idx = np.argmax(preds)

    return CONDITION_LABELS[idx], float(preds[idx])

# ==================================================
# INTENSITY TEXT DISPLAY
# ==================================================
def show_text_intensity(instrument_summary):
    st.subheader("📊 Instrument Intensity")

    for inst in LABELS:
        score = instrument_summary.get(inst, 0.0)
        if score >= PRESENCE_THRESHOLD:
            bar = "█" * max(1, int(score * 10))
            st.markdown(f"**{inst.capitalize():<10}** : `{bar}` ({score:.2f})")

# ==================================================
# WAVEFORM TIMELINE
# ==================================================
def show_waveform_timeline(audio_path):
    st.subheader("📈 Instrument Timeline")

    y, sr = librosa.load(audio_path, sr=22050)

    fig, ax = plt.subplots(figsize=(8, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.85)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Audio Waveform Timeline")

    st.pyplot(fig)

# ==================================================
# MEL SPECTROGRAM
# ==================================================
def show_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax
    )

    ax.set_title("Mel-Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    st.pyplot(fig)

# ==================================================
# JSON EXPORT
# ==================================================
def export_json_report(filename, label, confidence, instrument_summary, segment_preds, condition):
    report = {
        "audio_file": filename,
        "instrument_prediction": {
            "instrument": label,
            "confidence": round(confidence, 4)
        },
        "condition_prediction": condition,
        "instrument_summary": instrument_summary,
        "segments": segment_preds
    }

    path = filename.replace(".wav", "_analysis.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=4)

    return path

# ==================================================
# PDF EXPORT
# ==================================================
def export_pdf_report(filename, label, confidence, instrument_summary, segment_preds, condition):
    from datetime import datetime

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "InstruNet AI - Analysis Report", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.now()}", ln=True)
    pdf.cell(0, 8, f"Audio File: {filename}", ln=True)

    pdf.ln(5)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Instrument Prediction", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Instrument: {label}", ln=True)
    pdf.cell(0, 8, f"Confidence: {confidence:.3f}", ln=True)

    pdf.ln(5)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Condition Classification", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Condition: {condition}", ln=True)

    path = filename.replace(".wav", "_analysis.pdf")
    pdf.output(path)

    return path

# ==================================================
# FILE UPLOADER
# ==================================================
uploaded_file = st.file_uploader("📂 Upload Audio File (.wav)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    if st.button("🎯 Analyze Track"):
        st.session_state.analysis_done = True

        label, confidence, preds = predict_instrument(temp_path)
        segment_preds, instrument_summary = predict_segments(temp_path)
        condition, cond_conf = predict_condition(temp_path)

        st.session_state.results = {
            "label": label,
            "confidence": confidence,
            "instrument_summary": instrument_summary,
            "segment_preds": segment_preds,
            "temp_path": temp_path,
            "condition": condition,
            "cond_conf": cond_conf
        }

# ==================================================
# RESULTS
# ==================================================
if st.session_state.analysis_done:
    r = st.session_state.results

    st.subheader("🔎 Final Prediction")
    st.success(f"🎵 Predicted Instrument: {r['label'].capitalize()}")
    st.progress(r["confidence"])
    st.caption(f"Confidence: {r['confidence']:.3f}")

    st.subheader("🧱 Instrument Condition")

    if r["condition"] == "Healthy":
        st.success(f"Condition: {r['condition']}")
    elif r["condition"] == "Aged":
        st.warning(f"Condition: {r['condition']}")
    else:
        st.error(f"Condition: {r['condition']}")

    st.caption(f"Confidence: {r['cond_conf']:.3f}")

    show_text_intensity(r["instrument_summary"])
    show_waveform_timeline(r["temp_path"])

    st.subheader("🎼 Audio Representation")
    show_spectrogram(r["temp_path"])

    json_path = export_json_report(
        os.path.basename(r["temp_path"]),
        r["label"],
        r["confidence"],
        r["instrument_summary"],
        r["segment_preds"],
        r["condition"]
    )

    pdf_path = export_pdf_report(
        os.path.basename(r["temp_path"]),
        r["label"],
        r["confidence"],
        r["instrument_summary"],
        r["segment_preds"],
        r["condition"]
    )

    with open(json_path, "rb") as f:
        st.download_button("⬇️ Download JSON Report", f, file_name=os.path.basename(json_path))

    with open(pdf_path, "rb") as f:
        st.download_button("⬇️ Download PDF Report", f, file_name=os.path.basename(pdf_path))

# ==================================================
# FOOTER
# ==================================================
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:12px; color:gray;">
    InstruNet AI • CNN-based Musical Instrument Recognition & Condition Analysis
    </p>
    """,
    unsafe_allow_html=True
)