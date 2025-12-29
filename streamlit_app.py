import os
import re
import tempfile
import subprocess
import streamlit as st
import torch
import librosa

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from google import genai

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# =========================
# CONFIG
# =========================
MODEL_ID = "openai/whisper-medium"
LANG = "id"
DEVICE = "cpu"
SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 448

# =========================
# LOAD MODELS (CACHED)
# =========================
@st.cache_resource(show_spinner="📦 Memuat model Whisper Medium...")
def load_whisper():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="📦 Memuat Gemini...")
def load_gemini():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


processor, whisper_model = load_whisper()
gemini_client = load_gemini()

# =========================
# AUDIO HELPERS
# =========================
def save_upload_to_tmp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def run_ffmpeg_to_wav16k(input_path):
    out_path = input_path + "_16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", out_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return out_path

# =========================
# WHISPER TRANSCRIPTION
# =========================
def whisper_transcribe(wav_path):
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    forced_ids = processor.get_decoder_prompt_ids(
        language=LANG,
        task="transcribe"
    )

    with torch.no_grad():
        pred_ids = whisper_model.generate(
            input_features=inputs.input_features.to(DEVICE),
            attention_mask=inputs.attention_mask.to(DEVICE),
            forced_decoder_ids=forced_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    text = processor.batch_decode(
        pred_ids,
        skip_special_tokens=True
    )[0]

    # autoclean ringan
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# GEMINI SUMMARIZATION
# =========================
def gemini_summarize(full_text):
    prompt = f"""
Ringkas materi berikut dalam bentuk BULLET POINTS yang komprehensif.

Aturan:
- Ikuti alur pembahasan dari awal sampai akhir
- Sertakan semua konsep dan topik penting
- Jelaskan definisi, klasifikasi, dan perbedaan konsep utama jika ada
- Gunakan Bahasa Indonesia akademik
- Maksimal 10–12 bullet points
- Jangan menyalin kalimat mentah

MATERI:
{full_text}
"""

    response = gemini_client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()

# =====================
# HEADER (UI DIPERTAHANKAN)
# =====================
st.markdown(
    """
    <div style="padding:30px 10px;">
        <h1>🎧 HEARity</h1>
        <p style="color:#b0b0b0;">
             Konversi Suara ke Teks & Ringkasan Otomatis dengan Whisper + AI Generatif
        </p>
        <ul>
            <li>Mengubah file audio atau video menjadi teks tertulis</li>
            <li>Membuat ringkasan materi pembelajaran secara otomatis</li>
        </ul>
        <b>Kelompok 8 – Proyek Akhir</b>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================
# SESSION STATE
# =====================
for key in ["selesai", "teks_lengkap", "ringkasan"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key != "selesai" else False

# =====================
# UPLOADER
# =====================
file_diunggah = st.file_uploader(
    "📤 Unggah file audio atau video",
    type=["wav", "mp3", "mp4", "m4a", "mkv"]
)

if file_diunggah:
    if st.button("Mulai Proses", use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file_diunggah.read())
            input_path = tmp.name

        wav_path = run_ffmpeg_to_wav16k(input_path)

        with st.spinner("🔊 Mengubah suara menjadi teks..."):
            st.session_state.teks_lengkap = whisper_transcribe(wav_path)

        with st.spinner("✍🏻 Membuat ringkasan..."):
            st.session_state.ringkasan = gemini_summarize(
                st.session_state.teks_lengkap
            )

        st.session_state.selesai = True

# =====================
# OUTPUT
# =====================
if st.session_state.selesai:
    st.subheader("📄 Transkrip Lengkap")
    st.text_area("", st.session_state.teks_lengkap, height=260)

    st.subheader("📝 Ringkasan Materi")

    ringkasan_rapi = []
    for baris in st.session_state.ringkasan.split("\n"):
        if baris.strip().startswith("*"):
            ringkasan_rapi.append("• " + baris.lstrip("* ").strip())

    st.text_area("", "\n".join(ringkasan_rapi), height=260)
