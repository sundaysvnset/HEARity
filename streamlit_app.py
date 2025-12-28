import os
import re
import tempfile
import subprocess
import streamlit as st
import torch
import librosa
import numpy as np

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from google import genai

# =========================
# CONFIG
# =========================
MODEL_ID = "jovangelo/whispermodelproyek"
LANG = "id"
DEVICE = "cpu"

SAMPLE_RATE = 16000
CHUNK_DURATION = 20        # seconds (AMAN & CEPAT)
MAX_NEW_TOKENS = 200       # JANGAN > 200

# =========================
# LOAD MODELS
# =========================
@st.cache_resource(show_spinner="📦 Memuat model Whisper...")
def load_whisper():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="📦 Memuat Gemini...")
def load_gemini():
    if "GEMINI_API_KEY" not in st.secrets:
        raise RuntimeError("GEMINI_API_KEY belum diset di Secrets")
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


processor, whisper_model = load_whisper()

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
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", "16000",
            out_path
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return out_path


# =========================
# AUDIO CHUNKING
# =========================
def split_audio(audio, sr=SAMPLE_RATE, chunk_duration=CHUNK_DURATION):
    chunk_size = int(chunk_duration * sr)
    return [
        audio[i:i + chunk_size]
        for i in range(0, len(audio), chunk_size)
    ]


# =========================
# TRANSCRIPT AUTO CLEAN
# =========================
def clean_transcript(text: str) -> str:
    text = text.lower()

    # hapus filler umum
    fillers = [
        r"\buh+\b", r"\bum+\b", r"\beh+\b", r"\bhmm+\b",
        r"\banu\b", r"\bgitu\b", r"\bjadi\b"
    ]
    for f in fillers:
        text = re.sub(f, "", text)

    # hapus pengulangan kata berturut-turut
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)

    # rapikan spasi
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================
# WHISPER TRANSCRIPTION
# =========================
def whisper_transcribe(wav_path):
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    chunks = split_audio(audio)

    forced_ids = processor.get_decoder_prompt_ids(
        language=LANG,
        task="transcribe"
    )

    results = []

    for chunk in chunks:
        if len(chunk) < SAMPLE_RATE * 2:
            continue

        inputs = processor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        with torch.no_grad():
            pred_ids = whisper_model.generate(
                input_features=inputs.input_features.to(DEVICE),
                forced_decoder_ids=forced_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        text = processor.batch_decode(
            pred_ids,
            skip_special_tokens=True
        )[0]

        results.append(text)

    raw_text = " ".join(results)
    return clean_transcript(raw_text)


# =========================
# GEMINI SUMMARIZATION
# =========================
def gemini_summarize(full_text):
    client = load_gemini()

    prompt = f"""
Ringkas materi berikut dalam bentuk BULLET POINTS yang komprehensif.

Aturan:
- Tangkap alur pembahasan dari awal sampai akhir
- Sertakan konsep utama dan penjelasan penting
- Gunakan Bahasa Indonesia akademik & jelas
- Jangan menyalin kalimat mentah
- Maksimal 10–12 bullet points

MATERI:
{full_text}
"""

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()


# =========================
# UI
# =========================
st.set_page_config(
    page_title="HEARity",
    page_icon="🎧",
    layout="centered"
)

st.title("🎧 HEARity")
st.caption("Speech-to-Text & Automatic Summarization berbasis Whisper + Generative AI")

st.markdown("""
HEARity membantu penyandang gangguan pendengaran untuk:
- Mengubah audio/video menjadi teks
- Menghasilkan ringkasan otomatis yang mudah dipahami

🎓 **Final Project – Biomedical Engineering**
""")

st.divider()

# INPUT
uploaded_file = st.file_uploader(
    "📥 Unggah file audio / video",
    type=["mp3", "wav", "mp4", "mkv", "m4a"]
)

# SESSION STATE
st.session_state.setdefault("transcript", "")
st.session_state.setdefault("summary", "")

# BUTTON
if st.button("🚀 Proses", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.warning("⚠️ Silakan unggah file terlebih dahulu.")
        st.stop()

    try:
        with st.spinner("🎵 Menyiapkan audio..."):
            input_path = save_upload_to_tmp(uploaded_file)
            wav_path = run_ffmpeg_to_wav16k(input_path)

        with st.spinner("🧠 Transkripsi dengan Whisper..."):
            transcript = whisper_transcribe(wav_path)

        with st.spinner("✍️ Membuat ringkasan..."):
            summary = gemini_summarize(transcript)

        st.session_state.transcript = transcript
        st.session_state.summary = summary

        st.success("✅ Proses selesai!")

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.divider()

# OUTPUT
st.subheader("📄 Transkrip")
st.text_area(
    "Hasil Transkrip",
    st.session_state.transcript,
    height=220,
    key="transcript_box"
)

st.subheader("📝 Ringkasan")
st.text_area(
    "Ringkasan Otomatis",
    st.session_state.summary,
    height=220,
    key="summary_box"
)

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        "⬇️ Unduh Transkrip",
        st.session_state.transcript,
        "transcript.txt",
        mime="text/plain",
        use_container_width=True
    )

with col2:
    st.download_button(
        "⬇️ Unduh Ringkasan",
        st.session_state.summary,
        "summary.txt",
        mime="text/plain",
        use_container_width=True
    )
