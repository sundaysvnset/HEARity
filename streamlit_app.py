import os
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
CHUNK_DURATION = 30  # seconds
MAX_NEW_TOKENS = 300

# =========================
# LOAD MODELS (CACHED)
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


def download_youtube_audio(url):
    tmpdir = tempfile.mkdtemp()
    outtmpl = os.path.join(tmpdir, "audio.%(ext)s")

    subprocess.run(
        ["yt-dlp", "-f", "bestaudio", "-o", outtmpl, url],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    filename = os.listdir(tmpdir)[0]
    return run_ffmpeg_to_wav16k(os.path.join(tmpdir, filename))

# =========================
# AUDIO CHUNKING
# =========================
def split_audio(audio, sr=SAMPLE_RATE, chunk_duration=CHUNK_DURATION):
    chunk_size = int(chunk_duration * sr)
    chunks = []

    for start in range(0, len(audio), chunk_size):
        end = start + chunk_size
        chunks.append(audio[start:end])

    return chunks

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

    for i, chunk in enumerate(chunks):
        if len(chunk) < SAMPLE_RATE:
            continue

        inputs = processor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        with torch.no_grad():
            predicted_ids = whisper_model.generate(
                input_features=inputs.input_features.to(DEVICE),
                forced_decoder_ids=forced_ids,
                max_new_tokens=MAX_NEW_TOKENS
            )

        text = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        results.append(text.strip())

    return " ".join(results)

# =========================
# GEMINI SUMMARIZATION
# =========================
def gemini_summarize(full_text):
    client = load_gemini()

    prompt = f"""
Ringkas materi berikut dalam bentuk BULLET POINTS yang komprehensif.

Aturan:
- Tangkap alur pembahasan dari awal sampai akhir
- Sertakan semua konsep dan topik penting
- Jelaskan definisi dan perbedaan konsep jika ada
- Gunakan Bahasa Indonesia akademik
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
st.set_page_config(page_title="HEARity", page_icon="🎧")

st.title("🎧 HEARity — Speech-to-Text & Summarization")

uploaded_file = st.file_uploader(
    "Upload audio/video",
    type=["mp3", "wav", "mp4", "mkv", "m4a"]
)

video_url = st.text_input("Atau masukkan URL YouTube")

process_btn = st.button("🚀 Proses", type="primary")

if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

if process_btn:
    if not uploaded_file and not video_url:
        st.warning("Upload file atau masukkan URL.")
        st.stop()

    try:
        with st.spinner("🎧 Menyiapkan audio..."):
            if uploaded_file:
                raw_path = save_upload_to_tmp(uploaded_file)
                wav_path = run_ffmpeg_to_wav16k(raw_path)
            else:
                wav_path = download_youtube_audio(video_url)

        with st.spinner("🧠 Transkripsi (Whisper Finetuned)..."):
            transcript = whisper_transcribe(wav_path)

        with st.spinner("✍️ Ringkasan (Gemini)..."):
            summary = gemini_summarize(transcript)

        st.session_state.transcript = transcript
        st.session_state.summary = summary

        st.success("✅ Selesai!")

    except Exception as e:
        st.error(f"Gagal memproses: {e}")

st.subheader("📄 Transkrip")
st.text_area("", st.session_state.transcript, height=200)

st.subheader("📝 Ringkasan")
st.text_area("", st.session_state.summary, height=200)

st.download_button(
    "⬇️ Unduh Transkrip",
    st.session_state.transcript,
    "transcript.txt"
)

st.download_button(
    "⬇️ Unduh Ringkasan",
    st.session_state.summary,
    "summary.txt"
)
