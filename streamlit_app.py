import os
import tempfile
import subprocess
import streamlit as st
import torch
import librosa

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from google import genai

# =========================
# CONFIG
# =========================
MODEL_ID = "jovangelo/whispermodelproyek"
LANG = "id"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODELS (CACHED)
# =========================
@st.cache_resource(show_spinner="📦 Memuat model Whisper finetuned...")
def load_whisper():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )

    model.to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource
def load_gemini():
    if "GEMINI_API_KEY" not in st.secrets:
        raise RuntimeError("GEMINI_API_KEY belum diset di Streamlit Secrets")
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
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", out_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return out_path


def download_youtube_audio(url):
    tmpdir = tempfile.mkdtemp()
    outtmpl = os.path.join(tmpdir, "audio.%(ext)s")

    try:
        subprocess.run(
            [
                "yt-dlp",
                "-f", "bestaudio/best",
                "--no-playlist",
                "-o", outtmpl,
                url
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "Gagal download YouTube.\n"
            "Gunakan upload file audio sebagai alternatif."
        )

    files = os.listdir(tmpdir)
    if not files:
        raise RuntimeError("Audio YouTube tidak ditemukan.")

    return run_ffmpeg_to_wav16k(os.path.join(tmpdir, files[0]))

# =========================
# WHISPER TRANSCRIPTION
# =========================
def whisper_transcribe(wav_path):
    audio, _ = librosa.load(wav_path, sr=16000)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    forced_ids = processor.get_decoder_prompt_ids(
        language=LANG,
        task="transcribe"
    )

    with torch.no_grad():
        pred_ids = whisper_model.generate(
            **inputs,
            forced_decoder_ids=forced_ids,
            max_new_tokens=400   # ✅ aman dari overflow
        )

    return processor.batch_decode(
        pred_ids,
        skip_special_tokens=True
    )[0].strip()

# =========================
# GEMINI SUMMARIZATION
# =========================
def gemini_summarize(text):
    client = load_gemini()

    prompt = f"""
Ringkas materi berikut dalam bentuk BULLET POINTS yang komprehensif.

Aturan:
- Tangkap alur pembahasan dari awal sampai akhir
- Sertakan konsep penting
- Gunakan Bahasa Indonesia akademik
- Maksimal 10–12 bullet points
- Jangan menyalin kalimat mentah

MATERI:
{text}
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

st.title("🎧 HEARity – Finetuned Whisper + Gemini")

st.info(
    f"""
**Model Whisper:** Finetuned  
**Device:** {DEVICE.upper()}  
**Language:** Indonesian  
"""
)

uploaded_file = st.file_uploader(
    "Upload file audio / video",
    type=["mp3", "wav", "mp4", "mkv"]
)

video_url = st.text_input("Atau URL YouTube (opsional)")

if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

if st.button("🚀 Proses", type="primary"):
    if not uploaded_file and not video_url:
        st.warning("Upload file atau masukkan URL.")
        st.stop()

    try:
        with st.spinner("🎼 Menyiapkan audio..."):
            if uploaded_file:
                tmp = save_upload_to_tmp(uploaded_file)
                wav_path = run_ffmpeg_to_wav16k(tmp)
            else:
                wav_path = download_youtube_audio(video_url)

        with st.spinner("📝 Transkripsi (Whisper Finetuned)..."):
            transcript = whisper_transcribe(wav_path)

        with st.spinner("🧠 Ringkasan (Gemini)..."):
            summary = gemini_summarize(transcript)

        st.session_state.transcript = transcript
        st.session_state.summary = summary

        st.success("✅ Selesai")

    except Exception as e:
        st.error(f"Gagal memproses:\n{e}")

st.subheader("📄 Transkrip")
st.text_area("", st.session_state.transcript, height=200)

st.subheader("🧾 Ringkasan")
st.text_area("", st.session_state.summary, height=200)

st.download_button(
    "⬇️ Download Transkrip",
    st.session_state.transcript,
    "transcript.txt"
)

st.download_button(
    "⬇️ Download Ringkasan",
    st.session_state.summary,
    "summary.txt"
)
