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
MODEL_ID = "openai/whisper-small"   # Whisper bawaan (FREE)
LANG = "id"
DEVICE = "cpu"                     # Streamlit Cloud = CPU

# =========================
# LOAD MODELS (CACHED)
# =========================
@st.cache_resource(show_spinner="📦 Memuat model Whisper...")
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
    """
    OPTIONAL FEATURE
    Bisa gagal di Streamlit Cloud → handled gracefully
    """
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
            "Gagal mengunduh audio dari YouTube.\n\n"
            "Kemungkinan penyebab:\n"
            "- yt-dlp tidak tersedia di server\n"
            "- Video dibatasi (private / age-restricted)\n"
            "- YouTube memblokir akses\n\n"
            "Solusi: download audio secara manual lalu upload file."
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
            max_new_tokens=400   # ⬅️ FIX TOKEN LIMIT
        )

    return processor.batch_decode(
        pred_ids,
        skip_special_tokens=True
    )[0].strip()

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
- Jelaskan definisi, klasifikasi, dan perbedaan konsep utama
- Sertakan tujuan dan implikasi pembahasan
- Gunakan Bahasa Indonesia akademik dan netral
- Jangan menyalin kalimat mentah
- Maksimal 10–12 bullet points
- Jangan gunakan LaTeX

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
st.info(
    "ℹ️ Disarankan menggunakan **Upload File**.\n"
    "Fitur YouTube URL bersifat opsional dan bisa gagal di server publik."
)

uploaded_file = st.file_uploader(
    "📤 Unggah file audio atau video",
    type=["mp3", "wav", "mp4", "mkv"]
)

st.write("Atau masukkan URL YouTube (opsional)")
video_url = st.text_input("URL Video")

if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

if st.button("🚀 Proses Transkripsi & Ringkasan", type="primary"):
    if uploaded_file is None and not video_url:
        st.warning("Upload file atau masukkan URL terlebih dahulu.")
        st.stop()

    try:
        with st.spinner("🎼 Menyiapkan audio..."):
            if uploaded_file:
                input_path = save_upload_to_tmp(uploaded_file)
                wav_path = run_ffmpeg_to_wav16k(input_path)
            else:
                wav_path = download_youtube_audio(video_url)

        with st.spinner("📝 Melakukan transkripsi (Whisper)..."):
            transcript = whisper_transcribe(wav_path)

        with st.spinner("🧠 Membuat ringkasan (Gemini)..."):
            summary = gemini_summarize(transcript)

        st.session_state.transcript = transcript
        st.session_state.summary = summary

        st.success("✅ Selesai!")

    except Exception as e:
        st.error(f"Gagal memproses:\n{e}")

# =========================
# OUTPUT
# =========================
st.write("### 📄 Transkrip Lengkap")
st.text_area(
    "Hasil Transkrip",
    value=st.session_state.transcript,
    height=200
)

st.write("### 📝 Ringkasan Materi")
st.text_area(
    "Hasil Ringkasan",
    value=st.session_state.summary,
    height=200
)

st.download_button(
    "⬇️ Unduh Transkrip",
    st.session_state.transcript,
    "transcript.txt",
    mime="text/plain"
)

st.download_button(
    "⬇️ Unduh Ringkasan",
    st.session_state.summary,
    "summary.txt",
    mime="text/plain"
)
