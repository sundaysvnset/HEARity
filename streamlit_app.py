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
MODEL_ID = "openai/whisper-small"   # ⬅️ PAKAI WHISPER BAWAAN DULU
LANG = "id"
DEVICE = "cpu"

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
# WHISPER TRANSCRIPTION
# =========================
def whisper_transcribe(wav_path):
    audio, _ = librosa.load(wav_path, sr=16000)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    # ⬅️ FIX DEVICE
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    forced_ids = processor.get_decoder_prompt_ids(
        language=LANG,
        task="transcribe"
    )

    with torch.no_grad():
        pred_ids = whisper_model.generate(
            **inputs,
            forced_decoder_ids=forced_ids,
            max_new_tokens=448
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
- Sertakan semua konsep dan topik penting yang muncul
- Jelaskan definisi, klasifikasi, dan perbedaan konsep utama jika ada
- Sertakan tujuan, alasan pentingnya topik, dan implikasinya
- Gunakan Bahasa Indonesia yang rapi, netral, dan akademik
- Jangan menyalin kalimat mentah
- Maksimal 10–12 bullet points
- Jangan gunakan notasi LaTeX
- Gunakan simbol Unicode matematika jika diperlukan

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

st.write("### Pilih File Audio/Video atau URL Video untuk Diproses")

uploaded_file = st.file_uploader(
    "📤 Unggah file audio atau video",
    type=["mp3", "mp4", "wav", "mkv"]
)

st.write("Atau, masukkan URL video (misalnya YouTube)")
video_url = st.text_input("Masukkan URL Video (opsional)")

if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

process_btn = st.button(
    "Proses Transkripsi & Ringkasan",
    type="primary"
)

if process_btn:
    if uploaded_file is None and not video_url:
        st.warning("Silakan upload file atau masukkan URL terlebih dahulu.")
        st.stop()

    try:
        with st.spinner("Menyiapkan audio..."):
            if uploaded_file:
                input_path = save_upload_to_tmp(uploaded_file)
                wav_path = run_ffmpeg_to_wav16k(input_path)
            else:
                wav_path = download_youtube_audio(video_url)

        with st.spinner("Melakukan transkripsi (Whisper)..."):
            transcript = whisper_transcribe(wav_path)

        with st.spinner("Membuat ringkasan (Gemini)..."):
            summary = gemini_summarize(transcript)

        st.session_state.transcript = transcript
        st.session_state.summary = summary

        st.success("Selesai! Transkrip dan ringkasan tersedia.")

    except Exception as e:
        st.error(f"Gagal memproses: {e}")

st.write("### 📄 Transkrip Lengkap:")
st.text_area(
    "Transkrip",
    value=st.session_state.transcript or "Transkrip akan ditampilkan di sini.",
    height=200
)

st.write("### 📝 Ringkasan Materi")
st.text_area(
    "Ringkasan",
    value=st.session_state.summary or "Ringkasan akan ditampilkan di sini.",
    height=200
)

st.download_button(
    "⬇️ Unduh Transkrip (PDF)",
    st.session_state.transcript,
    "transcript.txt",
    mime="text/plain",
    use_container_width=True
)

st.download_button(
    "⬇️ Unduh Ringkasan (PDF)",
    st.session_state.summary,
    "summary.txt",
    mime="text/plain",
    use_container_width=True
)
