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

# =====================
# BAGIAN HEADER / JUDUL
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
# ANTARMUKA PENGGUNA UTAMA
# =====================
# 1. UPLOADER FILE
file_diunggah = st.file_uploader(
    "📤 Unggah file audio atau video",
    type=["wav", "mp3", "mp4", "m4a", "mkv"]
)

# 2. TOMBOL PROSES
if file_diunggah:
    if st.button("Mulai Proses", use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False) as sementara:
            sementara.write(file_diunggah.read())
            jalur_audio = sementara.name

        # --- Proses Transkripsi ---
        with st.spinner("🔊 Sedang mengubah suara menjadi teks..."):
            hasil = whisper_transcribe(jalur_audio)

        if "text" in hasil:
            st.session_state.teks_lengkap = hasil
        else:
            st.session_state.teks_lengkap = " ".join(
                c["text"] for c in hasil["chunks"]
            )

        # --- Proses Ringkasan ---
        with st.spinner("✍🏻 Sedang membuat ringkasan..."):
            instruksi = f"""
Ringkas materi berikut dalam bentuk POIN-POIN PENTING.

Aturan:
- Ikuti alur pembahasan dari awal sampai akhir
- Cantumkan semua konsep dan topik penting
- Jelaskan definisi, klasifikasi, dan perbedaan konsep utama jika ada
- Sertakan tujuan, alasan pentingnya topik, dan implikasinya
- Gunakan Bahasa Indonesia yang formal dan akademis
- Jangan menyalin kalimat asli secara mentah
- Maksimal 10–12 poin penting
- Jangan gunakan notasi LaTeX
- Gunakan simbol Unicode matematika jika diperlukan
MATERI:
{st.session_state.teks_lengkap}
"""
            respon = gemini_summarize(st.session_state.teks_lengkap)

            st.session_state.ringkasan_mentah = respon

        st.session_state.selesai = True


# =====================
# TAMPILAN HASIL
# =====================
if st.session_state.selesai:
    # 3. AREA TEKS TRANSCRIPT
    st.subheader("📄 Transkrip Lengkap")
    st.text_area("", st.session_state.teks_lengkap, height=260)

    # 4. AREA TEKS RINGKASAN
    st.subheader("📝 Ringkasan Materi")

    ringkasan_rapi = []
    for baris in st.session_state.ringkasan_mentah.split("\n"):
        if baris.strip().startswith("*"):
            bersih = re.sub(r"^\*\s*", "", baris)
            bersih = re.sub(r"\*+", "", bersih)
            ringkasan_rapi.append("• " + bersih)

    st.text_area("", "\n".join(ringkasan_rapi), height=260)

    # 5. TOMBOL UNDUH PDF
    kolom1, kolom2 = st.columns(2)

    with kolom1:
        st.download_button(
            "⬇️ Unduh Transkrip (PDF)",
            "file_transkrip",
            use_container_width=True
        )

    with kolom2:
        st.download_button(
            "⬇️ Unduh Ringkasan (PDF)",
            "file_ringkasan",
            use_container_width=True
        )
