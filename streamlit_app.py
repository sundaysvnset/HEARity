import os
import tempfile
import subprocess
import streamlit as st
import torch
import librosa
import re

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from google import genai


# =========================
# KONFIGURASI
# =========================
MODEL_ID = "openai/whisper-medium"
LANG = "id"
DEVICE = "cpu"


# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource(show_spinner="📦 Memuat model Whisper ...")
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
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


processor, whisper_model = load_whisper()
klien_gemini = load_gemini()


# =========================
# HELPER AUDIO
# =========================
def simpan_file_sementara(file):
    ekstensi = os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ekstensi) as tmp:
        tmp.write(file.read())
        return tmp.name


def konversi_ke_wav_16k(input_path):
    output_path = input_path + "_16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", output_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return output_path


# =========================
# TRANSKRIPSI WHISPER
# =========================
def whisper_transkripsi(wav_path):
    audio, _ = librosa.load(wav_path, sr=16000)

    input_model = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    forced_ids = processor.get_decoder_prompt_ids(
        language=LANG,
        task="transcribe"
    )

    with torch.no_grad():
        pred_ids = whisper_model.generate(
            **input_model,
            forced_decoder_ids=forced_ids,
            max_new_tokens=448
        )

    return processor.batch_decode(
        pred_ids,
        skip_special_tokens=True
    )[0]


# =========================
# RINGKASAN GEMINI
# =========================
def buat_ringkasan(teks_lengkap):
    prompt = f"""
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
{teks_lengkap}
"""

    respons = klien_gemini.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )

    return respons.text.strip()


# =====================
# ANTARMUKA PENGGUNA UTAMA
# =====================
st.set_page_config(page_title="HEARity", page_icon="🎧")

st.title("🎧 HEARity")
st.caption("Aplikasi transkripsi dan ringkasan audio berbasis AI")

# Session state
for k in ["teks_lengkap", "ringkasan_mentah", "selesai"]:
    if k not in st.session_state:
        st.session_state[k] = "" if k != "selesai" else False


# 1. UPLOADER FILE
file_diunggah = st.file_uploader(
    "📤 Unggah file audio atau video",
    type=["wav", "mp3", "mp4", "m4a", "mkv"]
)


# 2. TOMBOL PROSES
if file_diunggah:
    if st.button("Mulai Proses", use_container_width=True):
        try:
            with tempfile.NamedTemporaryFile(delete=False) as sementara:
                sementara.write(file_diunggah.read())
                jalur_awal = sementara.name

            with st.spinner("🎧 Menyiapkan audio..."):
                jalur_wav = konversi_ke_wav_16k(jalur_awal)

            with st.spinner("🔊 Sedang mengubah suara menjadi teks..."):
                st.session_state.teks_lengkap = whisper_transkripsi(jalur_wav)

            with st.spinner("✍🏻 Sedang membuat ringkasan..."):
                st.session_state.ringkasan_mentah = buat_ringkasan(
                    st.session_state.teks_lengkap
                )

            st.session_state.selesai = True
            st.success("Proses selesai!")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")


# =====================
# TAMPILAN HASIL
# =====================
if st.session_state.selesai:
    # 3. AREA TEKS TRANSKRIP
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

    # 5. TOMBOL UNDUH TXT
    kolom1, kolom2 = st.columns(2)

    with kolom1:
        st.download_button(
            "⬇️ Unduh Transkrip (TXT)",
            st.session_state.teks_lengkap,
            file_name="transkrip.txt",
            use_container_width=True
        )

    with kolom2:
        st.download_button(
            "⬇️ Unduh Ringkasan (TXT)",
            st.session_state.ringkasan_mentah,
            file_name="ringkasan.txt",
            use_container_width=True
        )
