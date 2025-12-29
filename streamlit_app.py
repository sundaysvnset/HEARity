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
MODEL_ID = "openai/whisper-small"
LANG = "id"
DEVICE = "cpu"


# =========================
# LOAD MODEL (CACHED)
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
klien_gemini = load_gemini()


# =========================
# HELPER AUDIO
# =========================
def simpan_file_sementara(uploaded_file):
    ekstensi = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ekstensi) as tmp:
        tmp.write(uploaded_file.read())
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
            max_new_tokens=448
        )

    return processor.batch_decode(
        pred_ids,
        skip_special_tokens=True
    )[0].strip()


# =========================
# RINGKASAN GEMINI
# =========================
def gemini_summarize(teks_lengkap):
    prompt = f"""
Ringkas materi berikut dalam bentuk BULLET POINTS yang komprehensif.

Aturan:
- Tangkap alur pembahasan dari awal sampai akhir
- Sertakan semua konsep dan topik penting
- Gunakan Bahasa Indonesia yang rapi dan akademik
- Maksimal 10–12 bullet points
- Jangan menyalin kalimat mentah

MATERI:
{teks_lengkap}
"""

    respons = klien_gemini.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )

    return respons.text.strip()


# =========================
# SESSION STATE
# =========================
for k in ["teks_lengkap", "ringkasan_mentah", "selesai"]:
    if k not in st.session_state:
        st.session_state[k] = "" if k != "selesai" else False


# =========================
# UI – HEADER
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

st.write("### Pilih File Audio atau Video untuk Diproses")


# =========================
# UPLOADER FILE
# =========================
uploaded_file = st.file_uploader(
    "📤 Unggah file audio / video",
    type=["mp3", "mp4", "wav", "mkv", "m4a"]
)


# =========================
# TOMBOL PROSES
# =========================
if uploaded_file:
    if st.button("Proses Transkripsi & Ringkasan", type="primary"):
        try:
            with st.spinner("🎧 Menyiapkan audio..."):
                path_awal = simpan_file_sementara(uploaded_file)
                wav_path = konversi_ke_wav_16k(path_awal)

            with st.spinner("🔊 Melakukan transkripsi (Whisper)..."):
                st.session_state.teks_lengkap = whisper_transkripsi(wav_path)

            with st.spinner("✍🏻 Membuat ringkasan (Gemini)..."):
                st.session_state.ringkasan_mentah = gemini_summarize(
                    st.session_state.teks_lengkap
                )

            st.session_state.selesai = True
            st.success("Proses selesai!")

        except Exception as e:
            st.error(f"Gagal memproses file: {e}")


# =========================
# TAMPILAN HASIL
# =========================
if st.session_state.selesai:
    # TRANSKRIP
    st.subheader("📄 Transkrip Lengkap")
    st.text_area("", st.session_state.teks_lengkap, height=260)

    # RINGKASAN
    st.subheader("📝 Ringkasan Materi")

    ringkasan_rapi = []
    for baris in st.session_state.ringkasan_mentah.split("\n"):
        if baris.strip().startswith("*"):
            bersih = re.sub(r"^\*\s*", "", baris)
            bersih = re.sub(r"\*+", "", bersih)
            ringkasan_rapi.append("• " + bersih)

    st.text_area("", "\n".join(ringkasan_rapi), height=260)

    # UNDUH TXT
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "⬇️ Unduh Transkrip (TXT)",
            st.session_state.teks_lengkap,
            file_name="transkrip.txt",
            use_container_width=True
        )

    with col2:
        st.download_button(
            "⬇️ Unduh Ringkasan (TXT)",
            st.session_state.ringkasan_mentah,
            file_name="ringkasan.txt",
            use_container_width=True
        )
