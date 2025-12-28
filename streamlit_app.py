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
# KONFIGURASI
# =========================
MODEL_ID = "jovangelo/whispermodelproyek"
BAHASA = "id"
PERANGKAT = "cuda" if torch.cuda.is_available() else "cpu"

SAMPEL_RATE = 16000

# =========================
# MUAT MODEL
# =========================
@st.cache_resource(show_spinner="📦 Memuat model Whisper...")
def muat_whisper():
    prosesor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    model.to(PERANGKAT)
    model.eval()
    return prosesor, model


@st.cache_resource(show_spinner="📦 Memuat Gemini...")
def muat_gemini():
    if "GEMINI_API_KEY" not in st.secrets:
        raise RuntimeError("GEMINI_API_KEY belum diset di Secrets")
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


prosesor, model_whisper = muat_whisper()

# =========================
# BANTUAN AUDIO
# =========================
def simpan_unggahan_ke_sementara(file_diunggah):
    ekstensi = os.path.splitext(file_diunggah.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ekstensi) as sementara:
        sementara.write(file_diunggah.read())
        return sementara.name


def jalankan_ffmpeg_ke_wav16k(jalur_input):
    jalur_keluaran = jalur_input + "_16k.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", jalur_input,
            "-ac", "1",
            "-ar", "16000",
            jalur_keluaran
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return jalur_keluaran


# =========================
# PEMBERSIH TRANSCRIPT OTOMATIS
# =========================
def bersihkan_transkrip(teks: str) -> str:
    teks = teks.lower()

    # hapus kata pengisi umum
    kata_pengisi = [
        r"\buh+\b", r"\bum+\b", r"\beh+\b", r"\bhmm+\b",
        r"\banu\b", r"\bgitu\b", r"\bjadi\b"
    ]
    for pengisi in kata_pengisi:
        teks = re.sub(pengisi, "", teks)

    # hapus pengulangan kata berturut-turut
    teks = re.sub(r"\b(\w+)( \1\b)+", r"\1", teks)

    # rapikan spasi
    teks = re.sub(r"\s+", " ", teks)

    return teks.strip()


# =========================
# TRANSCRIPT WHISPER (TANPA CHUNKING)
# =========================
def whisper_transkrip(jalur_wav):
    # Muat audio lengkap
    audio, _ = librosa.load(jalur_wav, sr=SAMPEL_RATE)
    
    # Hitung durasi audio
    durasi = len(audio) / SAMPEL_RATE
    st.info(f"📊 Durasi audio: {durasi:.2f} detik")
    
    # Siapkan input untuk Whisper (tanpa chunking)
    input_features = prosesor(
        audio,
        sampling_rate=SAMPEL_RATE,
        return_tensors="pt"
    ).input_features
    
    # Generate transcript
    dengan_torch = torch.no_grad()
    
    # Buat prompt untuk bahasa Indonesia
    prompt_ids = prosesor.get_decoder_prompt_ids(
        language=BAHASA,
        task="transcribe"
    )
    
    dengan_torch.__enter__()
    try:
        pred_ids = model_whisper.generate(
            input_features=input_features.to(PERANGKAT),
            forced_decoder_ids=prompt_ids,
            max_new_tokens=500,  # Cukup untuk 5-10 menit
            do_sample=False
        )
    finally:
        dengan_torch.__exit__(None, None, None)
    
    teks_mentah = prosesor.batch_decode(
        pred_ids,
        skip_special_tokens=True
    )[0]
    
    return bersihkan_transkrip(teks_mentah)


# =========================
# RINGKASAN GEMINI
# =========================
def gemini_ringkasan(teks_lengkap):
    klien = muat_gemini()

    prompt = f"""
Buat ringkasan dari materi berikut dengan format BULLET POINTS.

Panduan:
- Tangkap alur pembahasan dari awal sampai akhir
- Sertakan konsep utama dan penjelasan penting
- Gunakan Bahasa Indonesia yang jelas dan ringkas
- Jangan menyalin kalimat asli secara mentah
- Maksimal 8-10 bullet points

MATERI:
{teks_lengkap}
"""

    respon = klien.models.generate_content(
        model="models/gemini-1.5-flash",
        contents=prompt
    )

    return respon.text.strip()


# =========================
# ANTARMUKA PENGGUNA
# =========================
st.set_page_config(
    page_title="HEARity - Transkripsi & Ringkasan",
    page_icon="🎧",
    layout="centered"
)

# =====================
# BAGIAN HEADER / JUDUL
# =====================
st.markdown(
    """
    <div style="padding:20px 10px;">
        <h1>🎧 HEARity</h1>
        <p style="color:#b0b0b0;">
             Konversi Suara ke Teks & Ringkasan Otomatis
        </p>
        <ul>
            <li>Untuk video/audio 5-10 menit</li>
            <li>Tidak menggunakan chunking</li>
            <li>Hasil lebih natural dan cepat</li>
        </ul>
        <b>Kelompok 8 – Proyek Akhir</b>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# 1. UPLOADER FILE
file_diunggah = st.file_uploader(
    "📤 Unggah file audio atau video (5-10 menit)",
    type=["wav", "mp3", "mp4", "m4a", "mkv"]
)

# SESSION STATE
st.session_state.setdefault("transkrip", "")
st.session_state.setdefault("ringkasan", "")
st.session_state.setdefault("berhasil", False)

# BUTTON
if file_diunggah is not None:
    if st.button("🚀 Proses Sekarang", type="primary", use_container_width=True):
        try:
            # Langkah 1: Persiapan audio
            with st.spinner("🎵 Menyiapkan audio..."):
                jalur_input = simpan_unggahan_ke_sementara(file_diunggah)
                jalur_wav = jalankan_ffmpeg_ke_wav16k(jalur_input)
                st.success("✅ Audio siap diproses!")
            
            # Langkah 2: Transkripsi
            with st.spinner("🔊 Sedang mengubah suara menjadi teks (tanpa chunking)..."):
                transkrip = whisper_transkrip(jalur_wav)
                st.session_state.transkrip = transkrip
                st.success(f"✅ Transkripsi selesai! ({len(transkrip.split())} kata)")
            
            # Langkah 3: Ringkasan
            with st.spinner("✍🏻 Sedang membuat ringkasan..."):
                ringkasan = gemini_ringkasan(transkrip)
                st.session_state.ringkasan = ringkasan
                st.success("✅ Ringkasan selesai!")
            
            # Bersihkan file sementara
            os.unlink(jalur_input)
            os.unlink(jalur_wav)
            
            st.session_state.berhasil = True
            
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {str(e)}")
            st.info("ℹ️ Pastikan file tidak terlalu panjang (maksimal 10 menit)")

st.divider()

# OUTPUT
if st.session_state.berhasil or st.session_state.transkrip:
    st.subheader("📄 Hasil Transkrip")
    st.text_area(
        "Transkrip Lengkap",
        st.session_state.transkrip,
        height=200,
        key="kotak_transkrip"
    )
    
    st.subheader("📝 Hasil Ringkasan")
    # Format ringkasan menjadi bullet points yang lebih rapi
    ringkasan_rapi = st.session_state.ringkasan
    # Ganti * dengan • untuk bullet points
    ringkasan_rapi = re.sub(r'^\*', '•', ringkasan_rapi, flags=re.MULTILINE)
    ringkasan_rapi = re.sub(r'\*\*', '', ringkasan_rapi)  # Hapus bold markdown
    
    st.text_area(
        "Ringkasan Otomatis",
        ringkasan_rapi,
        height=200,
        key="kotak_ringkasan"
    )
    
    # Tombol Unduh
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "📥 Unduh Transkrip (TXT)",
            st.session_state.transkrip,
            "transkrip.txt",
            mime="text/plain",
            use_container_width=True,
            icon="📄"
        )
    
    with col2:
        st.download_button(
            "📥 Unduh Ringkasan (TXT)",
            st.session_state.ringkasan,
            "ringkasan.txt",
            mime="text/plain",
            use_container_width=True,
            icon="📝"
        )
    
    # Tombol Reset
    if st.button("🔄 Proses File Baru", use_container_width=True):
        st.session_state.transkrip = ""
        st.session_state.ringkasan = ""
        st.session_state.berhasil = False
        st.rerun()

# Informasi tambahan
with st.expander("ℹ️ Informasi Aplikasi"):
    st.markdown("""
    ### Fitur Aplikasi:
    1. **Tanpa Chunking**: Memproses audio secara utuh tanpa memotong-motong
    2. **Cepat**: Optimal untuk video 5-10 menit
    3. **Natural**: Hasil transkripsi lebih mengalir dan natural
    
    ### Batasan:
    - Maksimal 10 menit per file
    - Format yang didukung: WAV, MP3, MP4, M4A, MKV
    - Membutuhkan koneksi internet untuk Gemini API
    
    ### Cara Kerja:
    1. Upload file audio/video
    2. Sistem konversi ke WAV 16kHz
    3. Whisper transkripsi ke teks
    4. Gemini buat ringkasan
    5. Unduh hasil dalam format TXT
    """)
