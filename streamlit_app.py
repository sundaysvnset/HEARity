import streamlit as st
import torch
import tempfile
import os
import re
import subprocess

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)

from google import genai

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    ListFlowable,
    ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm


# =====================
# KONFIGURASI APLIKASI
# =====================
# Gunakan model Whisper Small yang lebih ringan untuk Streamlit
MODEL_ID = "openai/whisper-small"

# Cek apakah ada GPU (untuk local) atau CPU (untuk Streamlit Cloud)
device = 0 if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="HEARity",
    page_icon="🎧",
    layout="centered"
)


# =====================
# BAGIAN HEADER / JUDUL
# =====================
st.markdown(
    """
    <div style="padding:30px 10px;">
        <h1>🎧 HEARity</h1>
        <p style="color:#b0b0b0;">
             Speech-to-Text & Automatic Summarization berbasis Whisper + Generative AI
        </p>
        <ul>
            <li>Mengubah audio atau video menjadi teks</li>
            <li>Menyusun ringkasan materi secara otomatis</li>
        </ul>
        <b>Group 8 – Final Project</b>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================
# SESSION STATE
# Menyimpan hasil agar tidak hilang saat halaman refresh
# =====================
for k in ["done", "full_text", "summary_raw", "pdf_transcript", "pdf_summary"]:
    if k not in st.session_state:
        st.session_state[k] = "" if k != "done" else False


# =====================
# LOAD MODEL WHISPER & GEMINI
# =====================
@st.cache_resource(show_spinner=False)
def load_asr_pipeline():
    """Load Whisper Small model yang lebih ringan"""
    try:
        with st.spinner("🔄 Memuat model Whisper Small..."):
            # Gunakan model Whisper Small yang lebih ringan
            processor = WhisperProcessor.from_pretrained(MODEL_ID)
            
            # Tentukan tipe data berdasarkan device
            if torch.cuda.is_available():
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
                
            model = WhisperForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            )
            
            # Buat pipeline dengan konfigurasi optimal
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=device,
                chunk_length_s=30,
                stride_length_s=5,
                batch_size=4,
                return_timestamps=False
            )
            
            return pipe
            
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        # Fallback ke model yang lebih kecil
        st.info("Menggunakan model Whisper Base sebagai fallback...")
        return pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=device,
            chunk_length_s=30
        )


@st.cache_resource(show_spinner=False)
def load_gemini_client():
    """Load Gemini client dengan error handling"""
    try:
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("⚠️ GEMINI_API_KEY tidak ditemukan di Secrets")
            st.info("Tambahkan API key di: Settings → Secrets → GEMINI_API_KEY")
            return None
        
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception as e:
        st.error(f"Gagal memuat Gemini: {e}")
        return None


# =====================
# FUNGSI BANTUAN AUDIO
# =====================
def konversi_ke_wav16k(input_path, output_path=None):
    """Konversi file audio/video ke WAV 16kHz mono menggunakan ffmpeg"""
    if output_path is None:
        output_path = input_path + "_converted.wav"
    
    try:
        # Gunakan subprocess untuk ffmpeg
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ac", "1",          # Mono channel
            "-ar", "16000",      # 16kHz sample rate
            "-acodec", "pcm_s16le",  # 16-bit PCM
            output_path
        ]
        
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"Error konversi audio: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def cek_durasi_audio(file_path):
    """Cek durasi file audio"""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration
    except:
        pass
    return None


# =====================
# FUNGSI PEMBUATAN PDF
# =====================
def save_transcript_pdf(text, output_path):
    """Simpan transkrip ke PDF"""
    try:
        styles = getSampleStyleSheet()

        styles.add(ParagraphStyle(
            name="TitleStyle",
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            spaceAfter=20,
            alignment=1  # Center
        ))

        styles.add(ParagraphStyle(
            name="BodyStyle",
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            spaceAfter=12,
            alignment=4  # Justify
        ))

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm
        )

        story = [Paragraph("TRANSCRIPT MATERI", styles["TitleStyle"])]

        # Pisahkan teks menjadi paragraf
        paragraphs = text.split('\n')
        for para in paragraphs:
            para = para.strip()
            if para:
                story.append(Paragraph(para, styles["BodyStyle"]))

        doc.build(story)
        return True
    except Exception as e:
        st.error(f"Error membuat PDF transkrip: {e}")
        return False


def save_summary_pdf(summary, output_path):
    """Simpan ringkasan ke PDF dengan bullet points"""
    try:
        styles = getSampleStyleSheet()

        styles.add(ParagraphStyle(
            name="TitleStyle",
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            alignment=1,
            spaceAfter=20
        ))

        styles.add(ParagraphStyle(
            name="BulletStyle",
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            spaceAfter=8,
            alignment=4
        ))

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm
        )

        story = [Paragraph("RINGKASAN MATERI", styles["TitleStyle"])]

        bullets = []
        for line in summary.split("\n"):
            line = line.strip()
            if line.startswith("*") or line.startswith("-") or line.startswith("•"):
                clean = re.sub(r"^[*\-•]\s*", "", line)
                if clean:  # Hanya tambahkan jika tidak kosong
                    bullets.append(clean)

        if bullets:
            story.append(ListFlowable(
                [
                    ListItem(
                        Paragraph(b, styles["BulletStyle"]),
                        bulletText='•'
                    )
                    for b in bullets
                ],
                bulletType="bullet",
                leftIndent=20
            ))
        else:
            # Jika tidak ada bullet points, tampilkan sebagai paragraf biasa
            story.append(Paragraph(summary, styles["BulletStyle"]))

        doc.build(story)
        return True
    except Exception as e:
        st.error(f"Error membuat PDF ringkasan: {e}")
        return False


# =====================
# FUNGSI UTAMA
# =====================
def proses_file_audio(file_path):
    """Proses utama: transkripsi dan ringkasan"""
    
    # 1. Konversi ke format yang sesuai
    with st.spinner("🔄 Mengkonversi file audio..."):
        wav_path = konversi_ke_wav16k(file_path)
        if not wav_path or not os.path.exists(wav_path):
            st.error("Gagal mengkonversi file audio")
            return None, None
    
    # 2. Cek durasi
    duration = cek_durasi_audio(wav_path)
    if duration:
        if duration > 600:  # > 10 menit
            st.warning(f"⚠️ File ({duration/60:.1f} menit) mungkin membutuhkan waktu lebih lama")
    
    # 3. Load model (lazy loading)
    if 'pipe_ft' not in st.session_state:
        with st.spinner("📦 Memuat model Whisper..."):
            st.session_state.pipe_ft = load_asr_pipeline()
    
    if 'gemini_client' not in st.session_state:
        with st.spinner("📦 Memuat Gemini..."):
            st.session_state.gemini_client = load_gemini_client()
    
    # 4. Transkripsi
    with st.spinner("🔊 Melakukan transkripsi..."):
        try:
            result = st.session_state.pipe_ft(wav_path)
            
            if isinstance(result, dict) and "text" in result:
                full_text = result["text"]
            elif isinstance(result, dict) and "chunks" in result:
                full_text = " ".join(c["text"] for c in result["chunks"])
            else:
                full_text = str(result)
            
            # Bersihkan teks
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            # Update progress
            st.success(f"✅ Transkripsi selesai ({len(full_text.split())} kata)")
            
        except Exception as e:
            st.error(f"Error transkripsi: {e}")
            full_text = ""
    
    # 5. Ringkasan (jika ada teks dan Gemini client tersedia)
    summary_raw = ""
    if full_text and st.session_state.gemini_client:
        with st.spinner("✍🏻 Membuat ringkasan..."):
            try:
                prompt = f"""Buat ringkasan dari materi berikut dalam bentuk bullet points:

Aturan:
- Tangkap alur pembahasan utama
- Sertakan konsep dan poin penting
- Gunakan Bahasa Indonesia yang jelas
- Maksimal 8-10 bullet points
- Format: setiap poin dimulai dengan *

MATERI:
{full_text}
"""
                response = st.session_state.gemini_client.models.generate_content(
                    model="models/gemini-1.5-flash",  # Gunakan model yang lebih stabil
                    contents=prompt
                )
                
                summary_raw = response.text.strip()
                st.success("✅ Ringkasan selesai")
                
            except Exception as e:
                st.error(f"Error membuat ringkasan: {e}")
                summary_raw = "Gagal membuat ringkasan. Error: " + str(e)
    
    # 6. Bersihkan file temporary
    try:
        os.unlink(wav_path)
    except:
        pass
    
    return full_text, summary_raw


# =====================
# ANTARMUKA UTAMA (UI)
# =====================
st.divider()

# Uploader file
uploaded_file = st.file_uploader(
    "📤 Unggah file audio atau video (maksimal 10 menit)",
    type=["wav", "mp3", "mp4", "m4a", "mkv", "avi", "flac"],
    help="Format yang didukung: WAV, MP3, MP4, M4A, MKV, AVI, FLAC"
)

# Tombol proses
if uploaded_file is not None:
    # Tampilkan info file
    file_size = uploaded_file.size / (1024 * 1024)  # MB
    st.info(f"📄 File: {uploaded_file.name} ({file_size:.2f} MB)")
    
    if st.button("🚀 Proses File", type="primary", use_container_width=True):
        # Simpan file upload ke temporary
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name
        
        # Proses file
        full_text, summary_raw = proses_file_audio(audio_path)
        
        if full_text:
            # Simpan ke session state
            st.session_state.full_text = full_text
            st.session_state.summary_raw = summary_raw
            
            # Buat PDF
            os.makedirs("output", exist_ok=True)
            
            st.session_state.pdf_transcript = "output/transkrip.pdf"
            st.session_state.pdf_summary = "output/ringkasan.pdf"
            
            # Buat PDF
            with st.spinner("📄 Membuat PDF..."):
                pdf1_ok = save_transcript_pdf(full_text, st.session_state.pdf_transcript)
                pdf2_ok = save_summary_pdf(summary_raw, st.session_state.pdf_summary)
                
                if pdf1_ok and pdf2_ok:
                    st.session_state.done = True
                    st.success("✅ Semua proses selesai!")
                else:
                    st.warning("Proses selesai, tetapi ada masalah dengan pembuatan PDF")
        
        # Bersihkan file temporary
        try:
            os.unlink(audio_path)
        except:
            pass

# Tambahkan informasi aplikasi
with st.expander("ℹ️ Informasi Aplikasi"):
    st.markdown("""
    ### Fitur:
    - **Model**: Whisper Small (OpenAI) - optimal untuk Streamlit
    - **Bahasa**: Mendukung transkripsi ke Bahasa Indonesia
    - **Format Input**: Audio (WAV, MP3) dan Video (MP4, M4A, MKV, AVI)
    - **Output**: Teks transkrip + ringkasan otomatis + PDF
    
    ### Batasan:
    - Maksimal durasi: 10 menit untuk performa optimal
    - Kualitas terbaik: Audio dengan suara jelas, minim noise
    - Koneksi internet diperlukan untuk Gemini API
    
    ### Tips:
    1. Pastikan audio/video memiliki suara yang jelas
    2. File lebih pendek = proses lebih cepat
    3. Untuk hasil terbaik, gunakan format MP3 atau WAV
    """)

# =====================
# TAMPILAN HASIL
# =====================
if st.session_state.done and st.session_state.full_text:
    st.divider()
    st.subheader("📄 Hasil Transkrip")
    
    # Tampilkan transkrip dengan expander
    with st.expander("Tampilkan/Sembunyikan Transkrip Lengkap", expanded=True):
        st.text_area(
            "Transkrip",
            st.session_state.full_text,
            height=200,
            label_visibility="collapsed"
        )
    
    st.subheader("📝 Hasil Ringkasan")
    
    # Format ringkasan menjadi bullet points yang rapi
    if st.session_state.summary_raw:
        ringkasan_bersih = []
        for line in st.session_state.summary_raw.split("\n"):
            line = line.strip()
            if line:
                # Bersihkan formatting
                clean_line = re.sub(r'^\*\s*', '• ', line)
                clean_line = re.sub(r'^\-\s*', '• ', clean_line)
                ringkasan_bersih.append(clean_line)
        
        summary_display = "\n".join(ringkasan_bersih)
    else:
        summary_display = "Ringkasan tidak tersedia"
    
    with st.expander("Tampilkan/Sembunyikan Ringkasan", expanded=True):
        st.text_area(
            "Ringkasan",
            summary_display,
            height=200,
            label_visibility="collapsed"
        )
    
    # Tombol download
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if os.path.exists(st.session_state.pdf_transcript):
            with open(st.session_state.pdf_transcript, "rb") as f:
                st.download_button(
                    "⬇️ Unduh Transkrip (PDF)",
                    f.read(),
                    file_name="transkrip_materi.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.button("📄 Transkrip PDF", disabled=True, use_container_width=True)
    
    with col2:
        if os.path.exists(st.session_state.pdf_summary):
            with open(st.session_state.pdf_summary, "rb") as f:
                st.download_button(
                    "⬇️ Unduh Ringkasan (PDF)",
                    f.read(),
                    file_name="ringkasan_materi.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.button("📄 Ringkasan PDF", disabled=True, use_container_width=True)
    
    with col3:
        # Tombol reset
        if st.button("🔄 Proses File Baru", use_container_width=True):
            for k in ["done", "full_text", "summary_raw"]:
                st.session_state[k] = "" if k != "done" else False
            st.rerun()

# Footer
st.divider()
st.caption("✨ HEARity v2.0 | Menggunakan Whisper Small untuk performa optimal di Streamlit")
