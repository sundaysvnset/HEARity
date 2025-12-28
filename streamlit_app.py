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
DEVICE = "cpu"  

# =========================
# LOAD MODELS (CACHED)
# =========================
@st.cache_resource(show_spinner="📦 Memuat model Whisper...")
def load_whisper():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,  # Use dtype instead of torch_dtype
        low_cpu_mem_usage=True
    )
    model.to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource
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
# SPLIT AUDIO INTO CHUNKS
# =========================
def split_audio_into_chunks(audio, chunk_duration=30, sr=16000):
    # Split the audio into chunks of the given duration (in seconds)
    chunk_samples = int(chunk_duration * sr)
    num_chunks = len(audio) // chunk_samples
    chunks = [audio[i * chunk_samples:(i + 1) * chunk_samples] for i in range(num_chunks)]
    
    # Handle the remainder of the audio that doesn't fill a full chunk
    if len(audio) % chunk_samples != 0:
        chunks.append(audio[num_chunks * chunk_samples:])
    
    return chunks

# =========================
# WHISPER TRANSCRIPTION
# =========================
def whisper_transcribe(wav_path):
    audio, _ = librosa.load(wav_path, sr=16000)

    # Split the audio into 30-second chunks
    chunks = split_audio_into_chunks(audio)

    transcriptions = []
    for chunk in chunks:
        inputs = processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt"
        )

        forced_ids = processor.get_decoder_prompt_ids(
            language=LANG,
            task="transcribe"
        )

        with torch.no_grad():
            pred_ids = whisper_model.generate(
                **inputs,
                forced_decoder_ids=forced_ids,
                max_new_tokens=400
            )

        transcription = processor.batch_decode(
            pred_ids,
            skip_special_tokens=True
        )[0]
        transcriptions.append(transcription)

    # Combine all transcriptions
    full_transcription = " ".join(transcriptions)
    return full_transcription

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
st.set_page_config(page_title="HEARity", page_icon="🎧")

st.title(
    "HEARity: Speech-to-Text Summarization Berbasis Generative AI",
    anchor="title"
)

st.write("""
Penyandang gangguan pendengaran seringkali mengalami kesulitan untuk memahami percakapan,
perkuliahan, atau informasi berbasis suara lainnya.

Walaupun saat ini sudah ada teknologi speech recognition, hasil transkrip sering kali sangat panjang
dan sulit dipahami. Oleh karena itu, **HEARity** dikembangkan untuk mengubah suara menjadi teks
dan menghasilkan ringkasan otomatis.

**Final Project** ini bertujuan meningkatkan aksesibilitas komunikasi dan pendidikan
bagi penyandang gangguan pendengaran.
""")

st.write("### Pilih File Audio/Video atau URL Video untuk Diproses")

uploaded_file = st.file_uploader(
    "Pilih file audio/video",
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

        with st.spinner("Melakukan transkripsi (Whisper finetuned)..."):
            transcript = whisper_transcribe(wav_path)

        with st.spinner("Membuat ringkasan (Gemini)..."):
            summary = gemini_summarize(transcript)

        st.session_state.transcript = transcript
        st.session_state.summary = summary

        st.success("Selesai! Transkrip dan ringkasan tersedia.")

    except Exception as e:
        st.error(f"Gagal memproses: {e}")

st.write("### Transkrip:")
st.text_area(
    "Transkrip",
    value=st.session_state.transcript or "Transkrip akan ditampilkan di sini.",
    height=200
)

st.write("### Ringkasan:")
st.text_area(
    "Ringkasan",
    value=st.session_state.summary or "Ringkasan akan ditampilkan di sini.",
    height=200
)

st.download_button(
    "Unduh Transkrip",
    st.session_state.transcript,
    "transcript.txt",
    mime="text/plain",
    use_container_width=True
)

st.download_button(
    "Unduh Ringkasan",
    st.session_state.summary,
    "summary.txt",
    mime="text/plain",
    use_container_width=True
)
