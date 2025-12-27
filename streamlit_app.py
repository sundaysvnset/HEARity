import os
import tempfile
import subprocess
import streamlit as st
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from google import genai
from google.genai import types

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/whisper-medium-finetuned-id"  # Path to model folder in GitHub repo
LANG = "id"  # Language code for Indonesian

# =========================
# CACHED LOADERS (for efficiency)
# =========================
@st.cache_resource
def load_whisper(model_id: str):
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.eval()  # set to evaluation mode
    return processor, model

@st.cache_resource
def gemini_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# =========================
# HELPERS
# =========================
def save_upload_to_tmp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def run_ffmpeg_to_wav16k(input_path: str) -> str:
    out_path = input_path + "_16k.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",              # no video
        "-ac", "1",         # mono
        "-ar", "16000",     # 16kHz
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def download_youtube_audio(url: str) -> str:
    tmpdir = tempfile.mkdtemp()
    outtmpl = os.path.join(tmpdir, "yt.%(ext)s")
    cmd = ["yt-dlp", "-f", "bestaudio", "-o", outtmpl, url]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)]
    if not files:
        raise RuntimeError("Gagal download audio dari URL.")
    downloaded = files[0]
    return run_ffmpeg_to_wav16k(downloaded)

def whisper_transcribe(wav_path: str, processor, model) -> str:
    audio, _ = librosa.load(wav_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    forced_ids = processor.get_decoder_prompt_ids(language=LANG, task="transcribe")
    pred_ids = model.generate(**inputs, forced_decoder_ids=forced_ids)
    text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    return text.strip()

def gemini_summarize(text: str) -> str:
    client = gemini_client()
    model_name = st.secrets.get("GEMINI_MODEL", "gemini-2.0-flash-001")

    prompt = (
        "Buat ringkasan dalam Bahasa Indonesia dari transkrip berikut.\n"
        "Buat 5–8 poin bullet yang jelas, singkat, dan mudah dipahami.\n\n"
        f"{text}"
    )

    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=450
        ),
    )
    return (resp.text or "").strip()


# =========================
# LOAD MODELS (once)
# =========================
processor, whisper_model = load_whisper(MODEL_PATH)


# =========================
# UI (SAMA seperti yang kamu minta)
# =========================
st.title("HEARity: Speech-to-Text Summarization Berbasis Generative AI", anchor="title")

st.write("""
    Penyandang gangguan pendengaran seringkali mengalami kesulitan untuk memahami percakapan, perkuliahan, atau informasi berbasis suara lainnya. 
    Walaupun saat ini sudah ada teknologi speech recognition, tetapi hasil transkripnya terkadang sangat panjang, membingungkan, dan mungkin sulit dipahami. 
    Untuk mengatasi hal tersebut, kami berencana untuk mengembangkan sistem AI yang bisa mengubah suara menjadi teks (speech-to-text) dan membuat ringkasan otomatis dari hasil transkrip tersebut (text summarization). 
    **Final Project** ini diharapkan dapat meningkatkan aksesibilitas komunikasi dan pendidikan bagi penyandang gangguan pendengaran.
""", unsafe_allow_html=True)

st.write("### Pilih File Audio/Video atau URL Video untuk Diproses")
uploaded_file = st.file_uploader("Pilih file audio/video", type=["mp3", "mp4", "wav", "mkv"])

st.write("Atau, Anda dapat memberikan URL video (misalnya YouTube) untuk diproses.")
video_url = st.text_input("Masukkan URL Video (opsional)")

# biar hasil tetap ada saat rerun
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

# tombol proses (ini yang bikin alurnya rapi & nggak auto-run)
process_btn = st.button("Proses Transkripsi & Ringkasan", type="primary")

if process_btn:
    if uploaded_file is None and not video_url:
        st.warning("Silakan upload file atau masukkan URL terlebih dahulu.")
        st.stop()

    try:
        with st.spinner("Menyiapkan audio..."):
            if uploaded_file is not None:
                input_path = save_upload_to_tmp(uploaded_file)
                wav_path = run_ffmpeg_to_wav16k(input_path)
            else:
                wav_path = download_youtube_audio(video_url)

        with st.spinner("Melakukan transkripsi (Whisper finetuned)..."):
            transcript = whisper_transcribe(wav_path, processor, whisper_model)

        with st.spinner("Membuat ringkasan (Gemini)..."):
            summary = gemini_summarize(transcript)

        st.session_state.transcript = transcript
        st.session_state.summary = summary

        st.success("Selesai! Transkrip dan ringkasan sudah tersedia.")

    except Exception as e:
        st.error(f"Gagal memproses: {e}")

# tampilkan hasil (menggantikan placeholder)
st.write("### Transkrip:")
st.text_area(
    "Transkrip",
    value=st.session_state.transcript or "Transkrip akan ditampilkan di sini setelah diproses.",
    height=200
)

st.write("### Ringkasan:")
st.text_area(
    "Ringkasan",
    value=st.session_state.summary or "Ringkasan akan ditampilkan di sini setelah diproses.",
    height=200
)

st.download_button(
    label="Unduh Transkrip",
    data=st.session_state.transcript or "",
    file_name="transcript.txt",
    mime="text/plain",
    use_container_width=True
)

st.download_button(
    label="Unduh Ringkasan",
    data=st.session_state.summary or "",
    file_name="summary.txt",
    mime="text/plain",
    use_container_width=True
)
