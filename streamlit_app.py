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
MODEL_PATH = "models/whisper-medium-finetuned-id"
LANG = "id"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CACHED LOADERS
# =========================
@st.cache_resource(show_spinner="Loading Whisper model...")
def load_whisper(model_path: str):
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource
def load_gemini():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# =========================
# HELPERS
# =========================
def save_upload(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def ffmpeg_to_wav16k(input_path: str) -> str:
    out_path = input_path + "_16k.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            out_path
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return out_path


def download_youtube_audio(url: str) -> str:
    tmpdir = tempfile.mkdtemp()
    outtmpl = os.path.join(tmpdir, "audio.%(ext)s")

    subprocess.run(
        ["yt-dlp", "-f", "bestaudio", "-o", outtmpl, url],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    files = os.listdir(tmpdir)
    if not files:
        raise RuntimeError("YouTube download failed.")

    return ffmpeg_to_wav16k(os.path.join(tmpdir, files[0]))


def whisper_transcribe(wav_path: str, processor, model) -> str:
    audio, _ = librosa.load(wav_path, sr=16000)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(DEVICE)

    forced_ids = processor.get_decoder_prompt_ids(
        language=LANG,
        task="transcribe"
    )

    with torch.no_grad():
        pred_ids = model.generate(
            **inputs,
            forced_decoder_ids=forced_ids,
            max_new_tokens=448
        )

    return processor.batch_decode(
        pred_ids,
        skip_special_tokens=True
    )[0].strip()


def gemini_summarize(text: str) -> str:
    client = load_gemini()
    model_name = st.secrets.get("GEMINI_MODEL", "gemini-2.0-flash-001")

    prompt = f"""
Buat ringkasan Bahasa Indonesia dari transkrip berikut.
Gunakan 5–8 bullet point yang ringkas dan jelas.

Transkrip:
{text}
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=400
        )
    )

    return (response.text or "").strip()


# =========================
# LOAD MODELS
# =========================
processor, whisper_model = load_whisper(MODEL_PATH)

# =========================
# UI
# =========================
st.title("HEARity – Speech-to-Text & Summarization")

st.markdown("""
HEARity membantu penyandang gangguan pendengaran dengan:
- 🎧 Transkripsi audio/video (Whisper finetuned)
- 🧠 Ringkasan otomatis (Gemini)
""")

uploaded_file = st.file_uploader(
    "Upload audio / video",
    type=["wav", "mp3", "mp4", "mkv"]
)

video_url = st.text_input(
    "Atau masukkan URL YouTube (opsional)"
)

if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

if st.button("Proses", type="primary"):
    if not uploaded_file and not video_url:
        st.warning("Upload file atau masukkan URL terlebih dahulu.")
        st.stop()

    try:
        with st.spinner("Menyiapkan audio..."):
            if uploaded_file:
                path = save_upload(uploaded_file)
                wav_path = ffmpeg_to_wav16k(path)
            else:
                wav_path = download_youtube_audio(video_url)

        with st.spinner("Transkripsi Whisper..."):
            st.session_state.transcript = whisper_transcribe(
                wav_path,
                processor,
                whisper_model
            )

        with st.spinner("Membuat ringkasan Gemini..."):
            st.session_state.summary = gemini_summarize(
                st.session_state.transcript
            )

        st.success("Selesai!")

    except Exception as e:
        st.error(f"Error: {e}")

st.subheader("Transkrip")
st.text_area("", st.session_state.transcript, height=220)

st.subheader("Ringkasan")
st.text_area("", st.session_state.summary, height=220)

st.download_button(
    "Download Transkrip",
    st.session_state.transcript,
    "transcript.txt"
)

st.download_button(
    "Download Ringkasan",
    st.session_state.summary,
    "summary.txt"
)
