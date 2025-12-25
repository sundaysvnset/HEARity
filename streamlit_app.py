import streamlit as st

# Background color and theme
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
            color: #333;
        }
        .title {
            color: #4CAF50;
        }
        .header {
            color: #4CAF50;
            font-size: 24px;
            font-weight: bold;
        }
        .text {
            font-size: 18px;
            line-height: 1.6;
        }
        .button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Judul
st.title("HEARity: Speech-to-Text Summarization Berbasis Generative AI untuk Meningkatkan Aksesibilitas bagi Penyandang Gangguan Pendengaran", anchor="title")

# Deskripsi Aplikasi
st.write("""
    Penyandang gangguan pendengaran seringkali mengalami kesulitan untuk memahami percakapan, perkuliahan, atau informasi berbasis suara lainnya. 
    Walaupun saat ini sudah ada teknologi speech recognition, tetapi hasil transkripnya terkadang sangat panjang, membingungkan, dan mungkin sulit dipahami. 
    Untuk mengatasi hal tersebut, kami berencana untuk mengembangkan sistem AI yang bisa mengubah suara menjadi teks (speech-to-text) dan membuat ringkasan otomatis dari hasil transkrip tersebut (text summarization). 
    **Final Project** ini diharapkan dapat meningkatkan aksesibilitas komunikasi dan pendidikan bagi penyandang gangguan pendengaran.
""", unsafe_allow_html=True)

# File uploader for audio/video file
st.write("### Pilih File Audio/Video atau URL Video untuk Diproses")
uploaded_file = st.file_uploader("Pilih file audio/video", type=["mp3", "mp4", "wav", "mkv"])

# Section for providing a video URL (if needed)
st.write("Atau, Anda dapat memberikan URL video (misalnya YouTube) untuk diproses.")

video_url = st.text_input("Masukkan URL Video (opsional)")

# Processing when file is uploaded
if uploaded_file is not None:
    st.write(f"File yang diunggah: {uploaded_file.name}")
    st.write("Proses transkripsi dan ringkasan sedang dilakukan...")

if video_url:
    st.write(f"URL video yang dimasukkan: {video_url}")
    st.write("Proses video URL sedang dilakukan...")

# Display a placeholder for the transcript and summary (once model is integrated)
st.write("### Transkrip:")
st.text_area("Transkrip akan ditampilkan di sini setelah diproses.", height=200)

st.write("### Ringkasan:")
st.text_area("Ringkasan akan ditampilkan di sini setelah diproses.", height=200)

# Download buttons (will be implemented later)
st.download_button(
    label="Unduh Transkrip",
    data="Transkrip akan tersedia di sini setelah diproses.",
    file_name="transcript.txt",
    mime="text/plain",
    use_container_width=True
)

st.download_button(
    label="Unduh Ringkasan",
    data="Ringkasan akan tersedia di sini setelah diproses.",
    file_name="summary.txt",
    mime="text/plain",
    use_container_width=True
)
