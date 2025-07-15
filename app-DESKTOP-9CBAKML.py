import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import numpy as np
import soundfile as sf
import io
import os
from datetime import datetime
import sqlite3
import PyPDF2

from utils import (
    transcribe_audio,
    generate_response,
    text_to_speech,
    save_chat_log,
    get_chat_logs,
    load_knowledge_base_from_text,
    semantic_search,
)

# Global knowledge storage
knowledge_texts = []

def has_knowledge():
    return len(knowledge_texts) > 0

def get_knowledge_texts():
    return knowledge_texts

st.set_page_config(page_title="AI Capstone Project", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Project", ["Home", "AI Study Buddy", "Accessibility Tool", "Custom Project"])

# -------------------
# AI Study Buddy page
# -------------------
def ai_study_buddy():
    st.title("🎓 AI Study Buddy")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    mode = st.radio("Input Mode", ["📝 Text", "🎙️ Upload Audio", "🎤 Record Audio"])

    if mode == "📝 Text":
        user_input = st.text_input("Ask a question or type a command")
        if st.button("Submit") and user_input:
            response = generate_response(user_input)
            if response:
                st.session_state.chat_history.append({
                    "user": user_input,
                    "bot": response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                save_chat_log(user_input, response)

    elif mode == "🎙️ Upload Audio":
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])
        if uploaded_file:
            if st.button("Transcribe & Ask"):
                try:
                    transcribed_text = transcribe_audio(uploaded_file)
                    st.write(f"**Transcribed Text:** {transcribed_text}")
                    response = generate_response(transcribed_text)
                    if response:
                        st.session_state.chat_history.append({
                            "user": transcribed_text,
                            "bot": response,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        save_chat_log(transcribed_text, response)

                        mp3_file = text_to_speech(response, output_format="mp3")
                        with open(mp3_file, "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format="audio/mp3")
                        st.download_button("Download MP3 Response", data=audio_bytes, file_name="response_audio.mp3", mime="audio/mp3")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Upload an audio file to transcribe and get a response.")

    elif mode == "🎤 Record Audio":
        class AudioProcessor(AudioProcessorBase):
            def __init__(self):
                self.frames = []

            def recv(self, frame):
                self.frames.append(frame.to_ndarray(format="flt32"))
                return frame

        webrtc_ctx = webrtc_streamer(
            key="audio-recorder",
            mode="recvonly",
            audio_receiver_size=1024,
            media_stream_constraints={"audio": True, "video": False},
            processor_factory=AudioProcessor,
        )

        if webrtc_ctx.audio_receiver:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            if audio_frames:
                audio_data = np.concatenate([f.to_ndarray(format="flt32") for f in audio_frames])
                sample_rate = 16000

                buf = io.BytesIO()
                sf.write(buf, audio_data, sample_rate, format="wav")
                buf.seek(0)

                st.audio(buf.read(), format="audio/wav")

                os.makedirs("audio", exist_ok=True)
                wav_path = "audio/recorded_audio.wav"
                sf.write(wav_path, audio_data, sample_rate)

                with open(wav_path, "rb") as f:
                    st.download_button("Download Recorded WAV", data=f, file_name="recorded_audio.wav", mime="audio/wav")

                with open(wav_path, "rb") as f:
                    transcribed_text = transcribe_audio(f)

                st.write(f"**Transcribed Text:** {transcribed_text}")

                if st.button("Ask AI"):
                    response = generate_response(transcribed_text)
                    if response:
                        st.session_state.chat_history.append({
                            "user": transcribed_text,
                            "bot": response,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        save_chat_log(transcribed_text, response)

                        wav_file = text_to_speech(response, output_format="wav")
                        with open(wav_file, "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format="audio/wav")
                        st.download_button("Download WAV Response", data=audio_bytes, file_name="ai_response.wav", mime="audio/wav")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []

    st.subheader("Chat History")
    for entry in st.session_state.chat_history:
        st.markdown(f"**You:** {entry['user']}")
        st.markdown(f"**AI:** {entry['bot']}")
        st.markdown(f"*{entry['timestamp']}*")
        st.markdown("---")

    st.subheader("Recent Chat Logs")
    logs = get_chat_logs()
    for ts, user_in, bot_out in logs:
        st.markdown(f"**[{ts}] You:** {user_in}")
        st.markdown(f"**[{ts}] AI:** {bot_out}")
        st.markdown("---")

# -----------------------
# Accessibility Tool page
# -----------------------
def accessibility_ui():
    st.title("♿ Accessibility Tool")

    st.write("Use speech-to-text and text-to-speech features for accessibility.")

    audio_file = st.file_uploader("Upload audio for transcription", type=["wav", "mp3", "m4a", "ogg", "flac"])
    if audio_file:
        try:
            transcription = transcribe_audio(audio_file)
            st.write(f"**Transcribed text:** {transcription}")
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")

    text_input = st.text_area("Enter text to convert to speech")
    if st.button("Convert to Speech") and text_input.strip():
        try:
            audio_path = text_to_speech(text_input, output_format="mp3")
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button("Download Speech Audio", data=audio_bytes, file_name="speech.mp3", mime="audio/mp3")
        except Exception as e:
            st.error(f"Error generating speech: {e}")

# --------------------
# Custom Project page
# --------------------
def custom_project():
    global knowledge_texts

    st.title("🛠️ Custom Project: Domain-Specific Chatbot")

    st.markdown("Upload domain knowledge files (TXT or PDF). The bot will answer questions based on these.")

    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=["txt", "pdf"])

    if uploaded_files:
        count = 0
        for f in uploaded_files:
            if f.type == "text/plain":
                content = f.read().decode("utf-8")
            elif f.type == "application/pdf":
                pdf = PyPDF2.PdfReader(f)
                content = ""
                for page in pdf.pages:
                    content += page.extract_text()
            else:
                st.error("Unsupported file type. Please upload txt or pdf.")
                content = ""
            if content:
                success = load_knowledge_base_from_text(content)
                if success:
                    count += 1
        st.success(f"Loaded {count} document(s) into the knowledge base.")

    query = st.text_input("Ask a question about your domain knowledge")

    if st.button("Get Answer"):
        if not query.strip():
            st.warning("Please enter a question.")
        elif not knowledge_texts:
            st.warning("Upload domain knowledge files first.")
        else:
            matches = semantic_search(query, top_k=1)
            if matches:
                prompt = f"Answer the question based on the following context:\n\n{matches[0]}\n\nQuestion: {query}"
                answer = generate_response(prompt)
                st.markdown(f"**Answer:** {answer}")
                save_chat_log(query, answer)
            else:
                st.info("No matching info found. Try uploading more documents or rephrasing your question.")

    if st.checkbox("Show loaded knowledge excerpts"):
        for i, doc in enumerate(knowledge_texts, 1):
            st.markdown(f"**Doc {i}:**")
            st.write(doc[:1000] + ("..." if len(doc) > 1000 else ""))

# --------------
# Home page
# --------------
def home():
    st.title("🏠 Welcome to AI Capstone Project")
    st.write("""
    Select a project from the sidebar to get started:
    - AI Study Buddy: Chat with domain knowledge + speech/audio input.
    - Accessibility Tool: Speech-to-text and text-to-speech features.
    - Custom Project: Upload documents and ask domain-specific questions.
    """)

# --------------------
# Main app control flow
# --------------------
if page == "Home":
    home()
elif page == "AI Study Buddy":
    ai_study_buddy()
elif page == "Accessibility Tool":
    accessibility_ui()
elif page == "Custom Project":
    custom_project()
