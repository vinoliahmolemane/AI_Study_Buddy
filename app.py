import streamlit as st
from datetime import datetime
import sqlite3
from io import StringIO
from PyPDF2 import PdfReader
from fpdf import FPDF
import tempfile

from utils import (
    transcribe_audio,
    generate_response,
    text_to_speech_pyttsx3,
    save_chat_log,
    load_knowledge_base,
    generate_custom_content
)

st.set_page_config(page_title="AI Study Buddy", layout="centered")

# Sidebar Navigation
page = st.sidebar.radio("🏠 Home", [
    "AI Study Buddy 🤖",
    "Accessibility Tool 🎧",
    "Custom Content Generator ✍️",
    "Chat History 📚"
])

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = load_knowledge_base("knowledge_base.txt")

# Helper: extract text from file
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        return stringio.read()
    elif uploaded_file.type == "application/pdf":
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        return text
    else:
        return None

# === AI Study Buddy Page ===
if page == "AI Study Buddy 🤖":
    st.title("🤖 AI Study Buddy")
    st.info("Ask questions by typing, and upload notes (TXT/PDF) to enrich the knowledge base.")

    notes_file = st.file_uploader("📄 Upload Notes (TXT or PDF):", type=["txt", "pdf"])
    if notes_file:
        text = extract_text_from_file(notes_file)
        if text:
            st.success("✅ Notes loaded successfully!")
            st.session_state.knowledge_base += "\n" + text
        else:
            st.error("❌ Could not extract text from file.")

    user_text = st.text_input("💬 Type your question here:", key="ai_study_buddy_user_text")

    if st.button("Ask AI"):
        question = user_text.strip()
        if not question:
            st.warning("⚠️ Please enter a question.")
        else:
            try:
                prompt = f"{st.session_state.knowledge_base}\n\nQuestion: {question}\nAnswer:"
                response = generate_response(prompt)
                st.session_state.history.append(("You", question))
                st.session_state.history.append(("AI", response))
                save_chat_log(question, response)
                st.rerun()  # ✅ FIXED
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.history = []
        st.rerun()  # ✅ FIXED

    if st.session_state.history:
        st.markdown("### 💬 Chat History")
        for speaker, message in st.session_state.history:
            if speaker == "You":
                st.markdown(
                    f"<div style='text-align: right; background-color: #DCF8C6; padding: 10px; "
                    f"border-radius: 10px; margin: 5px;'>{message}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align: left; background-color: #F1F0F0; padding: 10px; "
                    f"border-radius: 10px; margin: 5px;'>{message}</div>",
                    unsafe_allow_html=True,
                )

        if st.button("📄 Download Chat History as PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.cell(200, 10, txt="AI Study Buddy - Chat Log", ln=True, align="C")
            pdf.ln(5)

            for speaker, message in st.session_state.history:
                label = "You: " if speaker == "You" else "AI: "
                pdf.multi_cell(0, 10, txt=f"{label}{message}\n")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                pdf.output(tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button(
                        "📥 Click to Download PDF", data=f, file_name="chat_history.pdf"
                    )

# === Accessibility Tool ===
elif page == "Accessibility Tool 🎧":
    st.title("🎧 Accessibility Tool")
    st.info("Convert speech to text or text to speech.")

    st.subheader("🗣️ Speech to Text (WAV or MP3)")
    voice_file = st.file_uploader("Upload WAV or MP3 file:", type=["wav", "mp3"], key="voice_upload")
    if voice_file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(voice_file.read())
                temp_audio.flush()
                transcript = transcribe_audio(temp_audio.name)
            st.success("✅ Transcription Complete!")
            st.markdown(f"**Transcript:** {transcript}")
        except Exception as e:
            st.error(f"❌ Error during transcription: {e}")

    st.markdown("---")
    st.subheader("🔊 Text to Speech (Offline, WAV)")
    tts_text = st.text_area("Enter text to convert into speech:", height=150, key="accessibility_text_area")
    if st.button("Convert to WAV"):
        if not tts_text.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            try:
                audio_data = text_to_speech_pyttsx3(tts_text)
                st.audio(audio_data, format="audio/wav")
            except Exception as e:
                st.error(f"❌ Error generating audio: {e}")

# === Custom Content Generator ===
elif page == "Custom Content Generator ✍️":
    st.title("✍️ Custom Content Generator")
    st.info("Generate educational content using AI. Enter a topic below.")

    topic = st.text_input("📚 Enter a topic (e.g. Photosynthesis, Python loops, World War II):", key="content_generator_topic")
    if st.button("⚡ Generate Content"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            try:
                result = generate_custom_content(topic)
                st.markdown("### 📖 Generated Educational Content:")
                st.write(result)
            except Exception as e:
                st.error(f"❌ Error: {e}")

# === Chat History Page ===
elif page == "Chat History 📚":
    st.title("📚 Chat Log Viewer")

    try:
        conn = sqlite3.connect("chat_log.db")
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, question, answer FROM chat_log ORDER BY id DESC")
        rows = cursor.fetchall()

        if not rows:
            st.info("No chat logs found.")
        else:
            for ts, q, a in rows:
                st.markdown(
                    f"""
                    <div style='background-color: #FAFAFA; padding: 15px; margin-bottom: 10px; border-radius: 10px; box-shadow: 0 0 5px #DDD;'>
                        <small><b>🕒 {ts}</b></small><br>
                        <b>You:</b> {q}<br>
                        <b>AI:</b> {a}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        conn.close()
    except Exception as e:
        st.error(f"❌ Error reading chat log: {e}")
