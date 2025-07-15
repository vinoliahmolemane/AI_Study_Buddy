import os
import sqlite3
from datetime import datetime
import tempfile
import requests
from dotenv import load_dotenv
import cohere

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("Please set your COHERE_API_KEY environment variable.")
if not ASSEMBLYAI_API_KEY:
    raise ValueError("Please set your ASSEMBLYAI_API_KEY environment variable.")

co = cohere.Client(COHERE_API_KEY)

knowledge_embeddings = []
knowledge_texts = []

def generate_response(prompt: str) -> str:
    try:
        response = co.chat(
            model="command",
            message=prompt,
            temperature=0.7,
            max_tokens=300,
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

def save_chat_log(user_text: str, bot_response: str):
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/session_logs.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            bot_response TEXT
        )
    """)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO chat_logs (timestamp, user_input, bot_response) VALUES (?, ?, ?)",
              (timestamp, user_text, bot_response))
    conn.commit()
    conn.close()

def load_knowledge_base_from_text(text: str):
    import numpy as np
    try:
        response = co.embed(
            texts=[text],
            model="small",
            truncate="RIGHT"
        )
        embedding = response.embeddings[0]
        knowledge_embeddings.append(embedding)
        knowledge_texts.append(text)
        return True
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return False

def semantic_search(query: str, top_k=1):
    import numpy as np
    if not knowledge_embeddings:
        return []
    try:
        response = co.embed(
            texts=[query],
            model="small",
            truncate="RIGHT"
        )
        query_emb = np.array(response.embeddings[0])
    except Exception as e:
        print(f"Error embedding query: {e}")
        return []

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sims = [cosine_sim(query_emb, np.array(doc_emb)) for doc_emb in knowledge_embeddings]
    top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
    return [knowledge_texts[i] for i in top_indices]

def has_knowledge():
    return len(knowledge_texts) > 0

def get_knowledge_texts():
    return knowledge_texts

def transcribe_audio(audio_file) -> str:
    headers = {"authorization": ASSEMBLYAI_API_KEY}

    if hasattr(audio_file, "read"):
        upload_response = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers=headers,
            files={"file": audio_file}
        )
    elif isinstance(audio_file, str):
        with open(audio_file, "rb") as f:
            upload_response = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers=headers,
                files={"file": f}
            )
    else:
        raise ValueError("Invalid audio file input.")

    audio_url = upload_response.json()["upload_url"]

    transcript_response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=headers,
        json={"audio_url": audio_url}
    )
    transcript_id = transcript_response.json()["id"]
    polling_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    while True:
        status_response = requests.get(polling_url, headers=headers).json()
        if status_response["status"] == "completed":
            return status_response["text"]
        elif status_response["status"] == "error":
            raise Exception(f"AssemblyAI error: {status_response['error']}")

def text_to_speech(text: str, output_format="mp3") -> str:
    from gtts import gTTS
    tts = gTTS(text)
    ext = output_format.lower()
    temp_dir = tempfile.gettempdir()

    if ext == "mp3":
        out_path = os.path.join(temp_dir, "tts_output.mp3")
        tts.save(out_path)
    elif ext == "wav":
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("pydub required for WAV conversion. Install with: pip install pydub")

        mp3_path = os.path.join(temp_dir, "tts_output.mp3")
        wav_path = os.path.join(temp_dir, "tts_output.wav")
        tts.save(mp3_path)
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(wav_path, format="wav")
        out_path = wav_path
    else:
        raise ValueError("Unsupported output format. Use 'mp3' or 'wav'.")

    return out_path

def get_chat_logs():
    conn = sqlite3.connect("data/session_logs.db")
    c = conn.cursor()
    c.execute("SELECT timestamp, user_input, bot_response FROM chat_logs ORDER BY id DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    return rows
