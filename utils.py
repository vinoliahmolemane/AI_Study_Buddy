import os
import sqlite3
import requests
from datetime import datetime
from dotenv import load_dotenv
import cohere
import pyttsx3
import tempfile

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("❌ Please set your COHERE_API_KEY environment variable.")
if not ASSEMBLYAI_API_KEY:
    raise ValueError("❌ Please set your ASSEMBLYAI_API_KEY environment variable.")

co = cohere.Client(COHERE_API_KEY)


knowledge_texts = [
    "Artificial Intelligence (AI) is a branch of computer science that aims to create machines capable of intelligent behavior. AI systems can learn from data, recognize patterns, and make decisions.",
    
    "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to find patterns in data.",
    
    "Natural Language Processing (NLP) allows computers to understand, interpret, and generate human language. Applications include chatbots, translation, and sentiment analysis.",
    
    "Deep Learning is a specialized form of machine learning using neural networks with many layers. It is highly effective for tasks like image recognition and speech processing.",
    
    "Data Science combines statistics, computer science, and domain knowledge to extract insights from data. It involves data cleaning, visualization, and predictive modeling.",
    
    "Ethics in AI is critical to ensure that AI systems are fair, transparent, and do not reinforce biases. Responsible AI development requires ongoing monitoring and regulation.",
    
    "Cloud computing provides scalable resources and infrastructure for deploying AI models and handling large datasets in a flexible, cost-efficient manner."
]


def transcribe_audio(audio_file):
    """Transcribe audio using AssemblyAI."""
    try:
        headers = {'authorization': ASSEMBLYAI_API_KEY}
        upload_url = 'https://api.assemblyai.com/v2/upload'

        # Upload audio file
        with open(audio_file, 'rb') as f:
            response = requests.post(upload_url, headers=headers, files={'file': f})
        audio_url = response.json()['upload_url']

        # Start transcription
        transcript_request = {
            'audio_url': audio_url,
            'language_code': 'en',
            'auto_chapters': False
        }
        transcript_response = requests.post(
            'https://api.assemblyai.com/v2/transcript',
            json=transcript_request,
            headers=headers
        )
        transcript_id = transcript_response.json()['id']

        # Polling for completion
        polling_url = f'https://api.assemblyai.com/v2/transcript/{transcript_id}'
        while True:
            poll_response = requests.get(polling_url, headers=headers).json()
            if poll_response['status'] == 'completed':
                return poll_response['text']
            elif poll_response['status'] == 'error':
                raise Exception(poll_response['error'])
    except Exception as e:
        return f"❌ Error during transcription: {e}"


def generate_response(user_input: str) -> str:
    """Generate a response using Cohere's 'command' model."""
    try:
        response = co.generate(
            model="command",
            prompt=user_input,
            max_tokens=300,
            temperature=0.7
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"❌ Error generating response: {e}"


def text_to_speech_pyttsx3(text: str, output_file: str = None):
    """Convert text to speech using pyttsx3 and save as .wav."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 0.9)

        if not output_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                output_file = temp_file.name

        engine.save_to_file(text, output_file)
        engine.runAndWait()
        return output_file
    except Exception as e:
        return f"❌ Error generating audio: {e}"


def save_chat_log(user_input: str, response: str, db_path: str = "chat_log.db"):
    """Save user input and AI response to a local SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS chat_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                question TEXT,
                answer TEXT
            )
        ''')
        c.execute('''
            INSERT INTO chat_log (timestamp, question, answer)
            VALUES (?, ?, ?)
        ''', (datetime.now().isoformat(), user_input, response))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Error saving chat log: {e}")


def load_knowledge_base(file_path: str):
    """
    Load knowledge base text from file.
    If file is missing or empty, return the built-in knowledge_texts joined as string.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                return content
    
    # Fallback: join built-in knowledge_texts
    return "\n\n".join(knowledge_texts)


def generate_custom_content(topic: str) -> str:
    """Generate educational content for a specific topic using Cohere."""
    prompt = (
        f"Write an easy-to-understand educational explanation about the topic: {topic}.\n"
        f"Include examples where helpful. Keep it clear and informative."
    )
    return generate_response(prompt)
