import librosa
import numpy as np
from sentence_transformers import SentenceTransformer, util
import language_tool_python
from langchain_core.tools import tool
import soundfile as sf
import warnings
import os

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', message='PySoundFile failed')
warnings.filterwarnings('ignore', message='.*audioread.*')
warnings.filterwarnings('ignore', category=UserWarning, module='soundfile')

def load_audio_file(audio_file):
    """Load audio file with multiple fallback options"""
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
    try:
        # Try loading with soundfile first
        y, sr = sf.read(audio_file)
        y = y.astype(np.float32)
        return y, sr
    except Exception as e1:
        try:
            # Try loading with librosa
            y, sr = librosa.load(audio_file, sr=None)
            return y, sr
        except Exception as e2:
            raise Exception(f"Failed to load audio file. SoundFile error: {str(e1)}, Librosa error: {str(e2)}")

class AnalysisTools:
    def __init__(self):
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
        self.ideal_answers = {
            "Tell me about yourself.": "A concise summary of your background, skills, and goals relevant to the role."
        }

@tool
def analyze_tone(audio_file: str) -> dict:
    """Analyze audio for tone metrics like pitch and intensity."""
    try:
        # Load audio file with fallback options
        y, sr = load_audio_file(audio_file)
        
        # Calculate pitch using librosa's pitch tracking
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = float(np.mean(pitches[magnitudes > np.median(magnitudes)]))
        
        # Calculate intensity using RMS energy
        intensity = float(np.mean(librosa.feature.rms(y=y)))
        
        # Normalize values to 0-1 range
        pitch = min(max(pitch / 1000, 0), 1)  # Assuming pitch range of 0-1000 Hz
        intensity = min(max(intensity / 0.5, 0), 1)  # Assuming max intensity of 0.5
        
        return {
            "pitch": pitch,
            "intensity": intensity,
            "feedback": "Good tone" if 0.1 < pitch < 0.5 else "Adjust pacing for clarity"
        }
    except Exception as e:
        print(f"Error in tone analysis: {str(e)}")
        return {
            "pitch": 0.5,
            "intensity": 0.5,
            "feedback": f"Tone analysis failed: {str(e)}"
        }

@tool
def analyze_relevance(transcription: str, question: str) -> dict:
    """Score response relevance against an ideal answer."""
    try:
        tools = AnalysisTools()
        ideal_answer = tools.ideal_answers.get(question, "")
        if not ideal_answer:
            return {"score": 0.0, "feedback": "No ideal answer defined"}
        embeddings = tools.st_model.encode([transcription, ideal_answer])
        score = float(util.cos_sim(embeddings[0], embeddings[1]).item())
        return {
            "score": score,
            "feedback": "Highly relevant" if score > 0.7 else "Include more relevant details"
        }
    except Exception as e:
        print(f"Error in relevance analysis: {str(e)}")
        return {"score": 0.0, "feedback": f"Relevance analysis failed: {str(e)}"}

@tool
def analyze_grammar(transcription: str) -> dict:
    """Check transcription for grammatical errors."""
    try:
        tools = AnalysisTools()
        matches = tools.grammar_tool.check(transcription)
        return {
            "errors": len(matches),
            "feedback": [str(match) for match in matches] or ["No grammar issues detected"]
        }
    except Exception as e:
        return {"error": str(e), "feedback": "Grammar analysis failed"}