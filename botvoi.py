# Voice-enabled AI Interview Coach with Post-Session Feedback

import os
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
import wave
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', message='FP16 is not supported on CPU')
warnings.filterwarnings('ignore', category=UserWarning, module='whisper')
warnings.filterwarnings('ignore', category=FutureWarning)

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated

from tools import analyze_tone, analyze_grammar, analyze_relevance

# Load API Key
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# LLM Setup
llm = ChatGroq(model="llama3-8b-8192")

# State type
class State(TypedDict):
    messgaes: Annotated[list[HumanMessage | AIMessage], add_messages]

# Node function
def simple_llm_response(state: State):
    response = llm.invoke(state["messgaes"])
    return {"messgaes": state["messgaes"] + [response]}

# Graph build
builder = StateGraph(State)
builder.add_node("llm_response", simple_llm_response)
builder.set_entry_point("llm_response")
builder.add_edge("llm_response", END)
graph = builder.compile()

# Voice recognition and saving
import whisper
import sounddevice as sd
import scipy.io.wavfile as wavfile

# Initialize Whisper model
model = whisper.load_model("base", device="cpu")  # Will automatically use FP32 on CPU

def get_voice_input(audio_path):
    """Process an audio file and return transcription"""
    try:
        result = model.transcribe(audio_path)
        transcription = result["text"].strip()
        return transcription
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return ""

# Voice output
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Report generation
def generate_interview_report(audio_path, chat_history):
    """Generate a comprehensive interview report"""
    try:
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            return {
                "Tone_Analysis": {
                    "pitch": 0.5,
                    "intensity": 0.5,
                    "feedback": "Audio file not found"
                },
                "Grammar_Summary": {
                    "total_errors": 0,
                    "feedback_samples": ["No audio available for analysis"]
                },
                "Relevance_Summary": {
                    "average_score": 0.0,
                    "individual_feedback": []
                }
            }

        qas = []
        last_human = None
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                last_human = msg.content
            elif isinstance(msg, AIMessage) and last_human:
                qas.append((msg.content, last_human))
                last_human = None

        if not qas:
            print("No Q&A pairs found in chat history")
            return {
                "Tone_Analysis": {
                    "pitch": 0.5,
                    "intensity": 0.5,
                    "feedback": "No conversation data available"
                },
                "Grammar_Summary": {
                    "total_errors": 0,
                    "feedback_samples": ["No conversation data available"]
                },
                "Relevance_Summary": {
                    "average_score": 0.0,
                    "individual_feedback": []
                }
            }

        try:
            tone_result = analyze_tone.invoke(audio_path)
        except Exception as e:
            print(f"Error in tone analysis: {str(e)}")
            tone_result = {
                "pitch": 0.5,
                "intensity": 0.5,
                "feedback": f"Tone analysis failed: {str(e)}"
            }

        total_grammar_errors = 0
        grammar_feedback = []
        total_relevance_score = 0
        relevance_feedback = []

        for question, response in qas:
            try:
                grammar_result = analyze_grammar.invoke(response)
                total_grammar_errors += grammar_result.get("errors", 0)
                grammar_feedback.extend(grammar_result.get("feedback", []))
            except Exception as e:
                print(f"Error in grammar analysis: {str(e)}")
                grammar_feedback.append(f"Grammar analysis failed: {str(e)}")

            try:
                relevance_result = analyze_relevance.invoke({
                    "transcription": response,
                    "question": question
                })
                total_relevance_score += relevance_result.get("score", 0.0)
                relevance_feedback.append({
                    "question": question,
                    "score": relevance_result.get("score", 0.0),
                    "feedback": relevance_result.get("feedback", "")
                })
            except Exception as e:
                print(f"Error in relevance analysis: {str(e)}")
                relevance_feedback.append({
                    "question": question,
                    "score": 0.0,
                    "feedback": f"Relevance analysis failed: {str(e)}"
                })

        avg_relevance = total_relevance_score / len(qas) if qas else 0
        report = {
            "Tone_Analysis": tone_result,
            "Grammar_Summary": {
                "total_errors": total_grammar_errors,
                "feedback_samples": grammar_feedback[:5]
            },
            "Relevance_Summary": {
                "average_score": avg_relevance,
                "individual_feedback": relevance_feedback
            }
        }

        return report
    except Exception as e:
        print(f"Error generating interview report: {str(e)}")
        return {
            "Tone_Analysis": {
                "pitch": 0.5,
                "intensity": 0.5,
                "feedback": f"Report generation failed: {str(e)}"
            },
            "Grammar_Summary": {
                "total_errors": 0,
                "feedback_samples": [f"Report generation failed: {str(e)}"]
            },
            "Relevance_Summary": {
                "average_score": 0.0,
                "individual_feedback": []
            }
        }
