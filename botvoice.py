# Voice-enabled AI Interview Coach with Post-Session Feedback

import os
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
import wave

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
def get_voice_input(save_path="user_response.wav"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        audio = recognizer.listen(source)

    # Save audio
    with open(save_path, "wb") as f:
        f.write(audio.get_wav_data())

    try:
        text = recognizer.recognize_google(audio)
        print(f"🗣️ You said: {text}")
        return text
    except sr.UnknownValueError:
        print("❗Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"❗Speech recognition error: {e}")
        return ""

# Voice output
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Report generation
def generate_interview_report(audio_path, chat_history):
    print("\n📋 Generating Full Interview Feedback Report...")

    qas = []
    last_human = None
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            last_human = msg.content
        elif isinstance(msg, AIMessage) and last_human:
            qas.append((msg.content, last_human))
            last_human = None

    if not qas:
        print("❗No Q&A pairs to analyze.")
        return

    tone_result = analyze_tone.invoke(audio_path)

    total_grammar_errors = 0
    grammar_feedback = []
    total_relevance_score = 0
    relevance_feedback = []

    for question, response in qas:
        grammar_result = analyze_grammar.invoke(response)
        total_grammar_errors += grammar_result.get("errors", 0)
        grammar_feedback.extend(grammar_result.get("feedback", []))

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

    avg_relevance = total_relevance_score / len(qas)
    report = {
        "Tone Analysis": tone_result,
        "Grammar Summary": {
            "total_errors": total_grammar_errors,
            "feedback_samples": grammar_feedback[:5]
        },
        "Relevance Summary": {
            "average_score": avg_relevance,
            "individual_feedback": relevance_feedback
        }
    }

    print("\n📝 Full Interview Performance Report:")
    from pprint import pprint
    pprint(report)

# Chat loop
print("🧠 Voice Interview Coach is ready! Say something (or 'exit' to finish):")
from langchain_core.messages import SystemMessage

chat_history = [
    SystemMessage(content="You are an AI interview coach assistant. Conduct a professional mock interview, asking one question at a time. Wait for the user's response before continuing.")
]


while True:
    user_input = get_voice_input("response.wav")
    if not user_input:
        continue
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("👋 Ending session...")
        break

    chat_history.append(HumanMessage(content=user_input))
    state = {"messgaes": chat_history}
    result = graph.invoke(state)

    chat_history = result["messgaes"]
    last_message = chat_history[-1]
    print(f"🤖 AI: {last_message.content}")
    speak_text(last_message.content)

# After session ends, generate feedback
generate_interview_report("response.wav", chat_history)
