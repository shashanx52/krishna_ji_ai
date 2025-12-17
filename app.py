import streamlit as st
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
import uuid
import random

from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from gtts import gTTS

import google.generativeai as genai

# ===================== PAGE CONFIG =====================

st.set_page_config(
    page_title="Krishna Ji – AI Spiritual Companion",
    page_icon="🕉️",
    layout="wide"
)

# ===================== HEADER =====================

st.markdown("""
<div style="text-align:center">
    <h1>🕉️ Krishna Ji – Voice-first AI Companion</h1>
    <p style="font-size:17px;">
    अपने मन की बात कहिए — शांति और मार्गदर्शन पाइए
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ===================== GEMINI SETUP (KEPT) =====================

genai.configure(api_key=os.getenv("AIzaSyAzg2YsmS24doBMQWZrUgZIFZoJBy5B8eU"))

@st.cache_resource
def load_gemini():
    return genai.GenerativeModel("gemini-1.5-flash")

gemini_model = load_gemini()

# ===================== LOAD MODELS =====================

@st.cache_resource
def load_whisper():
    return whisper.load_model("small")

@st.cache_resource
def load_toxicity_model():
    return pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        top_k=None
    )

@st.cache_resource
def load_tagging_model():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

whisper_model = load_whisper()
toxicity_classifier = load_toxicity_model()
tagging_classifier = load_tagging_model()
sentiment_analyzer = SentimentIntensityAnalyzer()

# ===================== TRANSLATION =====================

def translate_to_english(text):
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

# ===================== MODERATION =====================

def analyze_moderation(text):
    english_text = translate_to_english(text)

    scores = sentiment_analyzer.polarity_scores(english_text)
    compound = scores["compound"]

    sentiment = (
        "Positive" if compound >= 0.05
        else "Negative" if compound <= -0.05
        else "Neutral"
    )

    tox_results = toxicity_classifier(english_text)
    toxic_score = max(
        [r["score"] for r in tox_results[0] if r["label"].lower() == "toxic"],
        default=0.0
    )

    return {
        "sentiment": sentiment,
        "sentiment_confidence": round(max(scores["pos"], scores["neg"]), 2),
        "toxicity": "Offensive" if toxic_score >= 0.5 else "Safe",
        "toxicity_confidence": round(toxic_score, 2)
    }

# ===================== DEVOTIONAL TAGGING =====================

DEVOTIONAL_THEMES = [
    "Career and work related problems",
    "Love life and relationships",
    "Family and personal relationships",
    "Health related concerns",
    "Mental state, stress, or mood issues"
]

LABEL_MAP = {
    "Career and work related problems": "Career",
    "Love life and relationships": "Love Life",
    "Family and personal relationships": "Family",
    "Health related concerns": "Health",
    "Mental state, stress, or mood issues": "Mood"
}

def classify_devotional_theme(text):
    result = tagging_classifier(text, DEVOTIONAL_THEMES)
    return {
        "theme": LABEL_MAP[result["labels"][0]],
        "confidence": round(result["scores"][0], 2)
    }

# ===================== FALLBACK KRISHNA RESPONSES =====================

KRISHNA_RESPONSES = {
    "Career": [
        "कर्म पर ध्यान दो, परिणाम अपने समय पर आएगा।",
        "परिश्रम कभी व्यर्थ नहीं जाता।",
        "अपने कर्तव्य से मत डरो।"
    ],
    "Love Life": [
        "संतुलन और समझ से ही संबंध टिकते हैं।",
        "भावनाओं में धैर्य रखो।",
        "जो सत्य है वही स्थायी है।"
    ],
    "Family": [
        "परिवार में संवाद सबसे बड़ा समाधान है।",
        "अपनों को समय देना भी धर्म है।",
        "धैर्य से ही रिश्ते मजबूत होते हैं।"
    ],
    "Health": [
        "स्वास्थ्य शरीर और मन दोनों का होता है।",
        "संयम से जीवन संतुलित रहता है।",
        "अपने शरीर की सुनो।"
    ],
    "Mood": [
        "यह समय भी बीत जाएगा।",
        "मन की शांति भीतर से आती है।",
        "स्वयं पर विश्वास रखो।"
    ]
}

# ===================== GEMINI + FALLBACK =====================

def gemini_krishna_reply(user_text, theme):
    prompt = f"""
You are Lord Krishna.
Reply in Hindi or Hinglish.
Be calm, compassionate, and contextual.
Avoid repeating phrases.
Base guidance on Bhagavad Gita.

Theme: {theme}
User: {user_text}

Krishna:
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def get_final_reply(user_text, moderation, theme):
    if moderation["toxicity"] == "Offensive":
        return "क्रोध से विवेक नष्ट होता है। शांत होकर अपनी बात कहो।"

    try:
        return gemini_krishna_reply(user_text, theme)
    except Exception:
        return random.choice(KRISHNA_RESPONSES.get(theme, ["शांति रखो।"]))

# ===================== TEXT TO SPEECH =====================

def speak_and_play(text):
    filename = f"krishna_reply_{uuid.uuid4().hex}.mp3"
    gTTS(text=text, lang="hi").save(filename)
    return filename

# ===================== VOICE INPUT =====================

st.markdown("## 🎙️ कृष्ण जी से बात करें")

DURATION = 6
SAMPLE_RATE = 44100

if st.button("🎧 बोलना शुरू करें"):
    with st.spinner("🎙️ सुन रहे हैं..."):
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16
        )
        sd.wait()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wav.write(f.name, SAMPLE_RATE, audio)
        audio_path = f.name

    with st.spinner("🕉️ कृष्ण जी के पास बात जा रही है..."):
        result = whisper_model.transcribe(audio_path)
        user_text = result["text"].strip()
        os.remove(audio_path)

    moderation = analyze_moderation(user_text)
    tagging = classify_devotional_theme(user_text)
    reply_text = get_final_reply(user_text, moderation, tagging["theme"])
    reply_audio = speak_and_play(reply_text)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🗣️ आपने कहा")
        st.write(user_text)

        st.subheader("🏷️ समस्या श्रेणी")
        st.json(tagging)

    with col2:
        st.subheader("🛡️ सुरक्षा विश्लेषण")
        st.json(moderation)

    st.subheader("🕉️ कृष्ण जी का उत्तर")
    st.write(reply_text)
    st.audio(reply_audio)

    os.remove(reply_audio)

# ===================== FOOTER =====================

st.markdown("""
<hr>
<div style="text-align:center; font-size:14px;">
PsyTech AI Engineer Intern – Proof of Concept<br>
Voice • NLP • Safety • Tagging • LLM • Audio
</div>
""", unsafe_allow_html=True)
