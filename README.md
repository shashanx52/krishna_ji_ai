# 🕉️ Krishna Ji – Voice-first AI Spiritual Companion

Krishna Ji is a **voice-first AI proof-of-concept** that allows users to speak naturally and receive calm, contextual guidance inspired by the **Bhagavad Gita**.

This project was built as part of the **PsyTech AI Engineer Intern case study** and focuses on **AI system design**, not just model usage.

---

## 🌱 Project Motivation

The aim of this project is to explore how a **spiritual AI companion** can be designed responsibly using:
- voice interaction
- content understanding
- safety moderation
- intelligent tagging
- graceful fallbacks when AI services fail

The emphasis is on **clarity, safety, and explainability**, not on building a production-ready system.

---

## ✨ What Does the App Do?

1. User speaks using a microphone  
2. Speech is converted to text  
3. The text is analyzed for:
   - sentiment (positive / neutral / negative)
   - toxic or offensive language  
4. The content is **classified into a devotional problem category**
5. A Krishna-style response is generated
6. The response is returned as **audio**

All steps happen in real time.

---

## 🧠 High-Level Architecture

Voice Input
↓
Whisper (Speech-to-Text)
↓
Language Translation (if needed)
↓
Content Moderation (Sentiment + Toxicity)
↓
Devotional Tagging (Zero-shot NLP)
↓
Response Generation (Gemini / Fallback)
↓
Text-to-Speech
↓
Audio Output


---

## 🛡️ Safety & Moderation

Before generating any response, the system performs moderation:

- **Sentiment Analysis** using VADER  
- **Toxicity Detection** using a transformer-based classifier  

If the input is offensive, the system avoids advice generation and responds with a calm de-escalation message.

This safety check is intentionally placed **before** response generation.

---

## 🏷️ Devotional Tagging / Classification

User input is automatically classified into one of the following domains using **zero-shot classification**:

- Career  
- Love Life  
- Family Issues  
- Health Issues  
- Mood / Mental State  

Each classification includes a **confidence score**, making the decision transparent and explainable.

No training data is required for this step.

---

## 🧘 Response Generation Logic

- **Primary Path**:  
  Gemini API is used to generate contextual, Krishna-style responses.

- **Fallback Path**:  
  If Gemini is unavailable or fails, the system uses **predefined but diverse responses** based on the detected theme.

This hybrid approach ensures the app remains usable even when external APIs are unstable.

---

## 🔊 Voice Output

Responses are converted back into speech using text-to-speech so that the experience remains **voice-first**, as required by the case study.

---

## 🛠️ Tech Stack

- **Python 3.10**
- **Streamlit** – Web UI
- **Whisper** – Speech-to-Text
- **Transformers** – Toxicity detection & zero-shot classification
- **VADER** – Sentiment analysis
- **Gemini API** – Contextual response generation
- **gTTS** – Text-to-Speech

---


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/krishna-voice-ai.git
cd krishna-voice-ai

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set Gemini API Key
On Windows (Command Prompt):
setx GEMINI_API_KEY "YOUR_API_KEY_HERE"
Restart the terminal after setting the key.

Note: The code is written to safely handle Gemini failures using fallback logic.

5️⃣ Run the Application
streamlit run app.py

👤 Author

Shashank Jha
B.Tech (ECE), GGSIPU
AI / NLP / Data Enthusiast

