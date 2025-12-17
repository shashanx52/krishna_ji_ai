🕉️ Krishna Ji – Voice-first AI Spiritual Companion

This project is a voice-first AI proof-of-concept where users can speak naturally and receive calm, contextual guidance inspired by the Bhagavad Gita.
It was built as part of the PsyTech AI Engineer Intern case study.

The focus of this project is natural voice interaction, safety-aware AI design, and content understanding, rather than building a full production system.

✨ What does this app do?

The user speaks their thoughts using a microphone

The system converts speech to text

The text is analyzed for:

sentiment (positive / neutral / negative)

harmful or offensive language

The content is automatically tagged into a life domain such as:

Career

Love life

Family

Health

Mood / emotional state

Based on this, the system generates a Krishna-style response

The response is finally returned as audio

The entire flow is designed to feel natural and conversational.

🧠 Why this approach?

Instead of training custom models (which requires data and time), I used:

pre-trained NLP models

zero-shot classification

clear fallback logic

This makes the system:

fast to build

easy to explain

reliable for a demo

realistic for an internship-level PoC

🏗️ High-level Architecture
Voice Input
   ↓
Whisper (Speech-to-Text)
   ↓
Language Translation (if required)
   ↓
Content Moderation
   ↓
Devotional Theme Classification
   ↓
Response Generation (Gemini / Fallback)
   ↓
Text-to-Speech
   ↓
Audio Reply

🛡️ Safety & Moderation

Before generating any response, the system checks:

Sentiment using VADER

Toxicity / offensive language using a transformer-based classifier

If the input is offensive, the system avoids generating advice and instead responds with a calm de-escalation message.

This safety-first step was intentionally placed before response generation.

🏷️ Devotional Tagging

User queries are automatically classified into problem categories using zero-shot classification (no training data required).

Supported categories:

Career

Love Life

Family Issues

Health Issues

Mood / Mental State

Each prediction includes a confidence score, which can later be used for analytics or dashboards.

🧘 Response Generation Logic

Primary: Gemini API (context-aware, Krishna-style responses)

Fallback: Predefined but diverse responses based on the detected theme

This hybrid approach ensures the app continues to work even if the external API fails, which is important for real-world systems.

🔊 Voice Output

Responses are converted back into speech using text-to-speech so that the interaction remains voice-first, as required by the problem statement.

🛠️ Tech Stack

Python 3.10

Streamlit (UI)

Whisper (Speech-to-Text)

Transformers (Toxicity & Zero-shot classification)

VADER (Sentiment analysis)

Gemini API (Response generation)

gTTS (Text-to-Speech)

⚙️ How to run this project locally
1. Clone the repository
git clone https://github.com/your-username/krishna-voice-ai.git
cd krishna-voice-ai

2. Create a virtual environment
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Set Gemini API key

On Windows:

setx GEMINI_API_KEY "YOUR_API_KEY_HERE"


Restart the terminal after setting the key.

5. Run the application
streamlit run app.py

⚠️ Notes & Limitations

This is a PoC, not a production system

Gemini API may occasionally fail; fallback logic is implemented

First run may take time due to model downloads

Requires a working microphone

🎯 What this project demonstrates

Voice-first AI system design

NLP pipeline thinking

Safety-aware response handling

Practical use of pre-trained models

Clean fallback strategies

Ability to balance ambition with stability

👤 Author

Shashank Jha
B.Tech (ECE), GGSIPU
AI / NLP / Data Enthusiast

Built as part of an internship evaluation assignment.
