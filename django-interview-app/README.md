# Interview Emotion Assistant

This repo is now structured for a Django backend and a Next.js frontend.

## Structure

- `backend/` - Django REST API, Azure OpenAI interview flow, TTS, transcript saving, face emotion, voice emotion, fused mood, and recording uploads.
- `frontend/` - Next.js browser app. JavaScript asks for camera and microphone access, records the interview, uses browser speech recognition, sends video frames/audio samples to Django, and plays Django-generated TTS.
- The original `voicefacebot.py` remains in the parent folder for reference.

## Local Run

1. Copy `.env.example` to `.env` and set Azure OpenAI values.
2. Start the backend in one terminal.
3. Start the frontend in another terminal.
4. Open `http://localhost:3000`.

```powershell
copy .env.example .env
```

## Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

The Django API runs at `http://localhost:8000/api`.

## Frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The Next.js app runs at `http://localhost:3000`.

## Feature Mapping

- Browser camera/microphone permissions: Next.js via `getUserMedia`.
- Speech-to-text: browser `SpeechRecognition`, then text is sent to Django.
- Interview responses: Django calls Azure OpenAI.
- Text-to-speech: Django uses `gTTS` and returns playable MP3 data URLs.
- Face emotion and look-away warning: browser sends sampled frames; Django uses OpenCV.
- Voice emotion: browser sends short audio chunks; Django uses Librosa/scikit-learn rules or optional saved model files.
- Fused emotion context: Django combines face and voice scores and injects the current mood into the interview prompt.
- Transcript: Django writes a transcript under `backend/media/transcripts`.
- Recording: browser records WebM and uploads it under `backend/media/recordings`.

Localhost is allowed by browsers for camera and microphone permissions.

Optional trained voice model files can be copied into `backend/` as `voice_emotion_mlp.pkl` and `voice_emotion_scaler.pkl`. If they are absent, the backend uses the same acoustic rules from the original project.
