import base64
import tempfile

from django.conf import settings
from gtts import gTTS
from openai import AzureOpenAI

from .prompts import EMOTION_INTERVIEW_MAP


def _client():
    if not settings.AZURE_OPENAI_ENDPOINT or not settings.AZURE_OPENAI_API_KEY:
        return None
    return AzureOpenAI(
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
    )


def ask_interviewer(session, user_text: str):
    mood = session.current_mood()
    emotion_context = None
    if mood:
        signal = EMOTION_INTERVIEW_MAP.get(mood, mood)
        emotion_context = (
            f"Important context: browser media analysis suggests the candidate appears {signal} "
            f"(fused emotion: {mood}). Adapt supportively if anxious, probe more if confident, "
            "and clarify if confused, while keeping interview standards high."
        )

    messages = session.messages
    if emotion_context:
        messages = [session.messages[0], {"role": "system", "content": emotion_context}, *session.messages[1:]]
    messages = [*messages, {"role": "user", "content": user_text}]

    client = _client()
    if client is None:
        reply = "I am ready to continue, but Azure OpenAI environment variables are not configured on the server."
    else:
        try:
            response = client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                max_tokens=300,
                temperature=0.7,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as exc:
            reply = f"I had trouble reaching Azure OpenAI just now: {exc}"

    reply = reply.replace("*", "").replace("#", "").replace("`", "").replace("_", "")
    session.messages.append({"role": "user", "content": user_text})
    session.messages.append({"role": "assistant", "content": reply})
    session.log("Candidate", user_text)
    session.log("Interviewer", reply)
    return reply


def opening_question(session):
    client = _client()
    fallback = (
        "Hello, welcome to your interview session. We will go through questions covering your "
        "background, skills, and experience. To begin, please introduce yourself and briefly "
        f"explain why you are interested in the {session.job_role} role."
    )
    if client is None:
        reply = fallback
    else:
        try:
            response = client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    session.messages[0],
                    {
                        "role": "user",
                        "content": (
                            "Start the interview now. Give a brief professional welcome and ask exactly "
                            "one opening question. The question must be tailored to the job role."
                        ),
                    },
                ],
                max_tokens=180,
                temperature=0.7,
            )
            reply = response.choices[0].message.content.strip()
        except Exception:
            reply = fallback

    reply = reply.replace("*", "").replace("#", "").replace("`", "").replace("_", "")
    session.messages.append({"role": "assistant", "content": reply})
    session.log("Interviewer", reply)
    return reply


def tts_data_url(text: str):
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
            gTTS(text=text, lang="en", slow=False).save(tmp.name)
            tmp.seek(0)
            encoded = base64.b64encode(tmp.read()).decode("ascii")
        return f"data:audio/mpeg;base64,{encoded}"
    except Exception:
        return None
