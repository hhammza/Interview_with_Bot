from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from django.conf import settings

from .prompts import EMOTION_INTERVIEW_MAP, SYSTEM_PROMPT_TEMPLATE


VOICE_TO_UNIFIED = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fear",
    "disgust": "disgust",
    "surprised": "surprise",
    "uncertain": "uncertain",
}


@dataclass
class InterviewSession:
    id: UUID
    job_role: str
    messages: list[dict]
    transcript: list[dict] = field(default_factory=list)
    face_log: list[dict] = field(default_factory=list)
    voice_log: list[dict] = field(default_factory=list)
    fused_log: list[dict] = field(default_factory=list)
    fused_window: deque = field(default_factory=lambda: deque(maxlen=10))
    recording_paths: dict[str, Path] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    transcript_path: Path | None = None

    def log(self, speaker: str, text: str):
        self.transcript.append(
            {
                "time": datetime.utcnow().strftime("%H:%M:%S"),
                "speaker": speaker,
                "text": text.strip(),
            }
        )

    def current_mood(self):
        if self.fused_window:
            return Counter(list(self.fused_window)[-5:]).most_common(1)[0][0]
        if self.face_log:
            return Counter([e["dominant"] for e in self.face_log[-5:]]).most_common(1)[0][0]
        if self.voice_log:
            return Counter([e["dominant"] for e in self.voice_log[-5:]]).most_common(1)[0][0]
        return None


class SessionStore:
    def __init__(self):
        self._sessions: dict[UUID, InterviewSession] = {}

    def create(self, job_role: str) -> InterviewSession:
        session_id = uuid4()
        role = job_role.strip() or "Software Engineer"
        session = InterviewSession(
            id=session_id,
            job_role=role,
            messages=[{"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(job_role=role)}],
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id) -> InterviewSession:
        sid = UUID(str(session_id))
        return self._sessions[sid]

    def _question_answer_pairs(self, session: InterviewSession):
        pairs = []
        pending_question = None
        for entry in session.transcript:
            speaker = entry["speaker"].lower()
            if speaker == "interviewer":
                if pending_question is not None:
                    pairs.append({"question": pending_question, "answer": None})
                pending_question = entry
            elif speaker == "candidate":
                if pending_question is not None:
                    pairs.append({"question": pending_question, "answer": entry})
                    pending_question = None
                else:
                    pairs.append({"question": None, "answer": entry})
        if pending_question is not None:
            pairs.append({"question": pending_question, "answer": None})
        return pairs

    def _write_emotion_counts(self, f, title: str, counts: Counter, total: int, normalise=None):
        f.write(f"{title}\n")
        if total == 0:
            f.write("No samples were collected.\n\n")
            return
        for emotion, count in counts.most_common():
            label = normalise.get(emotion, emotion) if normalise else emotion
            meaning = EMOTION_INTERVIEW_MAP.get(label, label)
            percent = (count / total) * 100
            f.write(f"- {emotion}: {count} samples ({percent:.1f}%) - {meaning}\n")
        f.write("\n")

    def _write_emotion_summary(self, f, session: InterviewSession):
        face_emotions = [entry.get("dominant", "neutral") for entry in session.face_log]
        voice_emotions = [entry.get("dominant", "neutral") for entry in session.voice_log]
        fused_emotions = [entry.get("dominant", "neutral") for entry in session.fused_log]
        if not fused_emotions:
            fused_emotions = list(session.fused_window)

        f.write("\nCANDIDATE EMOTION ANALYSIS\n")
        f.write("=" * 60 + "\n")
        f.write("This section summarizes browser camera and microphone emotion signals collected during the interview.\n")
        f.write("These signals are supportive observations, not a final hiring decision.\n\n")

        self._write_emotion_counts(f, "Facial Expression Summary", Counter(face_emotions), len(face_emotions))
        self._write_emotion_counts(
            f,
            "Voice Emotion Summary",
            Counter(voice_emotions),
            len(voice_emotions),
            normalise=VOICE_TO_UNIFIED,
        )
        self._write_emotion_counts(f, "Combined Face + Voice Summary", Counter(fused_emotions), len(fused_emotions))

        total = max(len(fused_emotions), 1)
        fused_counts = Counter(fused_emotions)
        fused_entries = session.fused_log or [{"dominant": label, "scores": {label: 1.0}, "confidence": 1.0} for label in fused_emotions]
        confidence_total = sum(float(entry.get("confidence", 0.0)) for entry in fused_entries) or float(len(fused_entries))
        confident_calm = (
            sum(
                float(entry.get("confidence", 0.0))
                for entry in fused_entries
                if entry.get("dominant", "neutral") in {"happy", "neutral"}
            )
            / confidence_total
        ) * 100
        anxious_stressed = (
            sum(
                float(entry.get("confidence", 0.0))
                for entry in fused_entries
                if entry.get("dominant", "neutral") in {"fear", "sad", "disgust"}
            )
            / confidence_total
        ) * 100
        defensive_angry = (
            sum(
                float(entry.get("confidence", 0.0))
                for entry in fused_entries
                if entry.get("dominant", "neutral") == "angry"
            )
            / confidence_total
        ) * 100
        average_fused_confidence = confidence_total / max(len(fused_entries), 1)

        look_away_samples = sum(1 for entry in session.face_log if entry.get("lookAway"))
        voice_energies = [entry.get("energy", 0.0) for entry in session.voice_log if entry.get("energy") is not None]
        average_voice_energy = sum(voice_energies) / len(voice_energies) if voice_energies else 0.0

        f.write("Overall Candidate Signals\n")
        f.write(f"- Confident / calm signals: {confident_calm:.1f}%\n")
        f.write(f"- Anxious / stressed signals: {anxious_stressed:.1f}%\n")
        f.write(f"- Defensive / angry signals: {defensive_angry:.1f}%\n")
        f.write(f"- Average fused confidence: {average_fused_confidence:.2f}\n")
        f.write(f"- Look-away warning samples: {look_away_samples}\n")
        f.write(f"- Average voice energy: {average_voice_energy:.4f}\n\n")

        if fused_emotions:
            dominant = fused_counts.most_common(1)[0][0]
            meaning = EMOTION_INTERVIEW_MAP.get(dominant, dominant)
            dominant_confidence = 0.0
            for entry in fused_entries:
                if entry.get("dominant", "neutral") == dominant:
                    dominant_confidence = max(dominant_confidence, float(entry.get("confidence", 0.0)))
            if dominant == "uncertain":
                f.write("Dominant combined impression: uncertain (low-confidence or mixed signal).\n")
            else:
                f.write(f"Dominant combined impression: {dominant} ({meaning}), confidence {dominant_confidence:.2f}.\n")
        else:
            f.write("Dominant combined impression: unavailable because no emotion samples were collected.\n")
        f.write("\n")

    def save_transcript(self, session: InterviewSession) -> Path:
        transcript_dir = Path(settings.MEDIA_ROOT) / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        filename = f"interview_{session.created_at.strftime('%Y%m%d_%H%M%S')}_{session.id}.txt"
        path = transcript_dir / filename
        with path.open("w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("INTERVIEW TRANSCRIPT\n")
            f.write(f"Date  : {session.created_at.strftime('%Y-%m-%d')}\n")
            f.write(f"Role  : {session.job_role}\n")
            f.write(f"Model : {settings.AZURE_OPENAI_DEPLOYMENT}\n")
            f.write("Engine: Browser media + Django/Python analysis\n")
            f.write("=" * 60 + "\n\n")

            pairs = self._question_answer_pairs(session)
            for index, pair in enumerate(pairs, start=1):
                question = pair["question"]
                answer = pair["answer"]
                f.write(f"Question {index}\n")
                if question:
                    f.write(f"Interviewer [{question['time']}]:\n")
                    f.write(f"{question['text']}\n\n")
                else:
                    f.write("Interviewer:\n")
                    f.write("(No interviewer question recorded for this answer.)\n\n")

                f.write(f"Answer {index}\n")
                if answer:
                    f.write(f"Candidate [{answer['time']}]:\n")
                    f.write(f"{answer['text']}\n\n")
                else:
                    f.write("Candidate:\n")
                    f.write("(No answer recorded.)\n\n")
                f.write("-" * 60 + "\n\n")

            f.write("\nFULL CHRONOLOGICAL LOG\n")
            f.write("=" * 60 + "\n\n")
            for entry in session.transcript:
                f.write(f"[{entry['time']}] {entry['speaker'].upper()}\n")
                f.write(f"{entry['text']}\n\n")
            self._write_emotion_summary(f, session)
            f.write("=" * 60 + "\nEND OF TRANSCRIPT\n")
        session.transcript_path = path
        return path


store = SessionStore()
