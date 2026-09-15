# ╔══════════════════════════════════════════════════════╗
# ║  EMOTION ENGINE : MediaPipe (Face) + Librosa (Voice)║
# ║  Install:                                           ║
# ║    pip install mediapipe opencv-python              ║
# ║    pip install librosa soundfile scikit-learn numpy ║
# ║    pip install pyaudio                              ║
# ║    pip install openai                               ║
# ║  Approach:                                          ║
# ║    MediaPipe → facial muscle geometry (blendshapes) ║
# ║    Librosa   → MFCC/chroma/mel voice features       ║
# ║  Combined score fused at the AI context layer.      ║
# ╚══════════════════════════════════════════════════════╝

from openai import AzureOpenAI
import speech_recognition as sr
import threading
import pygame
import os
import tempfile
import time
import collections
import datetime
from gtts import gTTS

# ── MediaPipe imports ─────────────────────────────────
try:
    import cv2
    import mediapipe as mp
    FACE_EMOTION_AVAILABLE = True
except ImportError:
    FACE_EMOTION_AVAILABLE = False
    print(
        "[Warning] MediaPipe / OpenCV not found.\n"
        "  Run:  pip install mediapipe opencv-python\n"
        "  Continuing without facial emotion monitoring...\n"
    )

# ── Librosa imports ───────────────────────────────────
try:
    import numpy as np
    import librosa
    import soundfile as sf
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    import pickle
    VOICE_EMOTION_AVAILABLE = True
except ImportError:
    VOICE_EMOTION_AVAILABLE = False
    print(
        "[Warning] librosa / sklearn not found.\n"
        "  Run:  pip install librosa soundfile scikit-learn numpy\n"
        "  Continuing without voice emotion monitoring...\n"
    )

# ── PyAudio for raw audio capture ─────────────────────
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print(
        "[Warning] PyAudio not found — voice emotion capture disabled.\n"
        "  Run:  pip install pyaudio\n"
        "  Interview will still run, just without voice emotion analysis.\n"
    )

if not PYAUDIO_AVAILABLE:
    VOICE_EMOTION_AVAILABLE = False

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
AZURE_OPENAI_DEPLOYMENT  = "gpt-4o"

EMOTION_AWARE_AI        = True
EMOTION_SAMPLE_INTERVAL = 0.5   # Face sampling interval (seconds)
CAMERA_INDEX            = 0

# Voice emotion settings
VOICE_SAMPLE_DURATION   = 3     # seconds of audio per voice reading
VOICE_SAMPLE_INTERVAL   = 4     # seconds between voice reads

# Audio capture settings
AUDIO_RATE     = 22050
AUDIO_CHANNELS = 1
AUDIO_CHUNK    = 1024

# Path to save/load a pre-trained voice model
MODEL_PATH  = "voice_emotion_mlp.pkl"
SCALER_PATH = "voice_emotion_scaler.pkl"

SAVE_TRANSCRIPT = True
TRANSCRIPT_DIR  = "transcripts"

# ── Video + audio recording settings ──────────────────
SAVE_VIDEO     = True
RECORDINGS_DIR = "recordings"
VIDEO_FPS      = 20.0
# Use mp4v codec — writes a valid .mp4 directly, no conversion needed
VIDEO_FOURCC   = "mp4v"

REC_AUDIO_RATE     = 44100
REC_AUDIO_CHANNELS = 1
REC_AUDIO_CHUNK    = 1024

# Emotion labels
FACE_EMOTION_LABELS  = ["happy", "neutral", "fear", "sad", "angry", "disgust", "surprise"]
VOICE_EMOTION_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# Fusion weight: how much each modality contributes (must sum to 1.0)
FACE_WEIGHT  = 0.55
VOICE_WEIGHT = 0.45

EMOTION_NORMALISE = {
    "fearful":   "fear",
    "calm":      "neutral",
    "surprised": "surprise",
}

EMOTION_INTERVIEW_MAP = {
    "happy":    "confident and comfortable",
    "neutral":  "composed and professional",
    "fear":     "anxious or nervous",
    "sad":      "low confidence or demotivated",
    "angry":    "defensive or frustrated",
    "disgust":  "uncomfortable or disagreeing",
    "surprise": "caught off guard",
}

# ─────────────────────────────────────────
#  LOOK-AWAY DETECTION SETTINGS  [NEW]
# ─────────────────────────────────────────
LOOK_AWAY_THRESHOLD_SEC = 1.0   # seconds of looking away before warning triggers
_look_away_start        = None  # wall time when look-away began
_look_away_warning      = False # current warning state (shown on frame)
_look_away_lock         = threading.Lock()


# ─────────────────────────────────────────
#  SYSTEM PROMPT TEMPLATE  [UPDATED — stricter]
# ─────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """You are a senior hiring manager conducting a rigorous, structured professional job interview for the role of {job_role}.

ROLE & TONE
- Maintain a calm, professional, and measured tone. You are evaluating whether this candidate truly meets the bar for this role.
- Do NOT over-praise. Only acknowledge an answer as good if it is genuinely specific, structured, and relevant. Generic or vague answers must be challenged.
- Speak naturally as in a real face-to-face interview. No bullet points, no markdown, no asterisks.
- Keep every response to 1-2 spoken sentences maximum.

INTERVIEW STRUCTURE
Tailor every question specifically to the {job_role} position. Follow this order, spending roughly equal time on each area:
1. Introduction & background (1–2 questions)
2. Technical skills relevant to the {job_role} role (2–3 questions)
3. Behavioural / situational questions using the STAR method (2–3 questions)
4. Problem-solving or scenario-based challenge relevant to {job_role} (1–2 questions)
5. Candidate's own questions and closing (1 question)

QUESTION RULES
- Ask only ONE question at a time. Never stack multiple questions.
- Always acknowledge the candidate's previous answer BRIEFLY (one short phrase) before moving on — do NOT over-validate.
- If an answer is vague, incomplete, or lacks specifics, you MUST probe with a direct follow-up such as "Can you give me a specific example?" or "What exactly was your personal contribution in that situation?" Do NOT move on until you get a concrete answer.
- If a candidate cannot answer a technical question after one attempt, note it and move on — do not help them arrive at the answer.
- Vary question types: open-ended, hypothetical, and competency-based.
- Do NOT repeat a topic already covered earlier in the session.

EVALUATION MINDSET — STRICT MODE
- You are evaluating with a high bar. Most candidates will not fully meet it; that is expected.
- If an answer lacks specifics, structure, or relevance, say so directly but professionally: "That answer was quite general — I'd like to hear a concrete example."
- If the candidate hedges, deflects, or gives textbook answers without substance, push back once firmly.
- Do NOT let weak answers pass unchallenged.
- Only affirm an answer if it is genuinely strong and specific. Otherwise, remain neutral and press for depth.
- If a candidate seems overly rehearsed or gives buzzword-heavy answers, ask a sharp clarifying question to test real understanding.

CLOSING
- When wrapping up, thank the candidate professionally.
- Provide an honest, balanced 1-2 sentence summary of what you observed — both strengths and areas of concern.
- Explain the next steps (e.g., "We will review your responses and be in touch within a few days.").
- Do NOT give false hope or overly positive closing remarks if the interview was mediocre.
"""


# ─────────────────────────────────────────
#  SETUP — Azure OpenAI client
# ─────────────────────────────────────────
client = AzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

conversation_history = []
transcript_log       = []

recognizer = sr.Recognizer()
recognizer.energy_threshold = 400
recognizer.pause_threshold  = 1.5
recognizer.non_speaking_duration = 0.5
recognizer.dynamic_energy_threshold = True

pygame.mixer.init()
interrupt_flag = threading.Event()

face_emotion_log     = []
voice_emotion_log    = []
face_emotion_window  = collections.deque(maxlen=10)
voice_emotion_window = collections.deque(maxlen=10)

combined_emotion_window = collections.deque(maxlen=10)

emotion_monitoring = threading.Event()
interruption_stop  = threading.Event()

_latest_frame      = None
_latest_frame_lock = threading.Lock()
_recording_active  = threading.Event()

_mic_audio_chunks = []
_mic_audio_lock   = threading.Lock()

_tts_clips       = []
_tts_clips_lock  = threading.Lock()
_record_start_time = 0.0

# ── Updated: both silent and final paths are now .mp4 ──
_video_silent_path = ""
_video_final_path  = ""
_video_timestamp   = ""


# ─────────────────────────────────────────
#  JOB ROLE INPUT
# ─────────────────────────────────────────
def get_job_role() -> str:
    print("╔══════════════════════════════════════════════════════╗")
    print("║         INTERVIEW ASSISTANT — JOB ROLE SETUP        ║")
    print("╚══════════════════════════════════════════════════════╝")
    role = input("\n  Enter the job role you are interviewing for: ").strip()
    if not role:
        role = "Software Engineer"
        print(f"  [No input — defaulting to: {role}]")
    print(f"\n  Interview will be tailored for: {role}\n")
    return role


# ─────────────────────────────────────────
#  MEDIAPIPE BLENDSHAPE → EMOTION MAPPER
# ─────────────────────────────────────────
def blendshapes_to_emotion(blendshapes):
    bs = {b.category_name: b.score for b in blendshapes}

    smile_l    = bs.get("mouthSmileLeft",  0)
    smile_r    = bs.get("mouthSmileRight", 0)
    frown_l    = bs.get("mouthFrownLeft",  0)
    frown_r    = bs.get("mouthFrownRight", 0)
    brow_down  = bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)
    brow_up    = bs.get("browInnerUp", 0)
    jaw_open   = bs.get("jawOpen", 0)
    eye_wide_l = bs.get("eyeWideLeft", 0)
    eye_wide_r = bs.get("eyeWideRight", 0)
    eye_sq_l   = bs.get("eyeSquintLeft", 0)
    eye_sq_r   = bs.get("eyeSquintRight", 0)
    nose_sneer = bs.get("noseSneerLeft", 0) + bs.get("noseSneerRight", 0)
    lip_press  = bs.get("lipsPucker", 0)

    smile_avg = (smile_l + smile_r) / 2
    frown_avg = (frown_l + frown_r) / 2
    eye_wide  = (eye_wide_l + eye_wide_r) / 2
    eye_sq    = (eye_sq_l + eye_sq_r) / 2

    scores = {
        "happy":    smile_avg * 1.5 + eye_sq * 0.3,
        "sad":      frown_avg * 1.2 + brow_up * 0.4,
        "angry":    brow_down * 0.8 + eye_sq * 0.4 + frown_avg * 0.3,
        "surprise": jaw_open  * 0.8 + eye_wide * 0.8 + brow_up * 0.4,
        "fear":     eye_wide  * 0.6 + brow_up * 0.5 + jaw_open * 0.3,
        "disgust":  nose_sneer * 1.0 + lip_press * 0.5,
        "neutral":  0.3,
    }

    dominant = max(scores, key=scores.get)
    if scores[dominant] < 0.15:
        dominant = "neutral"

    return dominant, scores


# ─────────────────────────────────────────
#  IMPROVED FACE EMOTION DETECTOR  [FIX #1]
#  Uses HOG-style edge density, regional
#  variance, eye-aspect-ratio, mouth-open
#  ratio and forehead/cheek brightness to
#  distinguish all 7 emotion classes.
# ─────────────────────────────────────────
def analyse_face_roi(gray_frame, x, y, w, h):
    """
    Compute emotion scores from a grayscale face ROI.
    Returns (dominant_emotion, scores_dict).
    """
    face = gray_frame[y:y+h, x:x+w]
    if face.size == 0:
        return "neutral", {e: (1.0 if e == "neutral" else 0.0) for e in FACE_EMOTION_LABELS}

    # ── Region boundaries (normalised) ────────────────
    # Forehead: top 20%, Eye zone: 20-50%, Nose: 40-65%, Mouth: 60-90%
    forehead = face[0          : int(h*0.20), int(w*0.15):int(w*0.85)]
    eye_zone = face[int(h*0.20): int(h*0.50), :]
    nose_zone= face[int(h*0.40): int(h*0.65), int(w*0.25):int(w*0.75)]
    mouth    = face[int(h*0.60): int(h*0.90), int(w*0.15):int(w*0.85)]
    l_cheek  = face[int(h*0.35): int(h*0.70), 0          :int(w*0.35)]
    r_cheek  = face[int(h*0.35): int(h*0.70), int(w*0.65):]

    def safe_var(roi):
        return float(roi.var()) if roi.size > 0 else 0.0

    def safe_mean(roi):
        return float(roi.mean()) if roi.size > 0 else 128.0

    def edge_density(roi):
        if roi.size == 0:
            return 0.0
        edges = cv2.Laplacian(roi, cv2.CV_64F)
        return float(np.mean(np.abs(edges)))

    # ── Feature extraction ─────────────────────────────
    mouth_var     = safe_var(mouth)
    mouth_bright  = safe_mean(mouth)
    eye_var       = safe_var(eye_zone)
    eye_edge      = edge_density(eye_zone)
    forehead_var  = safe_var(forehead)
    forehead_edge = edge_density(forehead)
    nose_edge     = edge_density(nose_zone)
    cheek_asym    = abs(safe_mean(l_cheek) - safe_mean(r_cheek))

    # Mouth openness: compare top-half vs bottom-half brightness of mouth ROI
    if mouth.shape[0] >= 4:
        m_top = safe_mean(mouth[:mouth.shape[0]//2, :])
        m_bot = safe_mean(mouth[mouth.shape[0]//2:, :])
        mouth_open_ratio = abs(m_top - m_bot) / (m_top + 1e-6)
    else:
        mouth_open_ratio = 0.0

    # Eye wideness: horizontal edge density in eye zone
    eye_wide_score = min(eye_edge / 12.0, 1.0)

    # Brow raise: forehead edge density
    brow_raise = min(forehead_edge / 10.0, 1.0)

    # Smile estimate: high mouth variance + high mouth brightness relative to cheeks
    cheek_mean  = (safe_mean(l_cheek) + safe_mean(r_cheek)) / 2
    smile_score = min(mouth_var / 600.0 + max(0, mouth_bright - cheek_mean) / 80.0, 1.0)

    # Frown estimate: forehead edge + low mouth brightness + brow lowering
    frown_score = min(forehead_var / 400.0 + max(0, cheek_mean - mouth_bright) / 60.0, 1.0)

    # Disgust: asymmetric cheeks + high nose edge
    disgust_score = min(cheek_asym / 30.0 + nose_edge / 15.0, 1.0)

    # ── Compute weighted emotion scores ───────────────
    scores = {
        "happy":   smile_score * 1.4 + mouth_open_ratio * 0.3,
        "sad":     frown_score * 1.2 + (1.0 - smile_score) * 0.3,
        "angry":   forehead_edge / 10.0 * 0.8 + frown_score * 0.5 + (1.0 - eye_wide_score) * 0.2,
        "surprise":mouth_open_ratio * 1.0 + eye_wide_score * 0.8 + brow_raise * 0.6,
        "fear":    eye_wide_score * 0.7 + brow_raise * 0.5 + mouth_open_ratio * 0.3,
        "disgust": disgust_score  * 1.2,
        "neutral": 0.35,   # base neutral pull keeps it competitive when nothing fires
    }

    # Clamp all to [0, 1]
    scores = {k: float(min(max(v, 0.0), 1.0)) for k, v in scores.items()}

    # Normalise so the scores are relative
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    dominant = max(scores, key=scores.get)
    # Only call non-neutral if it is meaningfully above neutral
    if scores[dominant] < 0.20:
        dominant = "neutral"

    return dominant, scores


# ─────────────────────────────────────────
#  STRICT GAZE ESTIMATOR  [UPDATED]
# ─────────────────────────────────────────
def estimate_gaze_direction(gray_frame, face_rect):
    """
    Strict gaze estimator combining head-pose symmetry,
    iris darkest-point position, and vertical tilt.
    Returns: "center", "left", "right", or "up"
    """
    x, y, w, h = face_rect
    fh, fw = gray_frame.shape[:2]

    def clamp_roi(y1, y2, x1, x2):
        return (max(0, y1), min(fh, y2), max(0, x1), min(fw, x2))

    # ── SIGNAL 1 — HEAD POSE via face horizontal symmetry ──
    ey1, ey2, ex1, ex2 = clamp_roi(
        y + int(h * 0.20), y + int(h * 0.50), x, x + w
    )
    eye_stripe = gray_frame[ey1:ey2, ex1:ex2]

    head_signal = "center"
    if eye_stripe.size > 0:
        sobel_x = cv2.Sobel(eye_stripe, cv2.CV_64F, 1, 0, ksize=3)
        col_energy = np.sum(np.abs(sobel_x), axis=0).astype(float)
        total_e = col_energy.sum()
        if total_e > 0:
            cols       = np.arange(len(col_energy))
            com        = float(np.dot(cols, col_energy) / total_e)
            norm_com   = com / (len(col_energy) + 1e-6)
            if norm_com < 0.42:
                head_signal = "left"
            elif norm_com > 0.58:
                head_signal = "right"

    # ── SIGNAL 2 — IRIS POSITION via darkest pixel (minMaxLoc) ──
    def pupil_position(roi):
        if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
            return 0.5
        blurred = cv2.GaussianBlur(roi, (7, 7), 0)
        vh = blurred.shape[0]
        strip = blurred[vh // 4: 3 * vh // 4, :]
        if strip.size == 0:
            return 0.5
        _, _, min_loc, _ = cv2.minMaxLoc(strip)
        px = min_loc[0]
        return float(px) / (strip.shape[1] + 1e-6)

    ry1, ry2 = y + int(h * 0.23), y + int(h * 0.46)

    re_y1, re_y2, re_x1, re_x2 = clamp_roi(ry1, ry2, x + int(w * 0.08), x + int(w * 0.44))
    le_y1, le_y2, le_x1, le_x2 = clamp_roi(ry1, ry2, x + int(w * 0.56), x + int(w * 0.92))

    r_eye_roi = gray_frame[re_y1:re_y2, re_x1:re_x2]
    l_eye_roi = gray_frame[le_y1:le_y2, le_x1:le_x2]

    rp = pupil_position(r_eye_roi)
    lp = pupil_position(l_eye_roi)
    avg_iris = (rp + lp) / 2.0

    iris_signal = "center"
    if avg_iris < 0.40:
        iris_signal = "left"
    elif avg_iris > 0.60:
        iris_signal = "right"

    # ── SIGNAL 3 — VERTICAL HEAD TILT ──
    ft_y1, ft_y2, ft_x1, ft_x2 = clamp_roi(y, y + int(h * 0.35), x + int(w*0.1), x + int(w*0.9))
    fb_y1, fb_y2, fb_x1, fb_x2 = clamp_roi(y + int(h * 0.60), y + h, x + int(w*0.1), x + int(w*0.9))

    top_roi = gray_frame[ft_y1:ft_y2, ft_x1:ft_x2]
    bot_roi = gray_frame[fb_y1:fb_y2, fb_x1:fb_x2]

    vert_signal = "center"
    if top_roi.size > 0 and bot_roi.size > 0:
        vert_ratio = float(top_roi.mean()) / (float(bot_roi.mean()) + 1e-6)
        if vert_ratio > 1.18:
            vert_signal = "up"

    # ── FINAL VOTE ──
    if vert_signal == "up":
        return "up"

    h_votes = [head_signal, iris_signal]
    left_votes  = h_votes.count("left")
    right_votes = h_votes.count("right")

    if left_votes >= 1:
        return "left"
    if right_votes >= 1:
        return "right"

    return "center"


def check_look_away(gaze: str) -> bool:
    """
    Update look-away timer and return True if warning should be shown.
    """
    global _look_away_start, _look_away_warning

    with _look_away_lock:
        if gaze != "center":
            if _look_away_start is None:
                _look_away_start = time.time()
            elapsed = time.time() - _look_away_start
            if elapsed >= LOOK_AWAY_THRESHOLD_SEC:
                _look_away_warning = True
            else:
                _look_away_warning = False
        else:
            _look_away_start   = None
            _look_away_warning = False
        return _look_away_warning


def draw_look_away_warning(frame):
    """
    Draw a flashing red warning banner on the frame in-place.
    """
    if int(time.time() * 2) % 2 == 0:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 200), -1)
        cv2.putText(
            frame,
            "!  LOOK AT THE CAMERA",
            (w // 2 - 200, h - 22),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 4)


# ─────────────────────────────────────────
#  VOICE FEATURE EXTRACTION (Librosa)
# ─────────────────────────────────────────
def extract_voice_features(audio_data: "np.ndarray", sample_rate: int) -> "np.ndarray":
    features = []

    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    features.extend(np.mean(mfcc.T, axis=0))

    stft   = np.abs(librosa.stft(audio_data))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    features.extend(np.mean(chroma.T, axis=0))

    mel = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    features.extend(np.mean(mel.T, axis=0))

    return np.array(features)


def build_or_load_voice_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            print(f"[Setup] Voice emotion model loaded from {MODEL_PATH}")
            return model, scaler
        except Exception as e:
            print(f"[Setup] Could not load saved model: {e}. Using built-in rules.")

    print("[Setup] No pre-trained voice emotion model found.")
    print("        Using acoustic feature rules for emotion detection.")
    print("        For better accuracy: train on RAVDESS and save to voice_emotion_mlp.pkl")
    return None, None


def classify_voice_emotion(audio_data: "np.ndarray", sr_rate: int, model, scaler) -> tuple:
    features = extract_voice_features(audio_data, sr_rate)

    if model is not None and scaler is not None:
        features_scaled = scaler.transform([features])
        pred   = model.predict(features_scaled)[0]
        proba  = model.predict_proba(features_scaled)[0]
        scores = dict(zip(model.classes_, proba.tolist()))
        return pred, scores

    rms    = float(np.sqrt(np.mean(audio_data ** 2)))
    zcr    = float(np.mean(librosa.feature.zero_crossing_rate(audio_data)[0]))

    cent   = librosa.feature.spectral_centroid(y=audio_data, sr=sr_rate)
    s_cent = float(np.mean(cent))

    f0, voiced_flag, _ = librosa.pyin(
        audio_data,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr_rate,
    )
    voiced_f0  = f0[voiced_flag] if voiced_flag is not None else np.array([])
    pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    pitch_std  = float(np.std(voiced_f0))  if len(voiced_f0) > 0 else 0.0
    voiced_ratio = float(np.sum(voiced_flag)) / len(f0) if (voiced_flag is not None and len(f0) > 0) else 0.0

    if rms < 0.008:
        dominant = "neutral"
        scores   = {e: 0.03 for e in VOICE_EMOTION_LABELS}
        scores["neutral"] = 0.82
        return dominant, scores

    ev = {e: 0.0 for e in VOICE_EMOTION_LABELS}

    ev["neutral"] += 0.30

    if rms > 0.04:
        ev["happy"] += 0.25
    if pitch_mean > 180:
        ev["happy"] += 0.20
    if 20 < pitch_std < 80:
        ev["happy"] += 0.15
    if s_cent > 1800:
        ev["happy"] += 0.15
    ev["calm"]  += 0.10

    if rms > 0.08:
        ev["angry"] += 0.30
    if zcr > 0.12:
        ev["angry"] += 0.20
    if pitch_std > 80:
        ev["angry"] += 0.20
    if s_cent > 2200:
        ev["angry"] += 0.15

    if rms < 0.025:
        ev["sad"] += 0.30
    if pitch_mean < 130 and pitch_mean > 0:
        ev["sad"] += 0.20
    if pitch_std < 15 and voiced_ratio > 0.3:
        ev["sad"] += 0.20
    if zcr < 0.04:
        ev["sad"] += 0.15

    if 0.01 < rms < 0.04:
        ev["fearful"] += 0.20
    if pitch_std > 50 and pitch_mean < 200:
        ev["fearful"] += 0.20
    if voiced_ratio < 0.35:
        ev["fearful"] += 0.25
    if 0.05 < zcr < 0.10:
        ev["fearful"] += 0.10

    if rms > 0.06 and pitch_std > 60:
        ev["surprised"] += 0.30
    if pitch_mean > 220:
        ev["surprised"] += 0.20
    if zcr > 0.14:
        ev["surprised"] += 0.15

    if pitch_mean < 120 and pitch_mean > 0:
        ev["disgust"] += 0.20
    if 0.03 < zcr < 0.08 and rms < 0.035:
        ev["disgust"] += 0.15
    if voiced_ratio < 0.30:
        ev["disgust"] += 0.10

    total = sum(ev.values())
    scores = {k: v / total for k, v in ev.items()} if total > 0 else ev

    dominant = max(scores, key=scores.get)

    if scores[dominant] < 0.30 or (dominant != "neutral" and scores[dominant] - scores["neutral"] < 0.08):
        dominant = "neutral"

    return dominant, scores


# ─────────────────────────────────────────
#  EMOTION FUSION
# ─────────────────────────────────────────
UNIFIED_EMOTIONS = ["happy", "neutral", "fear", "sad", "angry", "disgust", "surprise"]

VOICE_TO_UNIFIED = {
    "neutral":   "neutral",
    "calm":      "neutral",
    "happy":     "happy",
    "sad":       "sad",
    "angry":     "angry",
    "fearful":   "fear",
    "disgust":   "disgust",
    "surprised": "surprise",
}

FACE_TO_UNIFIED = {e: e for e in UNIFIED_EMOTIONS}


def fuse_emotions(face_scores: dict, voice_scores: dict) -> tuple:
    fused = {e: 0.0 for e in UNIFIED_EMOTIONS}

    for label, score in face_scores.items():
        unified = FACE_TO_UNIFIED.get(label)
        if unified:
            fused[unified] += score * FACE_WEIGHT

    for label, score in voice_scores.items():
        unified = VOICE_TO_UNIFIED.get(label)
        if unified:
            fused[unified] += score * VOICE_WEIGHT

    total = sum(fused.values())
    if total > 0:
        fused = {k: v / total for k, v in fused.items()}

    dominant = max(fused, key=fused.get)
    if fused[dominant] < 0.15:
        dominant = "neutral"

    return dominant, fused


# ─────────────────────────────────────────
#  TRANSCRIPT HELPERS
# ─────────────────────────────────────────
def log_transcript(speaker: str, text: str):
    transcript_log.append({
        "time":    datetime.datetime.now().strftime("%H:%M:%S"),
        "speaker": speaker,
        "text":    text.strip(),
    })


def save_transcript(job_role: str):
    if not transcript_log:
        print("\n[Transcript] Nothing to save.")
        return
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    filename = datetime.datetime.now().strftime("interview_%Y%m%d_%H%M%S.txt")
    filepath = os.path.join(TRANSCRIPT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("         INTERVIEW TRANSCRIPT\n")
        f.write(f"  Date  : {datetime.datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"  Role  : {job_role}\n")
        f.write(f"  Model : {AZURE_OPENAI_DEPLOYMENT}\n")
        f.write(f"  Engine: MediaPipe (Face) + Librosa (Voice)\n")
        f.write("=" * 60 + "\n\n")
        for entry in transcript_log:
            f.write(f"[{entry['time']}] {entry['speaker'].upper()}\n")
            f.write(f"  {entry['text']}\n\n")
        f.write("=" * 60 + "\nEND OF TRANSCRIPT\n")
    print(f"\n[Transcript] Saved → {filepath}")


# ─────────────────────────────────────────
#  AUDIO + VIDEO RECORDING
#  [FIXED] — records directly to .mp4 using
#  the mp4v codec. The silent file is already
#  a valid MP4, so ffmpeg/moviepy only need to
#  inject audio — no video re-encoding needed.
# ─────────────────────────────────────────

def mic_capture_thread():
    if not PYAUDIO_AVAILABLE:
        return

    global _record_start_time
    pa     = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=REC_AUDIO_CHANNELS,
        rate=REC_AUDIO_RATE,
        input=True,
        frames_per_buffer=REC_AUDIO_CHUNK,
    )
    _record_start_time = time.time()
    print("[Mic Capture] Recording microphone audio...")

    while _recording_active.is_set():
        try:
            data = stream.read(REC_AUDIO_CHUNK, exception_on_overflow=False)
            with _mic_audio_lock:
                _mic_audio_chunks.append(np.frombuffer(data, dtype=np.float32).copy())
        except Exception as e:
            print(f"[Mic Capture] Error: {e}")

    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("[Mic Capture] Microphone recording stopped.")


def video_recording_thread(job_role: str):
    """
    Records video frames directly to an MP4 file using the mp4v codec.
    The resulting _video_silent_path is a valid (silent) .mp4 file.
    mux_recording() will inject audio into it to produce the final .mp4.
    """
    if not FACE_EMOTION_AVAILABLE:
        return

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_role = "".join(
        c if c.isalnum() or c in (' ', '_') else '_' for c in job_role
    ).strip().replace(' ', '_')

    global _video_silent_path, _video_final_path, _video_timestamp
    _video_timestamp   = timestamp
    # Both paths are now .mp4 — no .avi intermediate
    _video_silent_path = os.path.join(RECORDINGS_DIR, f"interview_{safe_role}_{timestamp}_silent.mp4")
    _video_final_path  = os.path.join(RECORDINGS_DIR, f"interview_{safe_role}_{timestamp}.mp4")

    writer = None
    print(f"[Video Recorder] Will save recording to → {_video_final_path}")

    while _recording_active.is_set():
        with _latest_frame_lock:
            frame = _latest_frame

        if frame is not None:
            if writer is None:
                h, w = frame.shape[:2]
                # mp4v produces a proper .mp4 container on all platforms
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(_video_silent_path, fourcc, VIDEO_FPS, (w, h))
                if not writer.isOpened():
                    # Fallback: try avc1 (H.264 via QuickTime, available on macOS/some Linux)
                    print("[Video Recorder] mp4v failed, trying avc1...")
                    writer.release()
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    writer = cv2.VideoWriter(_video_silent_path, fourcc, VIDEO_FPS, (w, h))
                if writer.isOpened():
                    print(f"[Video Recorder] Recording started ({w}x{h} @ {VIDEO_FPS}fps) → {_video_silent_path}")
                else:
                    print("[Video Recorder] ERROR: Could not open VideoWriter. Check codec support.")
                    return

            writer.write(frame)

        time.sleep(1.0 / VIDEO_FPS)

    if writer is not None:
        writer.release()
        print(f"[Video Recorder] Silent MP4 saved → {_video_silent_path}")
    else:
        print("[Video Recorder] No frames captured; no file written.")


def mux_recording():
    """
    Mixes mic audio + TTS audio, then muxes with the silent MP4 video
    to produce the final MP4 using ffmpeg (preferred) or moviepy (fallback).

    Key changes vs original:
    - _video_silent_path is now .mp4 (not .avi)
    - ffmpeg uses -c:v copy (no re-encode) since source is already H.264/mp4v
    - Temp audio file uses .wav extension consistently
    - Cleanup targets updated to .mp4 silent file
    """
    if not (SAVE_VIDEO and FACE_EMOTION_AVAILABLE and PYAUDIO_AVAILABLE):
        return

    silent_path = _video_silent_path
    final_path  = _video_final_path

    if not silent_path or not os.path.exists(silent_path):
        print("[Mux] Silent video not found — skipping mux.")
        return

    rec_start     = _record_start_time
    total_dur     = time.time() - rec_start
    total_samples = int(total_dur * REC_AUDIO_RATE) + REC_AUDIO_RATE

    print("[Mux] Building microphone track...")
    with _mic_audio_lock:
        chunks = list(_mic_audio_chunks)

    if chunks:
        mic_array = np.concatenate(chunks).astype(np.float32)
    else:
        mic_array = np.zeros(total_samples, dtype=np.float32)

    if len(mic_array) < total_samples:
        mic_array = np.pad(mic_array, (0, total_samples - len(mic_array)))
    else:
        mic_array = mic_array[:total_samples]

    print("[Mux] Building bot TTS track...")
    bot_array = np.zeros(total_samples, dtype=np.float32)

    with _tts_clips_lock:
        clips = list(_tts_clips)

    for play_time, mp3_path in clips:
        if not os.path.exists(mp3_path):
            print(f"[Mux] TTS clip not found (skipping): {mp3_path}")
            continue
        try:
            clip_audio, clip_sr = librosa.load(mp3_path, sr=REC_AUDIO_RATE, mono=True)
            offset_samples = int((play_time - rec_start) * REC_AUDIO_RATE)
            end_sample     = offset_samples + len(clip_audio)

            if end_sample > len(bot_array):
                bot_array = np.pad(bot_array, (0, end_sample - len(bot_array)))
                mic_array = np.pad(mic_array, (0, max(0, end_sample - len(mic_array))))

            if offset_samples >= 0:
                bot_array[offset_samples:end_sample] += clip_audio
        except Exception as e:
            print(f"[Mux] Could not load TTS clip {mp3_path}: {e}")

    def norm(arr):
        peak = np.max(np.abs(arr))
        return arr / peak if peak > 1e-6 else arr

    max_len   = max(len(mic_array), len(bot_array))
    mic_array = np.pad(mic_array, (0, max_len - len(mic_array)))
    bot_array = np.pad(bot_array, (0, max_len - len(bot_array)))

    mixed = norm(mic_array) * 0.55 + norm(bot_array) * 0.80
    mixed = np.clip(mixed, -1.0, 1.0)

    # Derive the wav path from the silent mp4 path (strip _silent.mp4 suffix)
    mixed_wav = silent_path.replace("_silent.mp4", "_audio.wav")
    sf.write(mixed_wav, mixed, REC_AUDIO_RATE, subtype="PCM_16")
    print(f"[Mux] Mixed audio saved → {mixed_wav}")

    import subprocess, shutil

    mux_success = False

    if shutil.which("ffmpeg"):
        print(f"[Mux] ffmpeg found — muxing → {final_path}")
        cmd = [
            "ffmpeg", "-y",
            "-i", silent_path,   # silent MP4 video
            "-i", mixed_wav,     # mixed WAV audio
            "-c:v", "copy",      # copy video stream — no re-encode
            "-c:a", "aac",       # encode audio as AAC
            "-b:a", "192k",
            "-movflags", "+faststart",   # enables streaming / quick open
            "-shortest",
            final_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"[Mux] Final recording saved → {final_path}")
                mux_success = True
            else:
                print(f"[Mux] ffmpeg returned error:\n{result.stderr[-600:]}")
        except Exception as e:
            print(f"[Mux] ffmpeg exception: {e}")
    else:
        print("[Mux] ffmpeg not found on PATH — trying moviepy fallback...")

    if not mux_success:
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip

            print(f"[Mux] moviepy — muxing → {final_path}")
            video_clip = VideoFileClip(silent_path)
            audio_clip = AudioFileClip(mixed_wav)

            if audio_clip.duration > video_clip.duration:
                audio_clip = audio_clip.subclip(0, video_clip.duration)

            final_clip = video_clip.set_audio(audio_clip)

            # Derive a temp audio filename that does not clash with final_path
            tmp_audio = silent_path.replace("_silent.mp4", "_tmp_audio.m4a")
            final_clip.write_videofile(
                final_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=tmp_audio,
                remove_temp=True,
                verbose=False,
                logger=None,
            )
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            print(f"[Mux] Final recording saved → {final_path}")
            mux_success = True

        except ImportError:
            print(
                "[Mux] Neither ffmpeg nor moviepy is available.\n"
                "         Your video and audio have been saved separately:\n"
                f"           Video : {silent_path}\n"
                f"           Audio : {mixed_wav}\n"
                "         To fix, install one of:\n"
                "           pip install moviepy\n"
                "           — OR —\n"
                "           https://ffmpeg.org/download.html  (then add to PATH)"
            )
        except Exception as e:
            print(f"[Mux] moviepy error: {e}")
            print(
                f"[Mux] Separate files kept:\n"
                f"        Video: {silent_path}\n"
                f"        Audio: {mixed_wav}"
            )

    # Clean up temp files only on success
    if mux_success:
        for path in [silent_path, mixed_wav]:
            try:
                os.remove(path)
            except Exception:
                pass

        with _tts_clips_lock:
            for _, mp3_path in _tts_clips:
                try:
                    os.remove(mp3_path)
                except Exception:
                    pass


# ─────────────────────────────────────────
#  FACE EMOTION MONITOR THREAD (MediaPipe)
# ─────────────────────────────────────────
def face_emotion_monitor_thread():
    if not FACE_EMOTION_AVAILABLE:
        return

    global _latest_frame

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[Face Emotion Monitor] Could not open camera at index {CAMERA_INDEX}.")
        return

    print("[Face Emotion Monitor] MediaPipe started — facial expressions are being monitored.")

    color_map = {
        "happy":    (0, 255, 0),
        "neutral":  (200, 200, 200),
        "fear":     (0, 100, 255),
        "sad":      (255, 100, 50),
        "angry":    (0, 0, 255),
        "disgust":  (0, 180, 100),
        "surprise": (255, 255, 0),
    }

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while emotion_monitoring.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        frame = cv2.flip(frame, 1)

        try:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

            dominant = "neutral"
            scores   = {e: (1.0/7) for e in FACE_EMOTION_LABELS}
            gaze_dir = "center"

            if len(faces) > 0:
                (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])

                dominant, scores = analyse_face_roi(gray, x, y, w, h)
                gaze_dir = estimate_gaze_direction(gray, (x, y, w, h))

                cv2.rectangle(frame, (x, y), (x+w, y+h),
                              color_map.get(dominant, (255, 255, 255)), 2)

                gaze_color = (0, 255, 0) if gaze_dir == "center" else (0, 165, 255)
                cv2.putText(frame, f"Gaze: {gaze_dir}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, gaze_color, 1)

            show_warning = check_look_away(gaze_dir)

            face_emotion_log.append({
                "timestamp": time.time(),
                "dominant":  dominant,
                "scores":    scores,
            })
            face_emotion_window.append(dominant)

            _update_combined_window(face_dominant=dominant, face_scores=scores)

            color = color_map.get(dominant, (255, 255, 255))
            cv2.rectangle(frame, (0, 0), (400, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Face: {dominant.upper()}", (15, 42),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2)

            last_voice = list(voice_emotion_window)[-1] if voice_emotion_window else "-"
            voice_color = color_map.get(last_voice, (180, 180, 180))
            cv2.rectangle(frame, (0, 65), (400, 100), (20, 20, 20), -1)
            cv2.putText(frame, f"Voice: {last_voice.upper()}", (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, voice_color, 2)

            bar_x, bar_y = 15, 110
            for emo, score in sorted(scores.items(), key=lambda x: -x[1]):
                bar_len = int(score * 180)
                bar_col = color_map.get(emo, (180, 180, 180))
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + bar_len, bar_y + 14), bar_col, -1)
                cv2.putText(frame, f"{emo[:3]} {score:.2f}",
                            (bar_x + bar_len + 5, bar_y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
                bar_y += 22

            if show_warning:
                draw_look_away_warning(frame)

            cv2.imshow("Interview Emotion Monitor — Face + Voice", frame)
            cv2.waitKey(1)

            with _latest_frame_lock:
                _latest_frame = frame.copy()


        except Exception as e:
            print(f"[Face Emotion Monitor] Frame error: {e}")
            cv2.imshow("Interview Emotion Monitor — Face + Voice", frame)
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    print("[Face Emotion Monitor] Camera stopped.")


# ─────────────────────────────────────────
#  VOICE EMOTION MONITOR THREAD (Librosa)
# ─────────────────────────────────────────
def voice_emotion_monitor_thread():
    if not VOICE_EMOTION_AVAILABLE:
        return

    mlp_model, scaler = build_or_load_voice_model()

    pa     = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=AUDIO_CHANNELS,
        rate=AUDIO_RATE,
        input=True,
        frames_per_buffer=AUDIO_CHUNK,
    )

    print("[Voice Emotion Monitor] Librosa started — microphone is being monitored.")

    while emotion_monitoring.is_set():
        try:
            frames   = []
            n_chunks = int(AUDIO_RATE / AUDIO_CHUNK * VOICE_SAMPLE_DURATION)
            for _ in range(n_chunks):
                data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))

            audio_data = np.concatenate(frames)
            energy     = float(np.sqrt(np.mean(audio_data**2)))

            if energy < 0.005:
                dominant = "neutral"
                scores   = {e: 0.05 for e in VOICE_EMOTION_LABELS}
                scores["neutral"] = 0.70
            else:
                dominant, scores = classify_voice_emotion(
                    audio_data, AUDIO_RATE, mlp_model, scaler
                )

            voice_emotion_log.append({
                "timestamp": time.time(),
                "dominant":  dominant,
                "scores":    scores,
                "energy":    energy,
            })
            voice_emotion_window.append(dominant)

            signal = EMOTION_INTERVIEW_MAP.get(
                VOICE_TO_UNIFIED.get(dominant, dominant), dominant
            )
            print(f"[Voice Emotion] {dominant.upper():<12} energy={energy:.4f}  ({signal})")

        except Exception as e:
            print(f"[Voice Emotion Monitor] Error: {e}")

        time.sleep(VOICE_SAMPLE_INTERVAL)

    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("[Voice Emotion Monitor] Voice emotion monitoring stopped.")


# ─────────────────────────────────────────
#  COMBINED WINDOW UPDATER
# ─────────────────────────────────────────
def _update_combined_window(face_dominant: str, face_scores: dict):
    if voice_emotion_log:
        voice_scores = voice_emotion_log[-1]["scores"]
    else:
        voice_scores = {"neutral": 1.0}

    unified_voice_scores = {}
    for label, score in voice_scores.items():
        u = VOICE_TO_UNIFIED.get(label)
        if u:
            unified_voice_scores[u] = unified_voice_scores.get(u, 0.0) + score

    fused_dominant, _ = fuse_emotions(face_scores, unified_voice_scores)
    combined_emotion_window.append(fused_dominant)


# ─────────────────────────────────────────
#  EMOTION HELPERS
# ─────────────────────────────────────────
def get_current_mood():
    if combined_emotion_window:
        recent = list(combined_emotion_window)[-5:]
        return collections.Counter(recent).most_common(1)[0][0]
    elif face_emotion_window:
        recent = list(face_emotion_window)[-5:]
        raw = collections.Counter(recent).most_common(1)[0][0]
        return FACE_TO_UNIFIED.get(raw, raw)
    elif voice_emotion_window:
        recent = list(voice_emotion_window)[-5:]
        raw = collections.Counter(recent).most_common(1)[0][0]
        return VOICE_TO_UNIFIED.get(raw, raw)
    return None


# ─────────────────────────────────────────
#  LISTEN
# ─────────────────────────────────────────
def listen():
    with sr.Microphone() as source:
        print("\n[Listening...] Speak now.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=10)
            text  = recognizer.recognize_google(audio)
            print(f"You: {text}")
            log_transcript("Candidate", text)
            return text
        except sr.WaitTimeoutError:
            print("(No speech detected, trying again...)")
            return None
        except sr.UnknownValueError:
            print("(Couldn't understand, please repeat)")
            return None
        except sr.RequestError:
            print("(Internet error — check your connection)")
            return None


# ─────────────────────────────────────────
#  ASK GPT-4o (Azure OpenAI)
# ─────────────────────────────────────────
def ask_gpt(user_text):
    conversation_history.append({"role": "user", "content": user_text})
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=conversation_history,
            max_tokens=300,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        reply = reply.replace("*","").replace("#","").replace("`","").replace("_","")
        conversation_history.append({"role": "assistant", "content": reply})
    except Exception as e:
        reply = "Sorry, I had trouble getting a response right now."
        print(f"[Azure OpenAI Error]: {e}")
    print(f"Assistant: {reply}")
    log_transcript("Interviewer", reply)
    return reply


def ask_gpt_with_emotion(user_text):
    current_mood    = get_current_mood()
    emotion_context = None

    if current_mood:
        interview_signal = EMOTION_INTERVIEW_MAP.get(current_mood, current_mood)

        active_modalities = []
        if FACE_EMOTION_AVAILABLE:
            active_modalities.append("facial expression")
        if VOICE_EMOTION_AVAILABLE:
            active_modalities.append("voice analysis")
        modality_str = " and ".join(active_modalities) if active_modalities else "behavioural analysis"

        emotion_context = (
            f"IMPORTANT CONTEXT: Combined {modality_str} shows the candidate currently appears "
            f"{interview_signal} (fused emotion: {current_mood}). "
            f"If they seem anxious or nervous, be more reassuring and supportive. "
            f"If they seem confident, you may ask more challenging follow-up questions. "
            f"If they seem confused or surprised, clarify or rephrase your question."
        )
        print(f"[Emotion Context Injected: {current_mood} → {interview_signal}]")

    if emotion_context:
        messages_with_context = (
            [conversation_history[0]]
            + [{"role": "system", "content": emotion_context}]
            + conversation_history[1:]
            + [{"role": "user", "content": user_text}]
        )
    else:
        messages_with_context = conversation_history + [{"role": "user", "content": user_text}]

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages_with_context,
            max_tokens=300,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        reply = reply.replace("*","").replace("#","").replace("`","").replace("_","")
        conversation_history.append({"role": "user",      "content": user_text})
        conversation_history.append({"role": "assistant", "content": reply})
    except Exception as e:
        reply = "Sorry, I had trouble responding."
        print(f"[Azure OpenAI Error]: {e}")

    print(f"Assistant: {reply}")
    log_transcript("Interviewer", reply)
    return reply


# ─────────────────────────────────────────
#  SPEAK
# ─────────────────────────────────────────
def speak(text):
    interrupt_flag.clear()
    try:
        tts = gTTS(text=text, lang="en", slow=False)
    except Exception as e:
        print(f"[TTS Error]: {e}")
        return
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        temp_path = f.name
    try:
        tts.save(temp_path)
    except Exception as e:
        print(f"[TTS Save Error]: {e}")
        return

    play_start = time.time()
    with _tts_clips_lock:
        _tts_clips.append((play_start, temp_path))

    pygame.mixer.music.load(temp_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        if interrupt_flag.is_set():
            pygame.mixer.music.stop()
            print("[Interrupted — listening to you...]")
            break
        time.sleep(0.05)
    time.sleep(0.2)

    if not SAVE_VIDEO:
        try:
            os.remove(temp_path)
        except Exception:
            pass
        with _tts_clips_lock:
            try:
                _tts_clips.remove((play_start, temp_path))
            except ValueError:
                pass


# ─────────────────────────────────────────
#  INTERRUPTION LISTENER
# ─────────────────────────────────────────
def interruption_listener():
    ir = sr.Recognizer()
    ir.energy_threshold         = 2000
    ir.dynamic_energy_threshold = False
    while not interruption_stop.is_set():
        try:
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                try:
                    with sr.Microphone() as source:
                        ir.listen(source, timeout=1, phrase_time_limit=1)
                        interrupt_flag.set()
                except Exception:
                    pass
            else:
                time.sleep(0.1)
        except pygame.error:
            break


# ─────────────────────────────────────────
#  TRANSLATED INPUT
# ─────────────────────────────────────────
def process_translated_input(translated_text: str):
    print(f"[Translated Input]: {translated_text}")
    log_transcript("Candidate (translated)", translated_text)
    emotion_available = FACE_EMOTION_AVAILABLE or VOICE_EMOTION_AVAILABLE
    if EMOTION_AWARE_AI and emotion_available:
        reply = ask_gpt_with_emotion(translated_text)
    else:
        reply = ask_gpt(translated_text)
    speak(reply)


# ─────────────────────────────────────────
#  COMBINED EMOTION REPORT
# ─────────────────────────────────────────
def generate_emotion_report():
    has_face  = len(face_emotion_log)  > 0
    has_voice = len(voice_emotion_log) > 0

    if not has_face and not has_voice:
        print("\n[Report] No emotion data collected.")
        return

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║      INTERVIEW EMOTION ANALYSIS REPORT              ║")
    print("║      Engine: MediaPipe (Face) + Librosa (Voice)     ║")
    print("╚══════════════════════════════════════════════════════╝")

    if has_face:
        face_emotions = [e["dominant"] for e in face_emotion_log]
        face_counts   = collections.Counter(face_emotions)
        face_total    = len(face_emotions)
        face_dur      = face_total * EMOTION_SAMPLE_INTERVAL

        print(f"\n  -- FACIAL EMOTION  (MediaPipe)  --")
        print(f"  Samples: {face_total}  |  Duration: {face_dur:.0f}s (~{face_dur/60:.1f} min)\n")
        for emotion, count in face_counts.most_common():
            pct    = (count / face_total) * 100
            bar    = "#" * int(pct / 4)
            signal = EMOTION_INTERVIEW_MAP.get(emotion, "")
            print(f"    {emotion:<10} {bar:<26} {pct:5.1f}%   ({signal})")

    if has_voice:
        voice_emotions = [e["dominant"] for e in voice_emotion_log]
        voice_counts   = collections.Counter(voice_emotions)
        voice_total    = len(voice_emotions)
        voice_dur      = voice_total * VOICE_SAMPLE_INTERVAL
        energies       = [e.get("energy", 0) for e in voice_emotion_log]
        avg_energy     = float(np.mean(energies)) if energies else 0.0

        print(f"\n  -- VOICE EMOTION  (Librosa)  --")
        print(f"  Samples: {voice_total}  |  Duration: {voice_dur:.0f}s (~{voice_dur/60:.1f} min)")
        print(f"  Avg Voice Energy: {avg_energy:.4f}  (higher = louder / more confident)\n")
        for emotion, count in voice_counts.most_common():
            pct    = (count / voice_total) * 100
            bar    = "#" * int(pct / 4)
            unified = VOICE_TO_UNIFIED.get(emotion, emotion)
            signal  = EMOTION_INTERVIEW_MAP.get(unified, "")
            print(f"    {emotion:<12} {bar:<26} {pct:5.1f}%   ({signal})")

    fused_emotions = list(combined_emotion_window)
    if fused_emotions:
        fused_counts = collections.Counter(fused_emotions)
        fused_total  = len(fused_emotions)

        print(f"\n  -- FUSED EMOTION SUMMARY  (Face {int(FACE_WEIGHT*100)}% + Voice {int(VOICE_WEIGHT*100)}%)  --")
        for emotion, count in fused_counts.most_common():
            pct    = (count / fused_total) * 100
            bar    = "#" * int(pct / 4)
            signal = EMOTION_INTERVIEW_MAP.get(emotion, "")
            print(f"    {emotion:<10} {bar:<26} {pct:5.1f}%   ({signal})")

        conf_pct    = ((fused_counts.get("happy",0) + fused_counts.get("neutral",0)) / fused_total) * 100
        anxiety_pct = ((fused_counts.get("fear",0) + fused_counts.get("sad",0) + fused_counts.get("disgust",0)) / fused_total) * 100
        anger_pct   = (fused_counts.get("angry",0) / fused_total) * 100

        print(f"\n  Confident / Calm   : {conf_pct:5.1f}%")
        print(f"  Anxious / Stressed : {anxiety_pct:5.1f}%")
        print(f"  Defensive / Angry  : {anger_pct:5.1f}%")

        if conf_pct >= 65:
            profile = "Highly confident and composed throughout."
        elif conf_pct >= 45:
            profile = "Moderately confident with some uncertainty."
        elif anxiety_pct >= 50:
            profile = "Significant anxiety detected across face and voice."
        else:
            profile = "Mixed signals — qualitative review recommended."
        print(f"\n  {profile}")

    print("\n╚══════════════════════════════════════════════════════╝\n")


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    job_role = get_job_role()

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(job_role=job_role)
    conversation_history.append({"role": "system", "content": system_prompt})

    emotion_available = FACE_EMOTION_AVAILABLE or VOICE_EMOTION_AVAILABLE

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Azure OpenAI Interview Assistant — GPT-4o          ║")
    print(f"║    Role        : {job_role:<37}║")
    print(f"║    Model       : {AZURE_OPENAI_DEPLOYMENT:<37}║")
    print(f"║    Face Emotion: {'Active (MediaPipe)' if FACE_EMOTION_AVAILABLE else 'Inactive':<37}║")
    print(f"║    Voice Emotion: {'Active (Librosa)' if VOICE_EMOTION_AVAILABLE else 'Inactive':<36}║")
    print(f"║    AI Mood Aware: {'Yes' if (EMOTION_AWARE_AI and emotion_available) else 'No':<36}║")
    print(f"║    Transcript  : {'Saving → transcripts/' if SAVE_TRANSCRIPT else 'Off':<37}║")
    print(f"║    Video       : {'Saving → recordings/ (.mp4)' if SAVE_VIDEO else 'Off':<37}║")
    print(f"║    Look-Away   : Warning after {LOOK_AWAY_THRESHOLD_SEC:.1f}s              ║")
    print("║    Say 'goodbye' to end                             ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    threading.Thread(target=interruption_listener, daemon=True).start()

    emotion_monitoring.set()
    _recording_active.set()

    if FACE_EMOTION_AVAILABLE:
        threading.Thread(target=face_emotion_monitor_thread, daemon=True).start()

    if VOICE_EMOTION_AVAILABLE:
        threading.Thread(target=voice_emotion_monitor_thread, daemon=True).start()

    if SAVE_VIDEO and FACE_EMOTION_AVAILABLE:
        threading.Thread(
            target=video_recording_thread,
            args=(job_role,),
            daemon=True,
        ).start()

    if SAVE_VIDEO and PYAUDIO_AVAILABLE:
        threading.Thread(target=mic_capture_thread, daemon=True).start()

    greeting = (
        "Hello, welcome to your interview session. "
        "I'm glad you could make it today. "
        "We will go through a few questions covering your background, skills, and experience. "
        "Please take your time and answer naturally. "
        "Let's begin — could you please start by introducing yourself?"
    )
    log_transcript("Interviewer", greeting)
    speak(greeting)

    while True:
        user_input = listen()
        if user_input is None:
            continue

        if any(w in user_input.lower() for w in ["goodbye", "bye", "exit", "quit", "stop"]):
            closing = (
                "Thank you so much for your time today. "
                "It was a pleasure speaking with you. "
                "We will review everything and be in touch within a few days. "
                "Take care and goodbye!"
            )
            log_transcript("Interviewer", closing)
            speak(closing)
            emotion_monitoring.clear()
            _recording_active.clear()
            interruption_stop.set()
            time.sleep(2.0)
            generate_emotion_report()
            if SAVE_VIDEO:
                mux_recording()
            if SAVE_TRANSCRIPT:
                save_transcript(job_role)
            break

        emotion_available = FACE_EMOTION_AVAILABLE or VOICE_EMOTION_AVAILABLE
        if EMOTION_AWARE_AI and emotion_available:
            ai_reply = ask_gpt_with_emotion(user_input)
        else:
            ai_reply = ask_gpt(user_input)
        speak(ai_reply)


if __name__ == "__main__":
    main()
