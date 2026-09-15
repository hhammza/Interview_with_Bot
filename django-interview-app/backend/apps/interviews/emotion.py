import base64
import os
import pickle
import tempfile
import time

try:
    import cv2
    import numpy as np
    FACE_AVAILABLE = True
except Exception:
    FACE_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

try:
    import librosa
    import numpy as np
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False


FACE_EMOTION_LABELS = ["happy", "neutral", "fear", "sad", "angry", "disgust", "surprise"]
VOICE_EMOTION_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
UNIFIED_EMOTIONS = ["happy", "neutral", "fear", "sad", "angry", "disgust", "surprise", "uncertain"]
VOICE_TO_UNIFIED = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fear",
    "disgust": "disgust",
    "surprised": "surprise",
}
FACE_WEIGHT = 0.55
VOICE_WEIGHT = 0.45
LOOK_AWAY_THRESHOLD_SEC = 1.0

_look_away_started: dict[str, float] = {}


def analyse_face_roi(gray_frame, x, y, w, h):
    face = gray_frame[y:y + h, x:x + w]
    if face.size == 0:
        return "neutral", {e: (1.0 if e == "neutral" else 0.0) for e in FACE_EMOTION_LABELS}

    forehead = face[0:int(h * 0.20), int(w * 0.15):int(w * 0.85)]
    eye_zone = face[int(h * 0.20):int(h * 0.50), :]
    nose_zone = face[int(h * 0.40):int(h * 0.65), int(w * 0.25):int(w * 0.75)]
    mouth = face[int(h * 0.60):int(h * 0.90), int(w * 0.15):int(w * 0.85)]
    l_cheek = face[int(h * 0.35):int(h * 0.70), 0:int(w * 0.35)]
    r_cheek = face[int(h * 0.35):int(h * 0.70), int(w * 0.65):]

    def safe_var(roi):
        return float(roi.var()) if roi.size > 0 else 0.0

    def safe_mean(roi):
        return float(roi.mean()) if roi.size > 0 else 128.0

    def edge_density(roi):
        if roi.size == 0:
            return 0.0
        edges = cv2.Laplacian(roi, cv2.CV_64F)
        return float(np.mean(np.abs(edges)))

    mouth_var = safe_var(mouth)
    mouth_bright = safe_mean(mouth)
    eye_edge = edge_density(eye_zone)
    forehead_var = safe_var(forehead)
    forehead_edge = edge_density(forehead)
    nose_edge = edge_density(nose_zone)
    cheek_asym = abs(safe_mean(l_cheek) - safe_mean(r_cheek))

    if mouth.shape[0] >= 4:
        m_top = safe_mean(mouth[:mouth.shape[0] // 2, :])
        m_bot = safe_mean(mouth[mouth.shape[0] // 2:, :])
        mouth_open_ratio = abs(m_top - m_bot) / (m_top + 1e-6)
    else:
        mouth_open_ratio = 0.0

    eye_wide_score = min(eye_edge / 12.0, 1.0)
    brow_raise = min(forehead_edge / 10.0, 1.0)
    cheek_mean = (safe_mean(l_cheek) + safe_mean(r_cheek)) / 2
    smile_score = min(mouth_var / 600.0 + max(0, mouth_bright - cheek_mean) / 80.0, 1.0)
    frown_score = min(forehead_var / 400.0 + max(0, cheek_mean - mouth_bright) / 60.0, 1.0)
    disgust_score = min(cheek_asym / 30.0 + nose_edge / 15.0, 1.0)

    scores = {
        "happy": smile_score * 1.4 + mouth_open_ratio * 0.3,
        "sad": frown_score * 1.2 + (1.0 - smile_score) * 0.3,
        "angry": forehead_edge / 10.0 * 0.8 + frown_score * 0.5 + (1.0 - eye_wide_score) * 0.2,
        "surprise": mouth_open_ratio * 1.0 + eye_wide_score * 0.8 + brow_raise * 0.6,
        "fear": eye_wide_score * 0.7 + brow_raise * 0.5 + mouth_open_ratio * 0.3,
        "disgust": disgust_score * 1.2,
        "neutral": 0.35,
    }
    scores = {k: float(min(max(v, 0.0), 1.0)) for k, v in scores.items()}
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    dominant = max(scores, key=scores.get)
    if scores[dominant] < 0.20:
        dominant = "neutral"
    return dominant, scores


def estimate_gaze_direction(gray_frame, face_rect):
    x, y, w, h = face_rect
    fh, fw = gray_frame.shape[:2]

    def clamp_roi(y1, y2, x1, x2):
        return max(0, y1), min(fh, y2), max(0, x1), min(fw, x2)

    ey1, ey2, ex1, ex2 = clamp_roi(y + int(h * 0.20), y + int(h * 0.50), x, x + w)
    eye_stripe = gray_frame[ey1:ey2, ex1:ex2]
    head_signal = "center"
    if eye_stripe.size > 0:
        sobel_x = cv2.Sobel(eye_stripe, cv2.CV_64F, 1, 0, ksize=3)
        col_energy = np.sum(np.abs(sobel_x), axis=0).astype(float)
        total_e = col_energy.sum()
        if total_e > 0:
            cols = np.arange(len(col_energy))
            norm_com = float(np.dot(cols, col_energy) / total_e) / (len(col_energy) + 1e-6)
            if norm_com < 0.42:
                head_signal = "left"
            elif norm_com > 0.58:
                head_signal = "right"

    def pupil_position(roi):
        if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
            return 0.5
        blurred = cv2.GaussianBlur(roi, (7, 7), 0)
        strip = blurred[blurred.shape[0] // 4:3 * blurred.shape[0] // 4, :]
        if strip.size == 0:
            return 0.5
        _, _, min_loc, _ = cv2.minMaxLoc(strip)
        return float(min_loc[0]) / (strip.shape[1] + 1e-6)

    ry1, ry2 = y + int(h * 0.23), y + int(h * 0.46)
    re_y1, re_y2, re_x1, re_x2 = clamp_roi(ry1, ry2, x + int(w * 0.08), x + int(w * 0.44))
    le_y1, le_y2, le_x1, le_x2 = clamp_roi(ry1, ry2, x + int(w * 0.56), x + int(w * 0.92))
    avg_iris = (pupil_position(gray_frame[re_y1:re_y2, re_x1:re_x2]) + pupil_position(gray_frame[le_y1:le_y2, le_x1:le_x2])) / 2.0
    iris_signal = "left" if avg_iris < 0.40 else "right" if avg_iris > 0.60 else "center"

    t_y1, t_y2, t_x1, t_x2 = clamp_roi(y, y + int(h * 0.35), x + int(w * 0.1), x + int(w * 0.9))
    b_y1, b_y2, b_x1, b_x2 = clamp_roi(y + int(h * 0.60), y + h, x + int(w * 0.1), x + int(w * 0.9))
    top = gray_frame[t_y1:t_y2, t_x1:t_x2]
    bottom = gray_frame[b_y1:b_y2, b_x1:b_x2]
    if top.size > 0 and bottom.size > 0 and float(top.mean()) / (float(bottom.mean()) + 1e-6) > 1.18:
        return "up"
    if head_signal == "left" or iris_signal == "left":
        return "left"
    if head_signal == "right" or iris_signal == "right":
        return "right"
    return "center"


def check_look_away(session_id: str, gaze: str):
    if gaze != "center":
        _look_away_started.setdefault(session_id, time.time())
        return time.time() - _look_away_started[session_id] >= LOOK_AWAY_THRESHOLD_SEC
    _look_away_started.pop(session_id, None)
    return False


def analyse_face_image(session_id: str, data_url: str):
    if not FACE_AVAILABLE:
        return {"available": False, "dominant": "neutral", "scores": {"neutral": 1.0}, "confidence": 1.0, "gaze": "center", "lookAway": False}
    payload = data_url.split(",", 1)[-1]
    image_bytes = base64.b64decode(payload)
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image frame.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Try MediaPipe face detection first for more stable bounding boxes
    x = y = w = h = None
    if MEDIAPIPE_AVAILABLE:
        try:
            mp_fd = mp.solutions.face_detection
            with mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)
                if results.detections:
                    fh, fw = gray.shape[:2]
                    best_area = 0
                    best_box = None
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        bx = int(bbox.xmin * fw)
                        by = int(bbox.ymin * fh)
                        bw = int(bbox.width * fw)
                        bh = int(bbox.height * fh)
                        area = bw * bh
                        if area > best_area:
                            best_area = area
                            best_box = (bx, by, bw, bh)
                    if best_box is not None:
                        x, y, w, h = best_box
        except Exception:
            x = y = w = h = None

    # Fall back to Haar cascade if MediaPipe not available or failed
    if x is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces) == 0:
            return {"available": True, "dominant": "neutral", "scores": {"neutral": 1.0}, "confidence": 1.0, "gaze": "away", "lookAway": check_look_away(session_id, "away")}
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    dominant, scores = analyse_face_roi(gray, x, y, w, h)
    confidence = scores.get(dominant, 0.0)
    gaze = estimate_gaze_direction(gray, (x, y, w, h))
    return {"available": True, "dominant": dominant, "scores": scores, "confidence": confidence, "gaze": gaze, "lookAway": check_look_away(session_id, gaze)}


def extract_voice_features(audio_data, sample_rate):
    features = []
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    features.extend(np.mean(mfcc.T, axis=0))
    stft = np.abs(librosa.stft(audio_data))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    features.extend(np.mean(chroma.T, axis=0))
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    features.extend(np.mean(mel.T, axis=0))
    return np.array(features)


def build_or_load_voice_model(model_path="voice_emotion_mlp.pkl", scaler_path="voice_emotion_scaler.pkl"):
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            return model, scaler
        except Exception:
            return None, None
    return None, None


def classify_voice_emotion(audio_data, sr_rate, model=None, scaler=None):
    if model is not None and scaler is not None:
        features = extract_voice_features(audio_data, sr_rate)
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        return pred, dict(zip(model.classes_, proba.tolist()))

    rms = float(np.sqrt(np.mean(audio_data ** 2)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio_data)[0]))
    s_cent = float(np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr_rate)))
    f0, voiced_flag, _ = librosa.pyin(audio_data, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr_rate)
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    voiced_ratio = float(np.sum(voiced_flag)) / len(f0) if voiced_flag is not None and len(f0) > 0 else 0.0

    if rms < 0.008:
        scores = {e: 0.03 for e in VOICE_EMOTION_LABELS}
        scores["neutral"] = 0.82
        return "neutral", scores

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
    ev["calm"] += 0.10
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
    if 0 < pitch_mean < 130:
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
    if 0 < pitch_mean < 120:
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


def analyse_voice_file(uploaded_file):
    if not VOICE_AVAILABLE:
        return {"available": False, "dominant": "neutral", "scores": {"neutral": 1.0}, "confidence": 1.0, "energy": 0.0}
    suffix = os.path.splitext(uploaded_file.name or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in uploaded_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name
    try:
        audio_data, sr_rate = librosa.load(tmp_path, sr=22050, mono=True)
        energy = float(np.sqrt(np.mean(audio_data ** 2))) if len(audio_data) else 0.0
        if energy < 0.005:
            scores = {e: 0.05 for e in VOICE_EMOTION_LABELS}
            scores["neutral"] = 0.70
            return {"available": True, "dominant": "neutral", "scores": scores, "confidence": scores["neutral"], "energy": energy}
        model, scaler = build_or_load_voice_model()
        dominant, scores = classify_voice_emotion(audio_data, sr_rate, model, scaler)
        return {"available": True, "dominant": dominant, "scores": scores, "confidence": scores.get(dominant, 0.0), "energy": energy}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def fuse_emotions(face_scores=None, voice_scores=None):
    face_scores = face_scores or {"neutral": 1.0}
    voice_scores = voice_scores or {"neutral": 1.0}
    fused = {e: 0.0 for e in UNIFIED_EMOTIONS}
    for label, score in face_scores.items():
        if label in fused:
            fused[label] += float(score) * FACE_WEIGHT
    for label, score in voice_scores.items():
        unified = VOICE_TO_UNIFIED.get(label, label)
        if unified in fused:
            fused[unified] += float(score) * VOICE_WEIGHT
    total = sum(fused.values())
    if total > 0:
        fused = {k: v / total for k, v in fused.items()}
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    dominant, dominant_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if dominant_score < 0.25:
        dominant = "uncertain"
    elif dominant != "neutral" and dominant_score - runner_up_score < 0.08:
        dominant = "neutral"
    return dominant, fused, dominant_score
