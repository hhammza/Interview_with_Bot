from collections import Counter
from pathlib import Path
from urllib.parse import urljoin
import zipfile

from django.conf import settings
from django.core.files.storage import default_storage
from django.http import FileResponse
from django.utils import timezone
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework import status

from .ai import ask_interviewer, opening_question, tts_data_url
from .emotion import analyse_face_image, analyse_voice_file, fuse_emotions
from .session_store import store
from .media_merger import merge_video_and_audio


@api_view(["GET"])
def health(_request):
    return Response({"ok": True})


@api_view(["POST"])
def start_interview(request):
    session = store.create(request.data.get("job_role", "Software Engineer"))
    greeting = opening_question(session)
    return Response(
        {
            "sessionId": str(session.id),
            "jobRole": session.job_role,
            "reply": greeting,
            "audioUrl": tts_data_url(greeting),
        },
        status=status.HTTP_201_CREATED,
    )


@api_view(["POST"])
def send_message(request, session_id):
    session = store.get(session_id)
    text = (request.data.get("text") or "").strip()
    if not text:
        return Response({"detail": "text is required"}, status=status.HTTP_400_BAD_REQUEST)
    reply = ask_interviewer(session, text)
    return Response({"reply": reply, "audioUrl": tts_data_url(reply), "mood": session.current_mood()})


@api_view(["POST"])
def analyse_face_frame(request, session_id):
    session = store.get(session_id)
    data_url = request.data.get("image")
    if not data_url:
        return Response({"detail": "image data URL is required"}, status=status.HTTP_400_BAD_REQUEST)
    result = analyse_face_image(str(session_id), data_url)
    session.face_log.append({**result, "timestamp": timezone.now().isoformat()})
    latest_voice = session.voice_log[-1]["scores"] if session.voice_log else {"neutral": 1.0}
    fused_dominant, fused_scores, fused_confidence = fuse_emotions(result.get("scores"), latest_voice)
    session.fused_window.append(fused_dominant)
    session.fused_log.append(
        {
            "timestamp": timezone.now().isoformat(),
            "dominant": fused_dominant,
            "scores": fused_scores,
            "confidence": fused_confidence,
            "source": "face-frame",
        }
    )
    return Response({**result, "fused": {"dominant": fused_dominant, "scores": fused_scores}})


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def analyse_voice_sample(request, session_id):
    session = store.get(session_id)
    sample = request.FILES.get("audio")
    if not sample:
        return Response({"detail": "audio file is required"}, status=status.HTTP_400_BAD_REQUEST)
    result = analyse_voice_file(sample)
    session.voice_log.append({**result, "timestamp": timezone.now().isoformat()})
    latest_face = session.face_log[-1]["scores"] if session.face_log else {"neutral": 1.0}
    fused_dominant, fused_scores, fused_confidence = fuse_emotions(latest_face, result.get("scores"))
    session.fused_window.append(fused_dominant)
    session.fused_log.append(
        {
            "timestamp": timezone.now().isoformat(),
            "dominant": fused_dominant,
            "scores": fused_scores,
            "confidence": fused_confidence,
            "source": "voice-sample",
        }
    )
    return Response({**result, "fused": {"dominant": fused_dominant, "scores": fused_scores}})


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_recording(request, session_id):
    session = store.get(session_id)
    folder = Path("recordings") / str(session_id)
    saved_files = {}
    for field_name in ["recording", "video", "audio"]:
        uploaded_file = request.FILES.get(field_name)
        if not uploaded_file:
            continue
        filename = default_storage.save(str(folder / uploaded_file.name), uploaded_file)
        session.recording_paths[field_name] = Path(default_storage.path(filename))
        saved_files[field_name] = settings.MEDIA_URL + filename.replace("\\", "/")
    if not saved_files:
        return Response(
            {"detail": "recording, video, or audio file is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    return Response({"files": saved_files})


@api_view(["POST"])
def finish_interview(_request, session_id):
    session = store.get(session_id)
    transcript_path = store.save_transcript(session)
    emotions = [entry.get("dominant", "neutral") for entry in session.fused_log] or list(session.fused_window)
    counts = Counter(emotions)
    total = max(len(emotions), 1)
    report = {
        "fusedCounts": dict(counts),
        "confidentCalmPercent": round(((counts.get("happy", 0) + counts.get("neutral", 0)) / total) * 100, 1),
        "anxiousStressedPercent": round(((counts.get("fear", 0) + counts.get("sad", 0) + counts.get("disgust", 0)) / total) * 100, 1),
        "defensiveAngryPercent": round((counts.get("angry", 0) / total) * 100, 1),
    }
    return Response(
        {
            "transcriptUrl": urljoin(settings.MEDIA_URL, f"transcripts/{transcript_path.name}"),
            "transcriptDownloadUrl": f"/api/interviews/{session.id}/transcript/",
            "transcriptFilename": transcript_path.name,
            "bundleDownloadUrl": f"/api/interviews/{session.id}/bundle/",
            "bundleFilename": f"interview-{session.id}.zip",
            "report": report,
        }
    )


@api_view(["GET"])
def download_transcript(_request, session_id):
    session = store.get(session_id)
    if session.transcript_path is None or not session.transcript_path.exists():
        transcript_path = store.save_transcript(session)
    else:
        transcript_path = session.transcript_path
    return FileResponse(
        transcript_path.open("rb"),
        as_attachment=True,
        filename=transcript_path.name,
        content_type="text/plain",
    )


@api_view(["GET"])
def download_bundle(_request, session_id):
    session = store.get(session_id)
    if session.transcript_path is None or not session.transcript_path.exists():
        transcript_path = store.save_transcript(session)
    else:
        transcript_path = session.transcript_path

    bundle_dir = Path(settings.MEDIA_ROOT) / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = f"interview-{session.id}.zip"
    bundle_path = bundle_dir / bundle_name
    folder_name = f"interview-{session.id}"

    # Merge video and audio if both exist
    merged_video_path = None
    if session.recording_paths.get("video") and session.recording_paths.get("audio"):
        video_file = session.recording_paths["video"]
        audio_file = session.recording_paths["audio"]
        
        if video_file.exists() and audio_file.exists():
            # Create merged video file in bundles directory
            merged_filename = f"interview_session_{session.id}.mp4"
            merged_video_path = bundle_dir / merged_filename
            
            try:
                merge_video_and_audio(
                    video_path=video_file,
                    audio_path=audio_file,
                    output_path=merged_video_path,
                    session_id=session.id,
                )
            except Exception as e:
                # Log error but don't fail bundle generation
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to merge video and audio for session {session.id}: {str(e)}")
                merged_video_path = None

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        # Add transcript
        bundle.write(transcript_path, arcname=f"{folder_name}/{transcript_path.name}")
        
        # Add merged video if available
        if merged_video_path and merged_video_path.exists():
            bundle.write(merged_video_path, arcname=f"{folder_name}/{merged_video_path.name}")
        
        # Add original video and audio files
        for media_type, media_path in session.recording_paths.items():
            if media_path.exists():
                prefix = "camera-video" if media_type == "video" else "microphone-audio" if media_type == "audio" else media_type
                bundle.write(media_path, arcname=f"{folder_name}/{prefix}-{media_path.name}")

    return FileResponse(
        bundle_path.open("rb"),
        as_attachment=True,
        filename=bundle_name,
        content_type="application/zip",
    )
