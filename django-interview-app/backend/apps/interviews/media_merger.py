"""
Media merger utility for combining video and audio files.
Combines separate video and audio recordings into a single MP4 file.
"""

import subprocess
from pathlib import Path
from uuid import UUID
import logging

logger = logging.getLogger(__name__)


def merge_video_and_audio(
    video_path: Path | str,
    audio_path: Path | str,
    output_path: Path | str,
    session_id: UUID | str = None,
) -> Path:
    """
    Merge video and audio files into a single MP4 file using ffmpeg.
    
    Args:
        video_path: Path to the input video file (WebM)
        audio_path: Path to the input audio file (WebM)
        output_path: Path where the output MP4 file will be saved
        session_id: Optional session ID for logging purposes
    
    Returns:
        Path object pointing to the created MP4 file
    
    Raises:
        FileNotFoundError: If input files don't exist
        subprocess.CalledProcessError: If ffmpeg command fails
        Exception: For other unexpected errors
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    
    # Validate input files exist
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove output file if it already exists
    if output_path.exists():
        output_path.unlink()
    
    try:
        # Use ffmpeg to combine video and audio
        # -i: input file
        # -c:v copy: copy video codec without re-encoding (fast)
        # -c:a aac: encode audio to AAC (MP4 compatible)
        # -shortest: output shortest stream (video or audio)
        # -y: overwrite output file without asking
        command = [
            "ffmpeg",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",  # Copy video codec without re-encoding
            "-c:a", "aac",   # Use AAC for audio (MP4 compatible)
            "-shortest",     # Use shortest stream duration
            "-y",            # Overwrite output without asking
            str(output_path),
        ]
        
        # Run ffmpeg command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        
        logger.info(f"Successfully merged video and audio for session {session_id} to {output_path}")
        return output_path
    
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg merge failed for session {session_id}: {e.stderr}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
    except Exception as e:
        error_msg = f"Error merging video and audio for session {session_id}: {str(e)}"
        logger.error(error_msg)
        raise
