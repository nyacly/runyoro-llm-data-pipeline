
import os
import logging

from scripts.core_components import extract_audio_from_video_ffmpeg
from scripts.audio_processing import process_audio_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_video_source(video_path, audio_output_dir, segment_params=None):
    if segment_params is None:
        segment_params = {
            "min_silence_len": 1000,
            "silence_thresh": -40,
            "keep_silence": 200
        }

    extracted_audio_path = os.path.join(audio_output_dir, f"extracted_audio_{os.path.basename(video_path)}.wav")
    extracted_audio_path = extract_audio_from_video_ffmpeg(video_path, extracted_audio_path)

    if not extracted_audio_path:
        return []

    logging.info(f"Extracted audio from video {video_path}. Now processing audio...")
    processed_segments = process_audio_source(
        extracted_audio_path,
        audio_output_dir,
        **(segment_params or {})
    )
    return processed_segments


