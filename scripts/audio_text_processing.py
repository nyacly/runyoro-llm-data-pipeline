import os
import logging

from scripts.audio_processing import process_audio_source
from scripts.text_processing import process_text_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_audio_text_pair(directory_path, output_base_dir, segment_params=None):
    logging.info(f"Processing audio-text pair from directory: {directory_path}")

    files = os.listdir(directory_path)
    audio_file = None
    text_file = None

    SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_AUDIO_EXTS:
            audio_file = os.path.join(directory_path, f)
        elif ext == ".txt":
            text_file = os.path.join(directory_path, f)

    if not audio_file or not text_file:
        logging.error(f"Could not find an audio and a text file in {directory_path}")
        return []

    # Process audio
    audio_output_dir = os.path.join(output_base_dir, "processed_audio")
    os.makedirs(audio_output_dir, exist_ok=True)
    processed_audio_segments = process_audio_source(audio_file, audio_output_dir, **(segment_params or {}))

    # Process text
    text_output_dir = os.path.join(output_base_dir, "processed_text")
    os.makedirs(text_output_dir, exist_ok=True)
    processed_text_filepath = process_text_source(text_file, 'text_file', text_output_dir)

    if not processed_audio_segments or not processed_text_filepath:
        logging.error(f"Failed to process either audio or text for {directory_path}")
        return []

    # For now, we'll just return the metadata.
    # A more sophisticated implementation would align the audio segments with the text.
    return {
        "processed_audio": processed_audio_segments,
        "processed_text": processed_text_filepath
    }
