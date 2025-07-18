import os
import logging

from scripts.audio_processing import process_audio_source
from scripts.text_processing import process_text_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_audio_text_pairs(directory_path, output_base_dir, segment_params=None):
    """Process a directory containing multiple audio/text pairs.

    Audio and transcript files must share the same base name, e.g. ``sample.wav``
    and ``sample.txt``. Each detected pair is standardised and saved to the
    processed data directories.
    """
    logging.info(f"Processing audio-text pairs from directory: {directory_path}")

    files = os.listdir(directory_path)

    SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

    audio_files = {}
    text_files = {}
    for f in files:
        base, ext = os.path.splitext(f)
        ext = ext.lower()
        if ext in SUPPORTED_AUDIO_EXTS:
            audio_files[base] = os.path.join(directory_path, f)
        elif ext == ".txt":
            text_files[base] = os.path.join(directory_path, f)

    common_basenames = sorted(set(audio_files) & set(text_files))
    if not common_basenames:
        logging.error(
            f"Could not find matching audio/text pairs in {directory_path}"
        )
        return []

    results = []
    audio_output_dir = os.path.join(output_base_dir, "processed_audio")
    os.makedirs(audio_output_dir, exist_ok=True)
    text_output_dir = os.path.join(output_base_dir, "processed_text")
    os.makedirs(text_output_dir, exist_ok=True)

    for base in common_basenames:
        audio_file = audio_files[base]
        text_file = text_files[base]

        processed_audio_segments = process_audio_source(
            audio_file, audio_output_dir, **(segment_params or {})
        )
        processed_text_filepath = process_text_source(
            text_file, "text_file", text_output_dir
        )

        if not processed_audio_segments or not processed_text_filepath:
            logging.error(
                f"Failed to process audio or text for pair '{base}' in {directory_path}"
            )
            continue

        results.append(
            {
                "pair_id": base,
                "processed_audio": processed_audio_segments,
                "processed_text": processed_text_filepath,
            }
        )

    return results
