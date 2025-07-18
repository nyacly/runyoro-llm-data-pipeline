import os
import logging

from scripts.orchestrator import process_data_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def detect_source_type(path: str):
    if os.path.isdir(path):
        files = os.listdir(path)
        has_audio = any(os.path.splitext(f)[1].lower() in SUPPORTED_AUDIO_EXTS for f in files)
        has_text = any(os.path.splitext(f)[1].lower() == ".txt" for f in files)
        if has_audio and has_text:
            return "audio_text_pair"
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext in SUPPORTED_IMAGE_EXTS:
        return "image"
    if ext == ".txt":
        return "text_file"
    if ext in SUPPORTED_AUDIO_EXTS:
        return "audio"
    if ext in SUPPORTED_VIDEO_EXTS:
        return "video"
    return None


def process_all_raw_data(raw_data_dir: str = "./raw_data", processed_data_dir: str = "./processed_data"):
    metadata_file = os.path.join(processed_data_dir, "processed_data_metadata.json")
    os.makedirs(processed_data_dir, exist_ok=True)

    for name in os.listdir(raw_data_dir):
        path = os.path.join(raw_data_dir, name)
        source_type = detect_source_type(path)
        if not source_type:
            if os.path.isdir(path) or os.path.isfile(path):
                logging.warning(f"Skipping unsupported file type or directory structure: {path}")
            continue
        logging.info(f"Processing {path} as {source_type}")
        process_data_source(path, source_type, processed_data_dir, metadata_file)

    logging.info("Processing complete.")


if __name__ == "__main__":
    process_all_raw_data()
