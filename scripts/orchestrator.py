import os
import json
import hashlib
import logging
import requests

from scripts.text_processing import process_text_source
from scripts.audio_processing import process_audio_source
from scripts.video_processing import process_video_source
from scripts.audio_text_processing import process_audio_text_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def calculate_file_hash(filepath, hash_algo="md5"):
    hasher = hashlib.md5() if hash_algo == "md5" else hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_canonical_url(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.url
    except requests.exceptions.RequestException:
        return url

def save_metadata(metadata_list, metadata_filepath):
    try:
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=4)
        logging.info(f"Metadata saved to {metadata_filepath}")
    except Exception as e:
        logging.error(f"Error saving metadata to {metadata_filepath}: {e}")

def process_data_source(source_path_or_url, source_type, output_base_dir, metadata_filepath, segment_params=None):
    processed_outputs_metadata = []

    existing_metadata = []
    if os.path.exists(metadata_filepath):
        try:
            with open(metadata_filepath, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Metadata file {metadata_filepath} is corrupted or empty. Starting fresh.")
            existing_metadata = []

    original_source_identifier = source_path_or_url
    original_source_content_hash = None

    if source_type in ["pdf", "image", "text_file", "audio", "video"]:
        if not os.path.exists(source_path_or_url):
            logging.error(f"File not found: {source_path_or_url}")
            return []
        original_source_content_hash = calculate_file_hash(source_path_or_url)
    elif source_type == "audio_text_pair":
        # For a directory, we'll create a hash of the file list as a simple way to check for changes.
        try:
            file_list = sorted(os.listdir(source_path_or_url))
            hasher = hashlib.md5()
            for filename in file_list:
                hasher.update(filename.encode('utf-8'))
            original_source_content_hash = hasher.hexdigest()
        except OSError as e:
            logging.error(f"Error reading directory {source_path_or_url}: {e}")
            return []
    elif source_type in ["website_static", "website_dynamic"]:
        original_source_identifier = get_canonical_url(source_path_or_url)
    else:
        logging.error(f"Unknown source type: {source_type}")
        return []

    is_duplicate = False
    for entry in existing_metadata:
        if entry.get("original_source") == original_source_identifier and \
           (original_source_content_hash is None or entry.get("original_source_hash") == original_source_content_hash):
            logging.info(f"Skipping duplicate source: {source_path_or_url} (already processed).")
            is_duplicate = True
            break

    if is_duplicate:
        return []

    if source_type in ["pdf", "image", "website_static", "website_dynamic", "text_file"]:
        text_output_dir = os.path.join(output_base_dir, "processed_text")
        os.makedirs(text_output_dir, exist_ok=True)
        processed_filepath = process_text_source(source_path_or_url, source_type, text_output_dir)
        if processed_filepath:
            processed_outputs_metadata.append({
                "source_type": source_type,
                "original_source": original_source_identifier,
                "original_source_hash": original_source_content_hash,
                "processed_path": processed_filepath,
                "data_type": "text",
                "timestamp": "2025-06-20T10:00:00Z"
            })
    elif source_type == "audio":
        audio_output_dir = os.path.join(output_base_dir, "processed_audio")
        os.makedirs(audio_output_dir, exist_ok=True)
        processed_segments = process_audio_source(source_path_or_url, audio_output_dir, **(segment_params or {}))
        for segment_info in processed_segments:
            processed_outputs_metadata.append({
                "source_type": source_type,
                "original_source": original_source_identifier,
                "original_source_hash": original_source_content_hash,
                "processed_path": segment_info["path"],
                "segment_index": segment_info["segment_index"],
                "data_type": "audio_segment",
                "timestamp": "2025-06-20T10:00:00Z"
            })
    elif source_type == "video":
        audio_output_dir = os.path.join(output_base_dir, "processed_audio_from_video")
        os.makedirs(audio_output_dir, exist_ok=True)
        processed_segments = process_video_source(source_path_or_url, audio_output_dir, segment_params)
        for segment_info in processed_segments:
            processed_outputs_metadata.append({
                "source_type": source_type,
                "original_source": original_source_identifier,
                "original_source_hash": original_source_content_hash,
                "processed_path": segment_info["path"],
                "segment_index": segment_info["segment_index"],
                "data_type": "audio_segment_from_video",
                "timestamp": "2025-06-20T10:00:00Z"
            })
    elif source_type == "audio_text_pair":
        processed_pairs = process_audio_text_pairs(
            source_path_or_url, output_base_dir, segment_params
        )
        for pair in processed_pairs:
            processed_outputs_metadata.append(
                {
                    "source_type": source_type,
                    "original_source": original_source_identifier,
                    "pair_id": pair["pair_id"],
                    "processed_audio_paths": [seg["path"] for seg in pair["processed_audio"]],
                    "processed_text_path": pair["processed_text"],
                    "data_type": "audio_text_aligned",
                    "timestamp": "2025-06-20T10:00:00Z",
                }
            )
    else:
        logging.error(f"Unknown source type: {source_type}")

    all_metadata = existing_metadata + processed_outputs_metadata
    save_metadata(all_metadata, metadata_filepath)

    return processed_outputs_metadata


