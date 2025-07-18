import os
import json
import logging
import wave
import math
import struct

from scripts.orchestrator import process_data_source


def _create_dummy_wav(path, duration_sec=1, freq=440, sample_rate=16000):
    """Create a simple sine wave .wav file for testing."""
    if os.path.exists(path):
        return

    n_samples = int(duration_sec * sample_rate)
    amplitude = 32767
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            sample = int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
            wf.writeframes(struct.pack("<h", sample))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_pipeline_test():
    raw_data_dir = "./raw_data"
    processed_data_dir = "./processed_data"
    metadata_file = os.path.join(processed_data_dir, "processed_data_metadata.json")

    os.makedirs(processed_data_dir, exist_ok=True)

    # Test PDF processing
    pdf_path = os.path.join(raw_data_dir, "RUNYORO-RUTOORO-RADIO-SCRIPT.pdf")
    if os.path.exists(pdf_path):
        logging.info(f"\n--- Processing PDF: {pdf_path} ---")
        process_data_source(pdf_path, 'pdf', processed_data_dir, metadata_file)
    else:
        logging.warning(f"PDF file not found: {pdf_path}")

    # Test Audio processing
    audio_path = os.path.join(raw_data_dir, "Rutooro_Love_Tooro_Artist.mp3")
    if os.path.exists(audio_path):
        logging.info(f"\n--- Processing Audio: {audio_path} ---")
        process_data_source(audio_path, 'audio', processed_data_dir, metadata_file)
    else:
        logging.warning(f"Audio file not found: {audio_path}")

    # Test Image processing (assuming the image is text-heavy for OCR)
    image_path = os.path.join(raw_data_dir, "Runyoro_Rutooro_Pulpit_Bible.jpeg")
    if os.path.exists(image_path):
        logging.info(f"\n--- Processing Image: {image_path} ---")
        process_data_source(image_path, 'image', processed_data_dir, metadata_file)
    else:
        logging.warning(f"Image file not found: {image_path}")

    # Test Static Website processing
    static_website_url = "https://www.w3schools.com/howto/howto_website_static.asp"
    logging.info(f"\n--- Processing Static Website: {static_website_url} ---")
    process_data_source(static_website_url, 'website_static', processed_data_dir, metadata_file)

    # Test Text File processing
    text_file_path = os.path.join(raw_data_dir, "w3schools_static_website_example.txt")
    if os.path.exists(text_file_path):
        logging.info(f"\n--- Processing Text File: {text_file_path} ---")
        process_data_source(text_file_path, 'text_file', processed_data_dir, metadata_file)
    else:
        logging.warning(f"Text file not found: {text_file_path}")

    # Test Audio/Text Pair directory processing
    at_pair_dir = os.path.join(raw_data_dir, "test_audio_pairs")
    if os.path.isdir(at_pair_dir):
        # Create dummy audio files if they don't exist
        _create_dummy_wav(os.path.join(at_pair_dir, "sample1.wav"))
        _create_dummy_wav(os.path.join(at_pair_dir, "sample2.wav"), freq=660)
        logging.info(f"\n--- Processing Audio/Text Pairs: {at_pair_dir} ---")
        process_data_source(at_pair_dir, 'audio_text_pair', processed_data_dir, metadata_file)
    else:
        logging.warning(f"Audio/Text pair directory not found: {at_pair_dir}")

    logging.info("\n--- Pipeline Test Complete ---")
    logging.info(f"Check {processed_data_dir} for processed data and {metadata_file} for metadata.")

    # Verify metadata file content
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            for entry in metadata:
                if entry.get("source_type") == "audio_text_pair":
                    logging.info(
                        f"  - Pair {entry.get('pair_id')}: {entry.get('processed_audio_paths')}"
                        f" + {entry.get('processed_text_path')}"
                    )
                else:
                    logging.info(
                        f"  - Source: {entry.get('original_source')}, Type: {entry.get('source_type')}, Processed Path: {entry.get('processed_path')}"
                    )
            logging.info(f"Metadata entries: {len(metadata)}")
    else:
        logging.error(f"Metadata file not found: {metadata_file}")

if __name__ == "__main__":
    run_pipeline_test()


