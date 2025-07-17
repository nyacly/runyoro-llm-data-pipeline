import os
import json
import logging

from scripts.orchestrator import process_data_source

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

    logging.info("\n--- Pipeline Test Complete ---")
    logging.info(f"Check {processed_data_dir} for processed data and {metadata_file} for metadata.")

    # Verify metadata file content
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            for entry in metadata:
                logging.info(f"  - Source: {entry.get('original_source')}, Type: {entry.get('source_type')}, Processed Path: {entry.get('processed_path')}")
            logging.info(f"Metadata entries: {len(metadata)}")
    else:
        logging.error(f"Metadata file not found: {metadata_file}")

if __name__ == "__main__":
    run_pipeline_test()


