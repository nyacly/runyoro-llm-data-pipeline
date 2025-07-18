import os
import logging
import re
from pdf2image import convert_from_path

from scripts.core_components import (
    extract_text_from_pdf,
    extract_text_from_image_ocr,
    scrape_static_website,
    scrape_dynamic_website,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_and_preprocess_text(text):
    """Clean and normalize extracted text while keeping Unicode letters.

    The function removes unwanted symbols but preserves non-ASCII
    characters. Whitespace is normalised without discarding newlines.
    """

    if not isinstance(text, str):
        return ""

    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")

    # Remove characters that are not word characters, whitespace or common
    # punctuation while supporting Unicode letters.
    text = re.sub(r'[^\w\s.,?!-]', '', text, flags=re.UNICODE)

    # Normalise spaces and tabs but keep newlines to maintain sentence breaks
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse multiple line breaks into a single one
    text = re.sub(r'\n{2,}', '\n', text)

    return text.strip()

def process_text_source(source_path_or_url, source_type, output_dir):
    raw_text = ""
    if source_type == 'pdf':
        raw_text = extract_text_from_pdf(source_path_or_url)
        if raw_text is None or not raw_text.strip():
            logging.info(f"Native PDF extraction failed or empty for {source_path_or_url}, attempting OCR...")
            try:
                images = convert_from_path(source_path_or_url)
                ocr_text = ""
                for i, image in enumerate(images):
                    ocr_text += extract_text_from_image_ocr(image)
                raw_text = ocr_text
            except ImportError:
                logging.error("pdf2image not installed. Cannot perform OCR on PDF.")
                return None
            except Exception as e:
                logging.error(f"Error converting PDF to image for OCR {source_path_or_url}: {e}")
                return None
    elif source_type == 'image':
        raw_text = extract_text_from_image_ocr(source_path_or_url)
    elif source_type == 'website_static':
        raw_text = scrape_static_website(source_path_or_url)
    elif source_type == 'website_dynamic':
        raw_text = scrape_dynamic_website(source_path_or_url)
    elif source_type == 'text_file':
        try:
            with open(source_path_or_url, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except Exception as e:
            logging.error(f"Error reading text file {source_path_or_url}: {e}")
            return None
    else:
        logging.error(f"Unsupported source type for text processing: {source_type}")
        return None

    if not raw_text or not raw_text.strip():
        logging.warning(f"No text extracted from {source_path_or_url} ({source_type}).")
        return None

    processed_text = clean_and_preprocess_text(raw_text)

    base_name = os.path.basename(source_path_or_url) if source_type not in ['website_static', 'website_dynamic'] else source_path_or_url.replace('https://', '').replace('http://', '').replace('/', '_')
    output_filename = f"{os.path.splitext(base_name)[0]}_{source_type}.txt"
    output_filepath = os.path.join(output_dir, output_filename)
    
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(processed_text)
        logging.info(f"Successfully processed and saved text from {source_path_or_url} to {output_filepath}")
        return output_filepath
    except Exception as e:
        logging.error(f"Error saving processed text to {output_filepath}: {e}")
        return None


