import os
import PyPDF2
from PIL import Image
import pytesseract
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import re
from pydub import AudioSegment
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Text Extraction Components ---

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() or ""
        return text
    except Exception as e:
        logging.error(f"Error reading native PDF {pdf_path}: {e}")
        return None

def extract_text_from_image_ocr(image_or_path):
    """
    Accepts either a file path (str) or a PIL Image object and performs OCR.
    """
    try:
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path)
        else:
            image = image_or_path
        return pytesseract.image_to_string(image, lang="eng")  # Adjust lang as needed
    except Exception as e:
        logging.error(f"Error processing image {image_or_path} with OCR: {e}")
        return ""

def scrape_static_website(url):
    try:
        # Use a timeout to avoid hanging indefinitely on slow or unresponsive servers
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        main_content = soup.find("article") or soup.find("main") or soup.find("div", class_="content")
        if main_content:
            return main_content.get_text(separator=" ", strip=True)
        else:
            return soup.get_text(separator=" ", strip=True)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping static website {url}: {e}")
        return ""

def scrape_dynamic_website(url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = None
    try:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        driver.get(url)
        driver.implicitly_wait(5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        main_content = soup.find("article") or soup.find("main") or soup.find("div", class_="content")
        if main_content:
            return main_content.get_text(separator=" ", strip=True)
        else:
            return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        logging.error(f"Error scraping dynamic website {url}: {e}")
        return ""
    finally:
        if driver:
            driver.quit()

# --- Audio/Video Processing Components ---

def process_audio_file_standardize(input_audio_path, output_path, target_sr=16000, target_channels=1):
    try:
        audio = AudioSegment.from_file(input_audio_path)
        audio = audio.set_frame_rate(target_sr)
        audio = audio.set_channels(target_channels)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        logging.error(f"Error standardizing audio file {input_audio_path}: {e}")
        return None

def extract_audio_from_video_ffmpeg(input_video_path, output_audio_path, target_sr=16000, target_channels=1):
    try:
        command = [
            "ffmpeg",
            "-i", input_video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(target_sr),
            "-ac", str(target_channels),
            output_audio_path
        ]
        subprocess.run(command, check=True, capture_output=True)
        return output_audio_path
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error extracting audio from {input_video_path}: {e.stderr.decode()}")
        return None
    except Exception as e:
        logging.error(f"Error extracting audio from video {input_video_path}: {e}")
        return None

# --- Data Validation & Initial Cleaning ---

def clean_text_initial(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = " ".join(text.split())
    return text.strip()

def validate_audio_file(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        if audio.duration_seconds == 0:
            logging.warning(f"Audio file {audio_path} has zero duration.")
            return False
        return True
    except Exception as e:
        logging.error(f"Invalid audio file {audio_path}: {e}")
        return False


