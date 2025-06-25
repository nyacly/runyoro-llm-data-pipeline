# Runyoro/Rutooro LLM Data Processing Pipeline

This repository contains the implementation of a data processing pipeline for the Runyoro/Rutooro LLM project. The pipeline is designed to ingest and preprocess various data sources, including PDFs, images, websites, audio files, and video files, to prepare them for training a Large Language Model (LLM) in Runyoro/Rutooro.

## Project Structure

- `raw_data/`: Directory for raw, unprocessed data.
- `processed_data/`: Directory for processed and cleaned data, ready for LLM training.
- `scripts/`: Contains Python scripts for different stages of the pipeline.
- `docs/`: Documentation related to the pipeline.
- `requirements.txt`: Lists all Python dependencies.

## Getting Started

This guide provides step-by-step instructions on how to set up and use this data processing pipeline on both a local MacBook Pro M4 and cloud environments like Google Colab or Hugging Face.

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Git**: For cloning the repository.
*   **Python 3.8+**: The pipeline is developed with Python 3.8 and above.
*   **pip**: Python package installer (usually comes with Python).

### 1. Local Setup (MacBook Pro M4)

#### 1.1. Clone the Repository

Open your terminal and clone the repository:

```bash
git clone https://github.com/nyacly/runyoro-llm-data-pipeline.git
cd runyoro-llm-data-pipeline
```

#### 1.2. Install Python Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

#### 1.3. Install System Dependencies

This pipeline relies on external tools for PDF processing (Poppler), OCR (Tesseract), and audio/video manipulation (FFmpeg). Install them using Homebrew:

```bash
brew install poppler tesseract ffmpeg
```

#### 1.4. Run the Pipeline

Place your raw data (PDFs, images, audio, video, text files, or a list of URLs for websites) into the `raw_data/` directory. Then, you can run the `orchestrator.py` script to process your data. The `test_pipeline.py` script provides an example of how to use the orchestrator.

To run the example pipeline:

```bash
python3 scripts/test_pipeline.py
```

This will process the sample data in `raw_data/` and save the processed outputs in `processed_data/`.

To process your own data, you will need to modify `scripts/test_pipeline.py` or create a new script that calls the `process_data_source` function from `scripts/orchestrator.py` with your specific data paths and types.

### 2. Cloud Setup (Google Colab / Hugging Face Spaces)

For cloud environments, the setup process is similar but often simpler due to pre-installed dependencies or easier installation methods.

#### 2.1. Google Colab

1.  **Open a New Colab Notebook**: Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.
2.  **Clone the Repository**: In a code cell, run:
    ```python
    !git clone https://github.com/nyacly/runyoro-llm-data-pipeline.git
    %cd runyoro-llm-data-pipeline
    ```
3.  **Install Python Dependencies**: In a new code cell:
    ```python
    !pip install -r requirements.txt
    ```
4.  **Install System Dependencies**: Colab usually has `ffmpeg` pre-installed. For `poppler` and `tesseract`, you might need to install them via `apt-get`:
    ```python
    !sudo apt-get update
    !sudo apt-get install -y poppler-utils tesseract-ocr
    ```
5.  **Upload Data**: You can upload your raw data directly to the `raw_data/` directory in Colab's file browser or mount your Google Drive.
6.  **Run the Pipeline**: Execute the test script or your custom processing script:
    ```python
    !python3 scripts/test_pipeline.py
    ```

#### 2.2. Hugging Face Spaces

1.  **Create a New Space**: Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space. Choose a Docker-based template or a custom template if you need specific environments.
2.  **Clone the Repository**: You can either directly link your Space to this GitHub repository or clone it within the Space's environment.
3.  **Define Dependencies**: In your `Dockerfile` or `requirements.txt` (if using a Python template), ensure all Python and system dependencies (`poppler-utils`, `tesseract-ocr`, `ffmpeg`) are listed for installation.
4.  **Data Upload**: Upload your raw data to the Space's data directory.
5.  **Run the Pipeline**: Configure your Space's `app.py` or a custom script to run the `orchestrator.py` or `test_pipeline.py` script upon launch or via a Gradio/Streamlit interface.

## Data Input and Expected Output

### Input Data (`raw_data/`)

Place your raw, unprocessed data files into the `raw_data/` directory. The pipeline supports the following data types:

*   **PDFs (`.pdf`)**: Documents containing text and/or images.
*   **Images (`.png`, `.jpg`, `.jpeg`, etc.)**: Image files from which text can be extracted using OCR.
*   **Audio Files (`.mp3`, `.wav`, `.flac`, etc.)**: Audio recordings for transcription or further audio processing.
*   **Video Files (`.mp4`, `.avi`, `.mov`, etc.)**: Video files from which audio tracks can be extracted.
*   **Text Files (`.txt`)**: Plain text documents.
*   **Website URLs**: You can provide a list of URLs (static or dynamic) within a text file, or directly pass them to the `process_data_source` function.

### Expected Output (`processed_data/`)

After running the pipeline, the `processed_data/` directory will contain the cleaned and structured data, organized into subdirectories based on the processed data type:

*   `processed_data/processed_text/`: Contains `.txt` files with extracted and cleaned text from PDFs, images, text files, and websites. Each file will correspond to a processed text source.
*   `processed_data/processed_audio/`: Contains `.wav` files of standardized audio segments extracted from raw audio files.
*   `processed_data/processed_audio_from_video/`: Contains `.wav` files of standardized audio segments extracted from video files.
*   `processed_data/processed_data_metadata.json`: A JSON file that stores metadata for all processed items, including their original source, hash (for deduplication), processed path, data type, and timestamp. This file is crucial for tracking processed data and preventing duplicates.

## Using Processed Data for LLM Training

The data generated in the `processed_data/` directory is now in a format suitable for various stages of LLM training:

1.  **Text Data (`processed_data/processed_text/`)**: These `.txt` files contain clean, preprocessed text. They can be directly used for:
    *   **Pre-training**: For training a new LLM from scratch on the Runyoro/Rutooro language, or for continued pre-training of an existing multilingual LLM to adapt it to Runyoro/Rutooro.
    *   **Fine-tuning**: For task-specific fine-tuning (e.g., text generation, summarization, translation) once a base LLM is available.
    *   **Corpus Building**: To build a comprehensive text corpus for language modeling research.

2.  **Audio Data (`processed_data/processed_audio/` and `processed_data/processed_audio_from_video/`)**: These `.wav` files represent clean audio segments. They are primarily used for:
    *   **Speech-to-Text (STT) Model Training**: To train or fine-tune an STT model specifically for Runyoro/Rutooro. The audio segments can be paired with their corresponding text transcriptions (which you would generate separately, perhaps using a small initial STT model or manual transcription, and then integrate into the metadata).
    *   **Voice Activity Detection (VAD)**: The standardized audio can be used to train VAD models to identify speech segments more accurately.

3.  **Metadata (`processed_data/processed_data_metadata.json`)**: This file serves as a central registry for your dataset. It can be parsed to:
    *   **Manage Dataset Versions**: Track which sources have been processed.
    *   **Create Training Manifests**: Generate manifest files (e.g., CSV, JSONL) that link audio files to their transcriptions, or text files to their source metadata, which are commonly required by LLM and STT training frameworks (e.g., Hugging Face `datasets` library, PyTorch Lightning DataModules).
    *   **Filter and Select Data**: Easily filter data based on source type, processing status, or other metadata fields for specific training tasks.

**Workflow for LLM Training:**

*   **Initial STT Model (MVP)**: Start with a small subset of transcribed audio (manual or semi-automated) to train a basic STT model. This model can then be used to bootstrap transcription of larger audio datasets.
*   **Iterative Data Curation**: As more audio is transcribed and text is extracted, continuously feed this data back into the pipeline. The deduplication feature ensures efficient processing of new data.
*   **LLM Adaptation**: Use the growing Runyoro/Rutooro text corpus to pre-train or fine-tune an LLM. Consider using transfer learning from a large multilingual model if starting from scratch is too resource-intensive.

## Pipeline Components

Refer to the individual Python files in the `scripts/` directory for detailed information on each component:

*   `scripts/core_components.py`: Contains fundamental functions for text extraction, audio/video processing, and initial data cleaning.
*   `scripts/text_processing.py`: Handles text-specific data extraction and preprocessing.
*   `scripts/audio_processing.py`: Manages audio standardization and segmentation.
*   `scripts/video_processing.py`: Extracts and processes audio from video sources.
*   `scripts/orchestrator.py`: The central script for managing the data flow, including duplicate detection and metadata management.
*   `scripts/test_pipeline.py`: An example script to demonstrate the pipeline's functionality.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.


