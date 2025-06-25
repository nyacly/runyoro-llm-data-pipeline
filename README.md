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

