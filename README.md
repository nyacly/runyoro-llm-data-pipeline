# Runyoro/Rutooro LLM Data Processing and Training Pipeline

This repository contains a comprehensive pipeline for processing diverse data sources and training a Large Language Model (LLM) for the Runyoro/Rutooro language. It handles data ingestion, preprocessing, and then facilitates the training and evaluation of an LLM.

## Project Structure

- `raw_data/`: Directory for raw, unprocessed data.
- `processed_data/`: Directory for processed and cleaned data, ready for LLM training.
- `scripts/`: Contains Python scripts for different stages of the pipeline, including data processing, LLM training, and testing.
- `models/`: Directory to store trained LLM models.
- `configs/`: (Optional) Configuration files for training parameters.
- `docs/`: Documentation related to the pipeline.
- `requirements.txt`: Lists all Python dependencies.

## Getting Started

This guide provides step-by-step instructions on how to set up and use this data processing and training pipeline on both a local MacBook Pro M4 and cloud environments like Google Colab or Hugging Face.

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Git**: For cloning the repository.
*   **Python 3.8+**: The scripts are developed with Python 3.8 and above.
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

#### 1.4. Run the Data Processing Pipeline

Place your raw data (PDFs, images, audio, video, text files, or a list of URLs for websites) into the `raw_data/` directory. The repository now includes a convenience script that will iterate over everything in this folder and process it automatically.

To run the data processing pipeline on all files in `raw_data/`:

```bash
python3 scripts/process_raw_data.py
```

This script detects the file type (PDF, image, audio, video, or text) and calls the orchestrator accordingly. Processed outputs are saved in `processed_data/`, and metadata is written to `processed_data/processed_data_metadata.json`.

#### Example: Processing an Audio/Text Pair Directory

If you have a folder containing multiple audio files with matching transcript
files, you can process the entire directory at once. The audio and text files
must share the same base filename, such as `sample1.wav` and `sample1.txt`.

```
raw_data/test_audio_pairs/
├── sample1.wav
├── sample1.txt
├── sample2.wav
└── sample2.txt
```

To process this directory manually using the orchestrator:

```python
from scripts.orchestrator import process_data_source

process_data_source(
    "./raw_data/test_audio_pairs",
    "audio_text_pair",
    "./processed_data",
    "./processed_data/processed_data_metadata.json",
)
```

Each detected pair will appear in `processed_data_metadata.json` with a
`pair_id` linking the processed audio segments to the corresponding transcript
file, enabling easy alignment for STT model training.

`scripts/test_pipeline.py` remains as a simple example showing how to call `process_data_source` directly if you need more control or want to integrate specific sources (like websites) manually.

#### Running Training Locally vs. Google Colab

Data processing generally runs well on a MacBook Pro, but the first stage of
LLM training (`scripts/train_llm.py`) can easily exhaust laptop memory.  If you
see crashes during training, switch to Google Colab or another GPU-enabled
environment.  This repository now provides Colab notebooks under
`notebooks/` that automate the heavy training step.  After training finishes you
can copy the `models/` directory back to your MacBook and continue with local
testing or further fine‑tuning.

### 2. LLM Training

Once your data has been processed and is available in `processed_data/processed_text/`, you can proceed with training your LLM.

#### 2.1. Run the Training Script

The `scripts/train_llm.py` script is used to train the LLM. You can specify various parameters such as the path to your processed text data, the base model to use (e.g., `gpt2`), and the output directory for the trained model.

```bash
python3 scripts/train_llm.py \
    --processed_data_path ./processed_data/processed_text \
    --model_name gpt2 \
    --output_dir ./models/runyoro_llm_model
```

**Key Parameters:**

*   `--processed_data_path`: Path to the directory containing your processed `.txt` files (e.g., `./processed_data/processed_text`). Each `.txt` file will be treated as a document for training.
*   `--model_name`: The name of a pre-trained model from Hugging Face Transformers to use as a base. For initial experimentation, `gpt2` is a good starting point. For low-resource languages, you might consider smaller models or multilingual models that can be further fine-tuned.
*   `--output_dir`: The directory where the trained model and tokenizer will be saved.

**Expected Results during Training:**

During training, you will see logs indicating the progress, including:

*   **Loss values**: These should generally decrease over time, indicating that the model is learning.
*   **Evaluation metrics**: If `eval_steps` is set (within `scripts/train_llm.py`), the model will be evaluated periodically on a validation set, showing metrics like validation loss.
*   **Model checkpoints**: Intermediate models will be saved in the `output_dir` at specified `save_steps`.

Upon successful completion, a `final_model` directory will be created inside your specified `output_dir` (e.g., `./models/runyoro_llm_model/final_model`), containing the trained model weights and tokenizer files.

### 3. Model Testing and Evaluation

#### 3.1. Run the Testing Script

The `scripts/test_llm.py` script allows you to test your trained LLM by generating text based on a given prompt.

```bash
python3 scripts/test_llm.py \
    --model_path ./models/runyoro_llm_model/final_model \
    --prompt "Ekiro kyona" \
    --max_new_tokens 100
```

**Key Parameters:**

*   `--model_path`: Path to the directory containing your trained model and tokenizer (the `final_model` directory from training).
*   `--prompt`: An optional initial text prompt in Runyoro/Rutooro for the model to continue. If no prompt is provided, the model will generate text from scratch.
*   `--max_new_tokens`: The maximum number of new tokens (words/subwords) the model should generate.

**Expected Results from Testing:**

The script will output the generated text sequences. The quality of the generated text will depend on:

*   **Amount and quality of training data**: More diverse and relevant Runyoro/Rutooro text data will lead to better generation.
*   **Training epochs**: Sufficient training time is crucial.
*   **Model size and architecture**: Larger models generally capture more complex language patterns.

Initially, with a small dataset and a base model like GPT-2, the generated text might not be perfectly coherent or grammatically correct in Runyoro/Rutooro. However, you should observe that the model starts to generate sequences that resemble the Runyoro/Rutooro language in terms of character patterns and some common words, indicating that it has learned from your data.

### 4. Cloud Setup (Google Colab / Hugging Face Spaces)

For cloud environments, the setup process is similar but often simpler due to pre-installed dependencies or easier installation methods.

#### 4.1. Google Colab

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
5.  **Upload Data**: If your raw data is in Google Drive, mount your Drive and copy/symlink the data to `raw_data/`:
    ```python
    from google.colab import drive
    drive.mount("/content/drive")
    !cp -r "/content/drive/MyDrive/path/to/your/raw_data/*" raw_data/
    ```
    Alternatively, upload directly to the `raw_data/` directory.
6.  **Use the Provided Notebooks**: Instead of manually running each command,
    open the notebooks in the `notebooks/` directory (`01_Process_Data.ipynb`,
    `02_Train_LLM_Stage1.ipynb`, and `03_Test_LLM.ipynb`).  These notebooks
    contain the commands needed for processing, training, and testing on Colab.

#### 4.2. Hugging Face Spaces

1.  **Create a New Space**: Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space. Choose a Docker-based template or a custom template.
2.  **Clone the Repository**: Link your Space to this GitHub repository.
3.  **Define Dependencies**: Ensure `requirements.txt` is in your Space, and if using a Dockerfile, include commands to install system dependencies (e.g., `apt-get install ...`).
4.  **Data Upload**: Upload your raw data to the Space's data directory or configure a persistent storage solution.
5.  **Run the Pipeline**: Configure your Space's `app.py` or a custom script to run the processing, training, and testing scripts. For long-running training jobs, consider using Hugging Face's training APIs or a dedicated training infrastructure.

## Data Input and Expected Output

### Input Data (`raw_data/`)

Place your raw, unprocessed data files into the `raw_data/` directory. The pipeline supports the following data types:

*   **PDFs (`.pdf`)**: Documents containing text and/or images.
*   **Images (`.png`, `.jpg`, `.jpeg`, etc.)**: Image files from which text can be extracted using OCR.
*   **Audio Files (`.mp3`, `.wav`, `.flac`, etc.)**: Audio recordings for transcription or further audio processing.
*   **Video Files (`.mp4`, `.avi`, `.mov`, etc.)**: Video files from which audio tracks can be extracted.
*   **Text Files (`.txt`)**: Plain text documents.
*   **Audio-Text Pairs (directory)**: A directory containing one or more audio files and matching `.txt` transcripts that share the same base filename (e.g. `sample.wav` and `sample.txt`).
*   **Website URLs**: You can provide a list of URLs (static or dynamic) within a text file, or directly pass them to the `process_data_source` function.

### Expected Output (`processed_data/`)

After running the data processing pipeline, the `processed_data/` directory will contain the cleaned and structured data, organized into subdirectories based on the processed data type:

*   `processed_data/processed_text/`: Contains `.txt` files with extracted and cleaned text from PDFs, images, text files, and websites. Each file will correspond to a processed text source.
*   `processed_data/processed_audio/`: Contains `.wav` files of standardized audio segments extracted from raw audio files.
*   `processed_data/processed_audio_from_video/`: Contains `.wav` files of standardized audio segments extracted from video files.
*   `processed_data/audio_text_pair_template.txt`: Describes how processed audio/text pairs are structured.
*   `processed_data/processed_data_metadata.json`: A JSON file that stores metadata for all processed items, including their original source, hash (for deduplication), processed path, data type, and timestamp. This file is crucial for tracking processed data and preventing duplicates. When processing audio-text pair folders, each entry includes a `pair_id` linking an audio file to its transcript.

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
*   `scripts/test_pipeline.py`: An example script to demonstrate the data processing pipeline's functionality.
*   `scripts/train_llm.py`: The main script for training the LLM.
*   `scripts/test_llm.py`: Script for testing the trained LLM by generating text.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.


