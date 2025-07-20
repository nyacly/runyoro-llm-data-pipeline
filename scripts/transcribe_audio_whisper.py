import argparse
import os
import torch
from faster_whisper import WhisperModel


def transcribe_audio(input_path: str, output_path: str, model_size: str = "base") -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(input_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(segment.text.strip() + "\n")
    print(f"Transcription written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using Faster-Whisper")
    parser.add_argument("--input_path", required=True, help="Path to audio or video file")
    parser.add_argument("--output_path", default="transcription.txt", help="Where to save the transcript")
    parser.add_argument("--model_size", default="base", help="Whisper model size to use")
    args = parser.parse_args()
    transcribe_audio(args.input_path, args.output_path, args.model_size)
