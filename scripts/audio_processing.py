import os
import logging
from pydub import AudioSegment
from pydub.silence import split_on_silence

from scripts.core_components import process_audio_file_standardize

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def reduce_noise_audio_segment(audio_segment):
    # Placeholder for noise reduction. More advanced techniques would go here.
    # For a real implementation, consider libraries like `noisereduce` or pre-trained models.
    return audio_segment


def normalize_volume_audio_segment(audio_segment):
    # Normalize loudness to -20 dBFS (a common target for speech)
    return audio_segment.normalize(-20.0)


def process_audio_source(
    audio_path, output_dir, min_silence_len=1000, silence_thresh=-40, keep_silence=200
):
    standardized_audio_path = os.path.join(
        output_dir, f"standardized_{os.path.basename(audio_path)}.wav"
    )
    standardized_audio_path = process_audio_file_standardize(
        audio_path, standardized_audio_path
    )

    if not standardized_audio_path:
        return []

    try:
        audio = AudioSegment.from_file(standardized_audio_path)
        audio = reduce_noise_audio_segment(audio)
        audio = normalize_volume_audio_segment(audio)

        segments = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence,
        )

        processed_segments_info = []
        for i, segment in enumerate(segments):
            segment_output_path = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(audio_path))[0]}_segment_{i}.wav",
            )
            segment.export(segment_output_path, format="wav")
            processed_segments_info.append(
                {
                    "path": segment_output_path,
                    "original_path": audio_path,
                    "segment_index": i,
                }
            )
        logging.info(
            f"Successfully processed and segmented audio from {audio_path}. Found {len(segments)} segments."
        )
        return {
            "standardized_path": standardized_audio_path,
            "segments": processed_segments_info,
        }
    except Exception as e:
        logging.error(f"Error processing audio source {audio_path}: {e}")
        return []
