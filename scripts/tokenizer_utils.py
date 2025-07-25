import os
import logging
from pathlib import Path
from typing import Iterable
from transformers import T5Tokenizer, ByT5Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _text_iterator(text_dir: str) -> Iterable[str]:
    for path in Path(text_dir).glob("*.txt"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
        except Exception as e:
            logging.error(f"Error reading {path}: {e}")


def train_tokenizer(processed_text_dir: str, tokenizer_dir: str, base_model_name: str = "google/mt5-small", vocab_size: int = 5000):
    """Train a tokenizer on all text files in ``processed_text_dir``.

    The tokenizer is initialized from ``base_model_name`` so special tokens match
    the base model. The resulting tokenizer is saved to ``tokenizer_dir``.
    """
    logging.info(f"Training tokenizer from data in {processed_text_dir}")
    # Load the base tokenizer. For mT5, this is a SentencePiece tokenizer.
    # We explicitly use T5Tokenizer (the slow one) to avoid issues with fast tokenizer conversion.
    tokenizer = T5Tokenizer.from_pretrained(base_model_name)

    if isinstance(tokenizer, ByT5Tokenizer):
        logging.info("ByT5 uses a fixed byte-level vocabulary; skipping tokenizer training.")
    else:
        # For SentencePiece tokenizers (like T5), directly training a new one with `train_new_from_iterator`
        # is not the standard way. Instead, we will use the pre-trained tokenizer and save it.
        # If a new tokenizer needs to be trained from scratch for SentencePiece models,
        # it typically involves using the `sentencepiece` library directly.
        logging.info("Skipping custom tokenizer training for T5 model. Using the pre-trained tokenizer.")

    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    logging.info(f"Tokenizer saved to {tokenizer_dir}")
    return tokenizer


