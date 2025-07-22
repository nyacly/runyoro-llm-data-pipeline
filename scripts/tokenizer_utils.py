import os
import logging
from pathlib import Path
from typing import Iterable
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _text_iterator(text_dir: str) -> Iterable[str]:
    for path in Path(text_dir).glob("*.txt"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                yield f.read()
        except Exception as e:
            logging.error(f"Error reading {path}: {e}")


def train_tokenizer(processed_text_dir: str, tokenizer_dir: str, base_model_name: str = "google/mt5-small", vocab_size: int = 5000):
    """Train a tokenizer on all text files in ``processed_text_dir``.

    The tokenizer is initialized from ``base_model_name`` so special tokens match
    the base model. The resulting tokenizer is saved to ``tokenizer_dir``.
    """
    logging.info(f"Training tokenizer from data in {processed_text_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    iterator = _text_iterator(processed_text_dir)
    tokenizer = tokenizer.train_new_from_iterator(iterator, vocab_size=vocab_size)

    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    logging.info(f"Tokenizer saved to {tokenizer_dir}")
    return tokenizer
