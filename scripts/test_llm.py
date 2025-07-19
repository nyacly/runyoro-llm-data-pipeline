import argparse
import logging
import os

import pytest

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ModuleNotFoundError:  # pragma: no cover - handled via skip
    pytest.skip("transformers library not installed", allow_module_level=True)

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_llm(
    model_path: str = "./models/runyoro_llm_model/final_model",
    prompt: str = "",
    max_new_tokens: int = 50,
    num_return_sequences: int = 1,
):
    if not os.path.isdir(model_path):
        pytest.skip(f"Model path {model_path} not available")

    logging.info(f"Loading model and tokenizer from {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    except Exception as e:
        logging.error(f"Error loading model or tokenizer from {model_path}: {e}")
        logging.info("Please ensure the model path is correct and contains valid Hugging Face model files.")
        return

    model.eval() # Set model to evaluation mode

    if torch.cuda.is_available():
        model.to("cuda")
        logging.info("Model moved to GPU.")
    else:
        logging.info("CUDA not available, running on CPU. Generation might be slow.")

    if not prompt:
        logging.info("No prompt provided. Generating a default sequence.")
        input_ids = tokenizer.encode("", return_tensors="pt")
    else:
        logging.info(f"Generating text with prompt: \"{prompt}\"")
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")

    logging.info("Starting text generation...")
    generated_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,  # Enable sampling for more diverse outputs
        top_k=50,  # Consider top 50 tokens
        top_p=0.95,  # Nucleus sampling
        temperature=0.7,  # Control randomness
    )

    logging.info("Generated text:")
    for i, generated_id in enumerate(generated_ids):
        generated_text = tokenizer.decode(generated_id, skip_special_tokens=True)
        logging.info(f"--- Generated Sequence {i+1} ---")
        logging.info(generated_text)

    logging.info("Testing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained LLM for Runyoro/Rutooro.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/runyoro_llm_model/final_model",
        help="Path to the directory containing the trained model and tokenizer (e.g., ./models/runyoro_llm_model/final_model).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Optional: Initial text prompt for generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate.",
    )

    args = parser.parse_args()
    test_llm(
        model_path=args.model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
    )


