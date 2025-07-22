import argparse
import logging
"""Utility to generate text with a trained seq2seq model.

Historically this script attempted to load the model with
``AutoModelForCausalLM`` which only works with decoder-only architectures.
The Runyoro LLM is based on MT5 (a sequence-to-sequence model), so we need
``AutoModelForSeq2SeqLM`` to correctly load it.  This change prevents errors
such as ``Unrecognized configuration class ... MT5Config`` during loading.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_llm(
    model_path: str,
    prompt: str = "",
    max_new_tokens: int = 100,
    num_return_sequences: int = 3,
    temperature: float = 0.7,
    repetition_penalty: float = 2.0,
    top_k: int = 50,
    top_p: float = 0.95,
    no_repeat_ngram_size: int | None = 2,
    num_beams: int | None = None,
):
    """Generate text using a trained model with flexible decoding parameters.

    ``temperature`` and ``top_k``/``top_p`` control the randomness of sampling.
    Higher temperature or larger ``top_k`` can produce more creative text. Set
    ``no_repeat_ngram_size`` (e.g. 2 or 3) to reduce immediate repetition, and
    ``repetition_penalty`` to further discourage loops.
    If ``num_beams`` is provided, beam search is used instead of sampling,
    which can improve coherence at the cost of diversity.
    """

    logging.info(f"Loading model and tokenizer from {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
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
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
    )
    if num_beams:
        gen_config.num_beams = num_beams
        gen_config.do_sample = False
    else:
        gen_config.do_sample = True

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        generation_config=gen_config,
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
        default=100,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=3,
        help="Number of sequences to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (higher values = more randomness).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Sample from the top k tokens.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling probability threshold.",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=2,
        help="Prevent repetition of ngrams of this size.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=2.0,
        help="Penalty for repetition during generation.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Use beam search with this many beams (disables sampling).",
    )

    args = parser.parse_args()
    test_llm(
        model_path=args.model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        num_beams=args.num_beams,
    )


