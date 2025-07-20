import argparse
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_llm(
    model_path: str,
    prompt: str = "",
    max_new_tokens: int = 50,
    num_return_sequences: int = 1,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    no_repeat_ngram_size: int | None = None,
    num_beams: int | None = None,
):
    """Generate text using a trained model with flexible decoding parameters.

    ``temperature`` and ``top_k``/``top_p`` control the randomness of sampling.
    Higher temperature or larger ``top_k`` can produce more creative text. Set
    ``no_repeat_ngram_size`` (e.g. 2 or 3) to reduce immediate repetition.
    If ``num_beams`` is provided, beam search is used instead of sampling,
    which can improve coherence at the cost of diversity.
    """

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
    gen_args = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    if no_repeat_ngram_size:
        gen_args["no_repeat_ngram_size"] = no_repeat_ngram_size
    if num_beams:
        gen_args["num_beams"] = num_beams
        gen_args["do_sample"] = False
    else:
        gen_args["do_sample"] = True

    generated_ids = model.generate(**gen_args)

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
        default=None,
        help="Prevent repetition of ngrams of this size.",
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
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        num_beams=args.num_beams,
    )


