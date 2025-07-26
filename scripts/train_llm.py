import os
import argparse
import logging
import shutil
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from scripts.tokenizer_utils import train_tokenizer
import math


class NanDetectionCallback(TrainerCallback):
    """Detect NaN values in training metrics and stop training.

    If a NaN is encountered, the callback optionally checks the training
    dataset for NaN values to help locate problematic examples.
    """

    def __init__(self, train_dataset=None, tokenizer=None):
        super().__init__()
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer

    def _check_dataset_for_nan(self):
        if self.train_dataset is None:
            return None

        import numpy as np

        for idx, example in enumerate(self.train_dataset):
            for col in ["input_ids", "labels"]:
                if col not in example:
                    continue
                arr = np.asarray(example[col], dtype=float)
                if np.isnan(arr).any():
                    return idx, col
        return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        grad_norm = logs.get("grad_norm")
        loss = logs.get("loss")

        detected = False
        if (
            grad_norm is not None
            and isinstance(grad_norm, float)
            and math.isnan(grad_norm)
        ):
            logging.warning("grad_norm became NaN. Stopping training.")
            detected = True
        if loss is not None and isinstance(loss, float) and math.isnan(loss):
            logging.warning("Loss became NaN. Stopping training.")
            detected = True

        if detected:
            nan_loc = self._check_dataset_for_nan()
            if nan_loc is not None:
                idx, col = nan_loc
                logging.warning(
                    f"NaN detected in training data at index {idx} column '{col}'."
                )
                if self.tokenizer is not None:
                    problematic_example = self.train_dataset[idx]
                    decoded_text = self.tokenizer.decode(
                        problematic_example["input_ids"]
                    )
                    with open("problematic_examples.txt", "a") as f:
                        f.write(f"Index: {idx}, Column: {col}\\n")
                        f.write(f"Text: {decoded_text}\\n\\n")
            control.should_training_stop = True


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_llm(
    processed_data_path: str,
    model_name: str,
    output_dir: str,
    tokenizer_dir: str,
    vocab_size: int = 5000,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    save_steps: int = 5000,
    save_total_limit: int = 2,
    logging_steps: int = 10,
    learning_rate: float = 5e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    eval_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    block_size: int = 512,
    cache_dir: str | None = None,
    checkpoint_dir: str | None = None,
    cleanup_checkpoints: bool = False,
    mixed_precision: str | None = "no",
    load_in_8bit: bool = False,
    use_wandb: bool = False,
):
    """Train or resume a sequence-to-sequence language model.

    Increasing ``num_train_epochs`` allows longer training runs. Pass
    ``--num_train_epochs 5`` (or higher) on the command line to train for
    additional epochs. When ``use_wandb`` is enabled, training and evaluation
    losses are logged to Weights & Biases. Monitor ``eval_loss`` there and stop
    training if it begins to rise while ``train_loss`` continues to fall, which
    can indicate overfitting.

    To incorporate additional Runyoro/Rutooro data, simply place new ``.txt``
    files in ``processed_data_path`` before running this script again – the
    loader automatically reads all files in that directory.

    Experiment with ``learning_rate`` and ``gradient_accumulation_steps`` if the
    model is still not converging well after more epochs or data.
    Mixed precision can be enabled with ``mixed_precision='fp16'`` when running
    on a compatible GPU, but is disabled by default for stability.

    ``warmup_steps`` now defaults to 500 so the scheduler quickly reaches the base
    learning rate when training on small datasets.

    ``max_grad_norm`` can be adjusted to clip exploding gradients if training
    becomes unstable.
    """
    logging.info(f"Loading dataset from {processed_data_path}")

    processed_data_full_path = os.path.join(os.getcwd(), processed_data_path)
    logging.info(
        f"Attempting to load text files from: {processed_data_full_path}/*.txt"
    )

    if not os.path.isdir(processed_data_full_path):
        raise FileNotFoundError(
            f"Processed data directory not found: {processed_data_full_path}"
        )

    data_files = {"train": os.path.join(processed_data_full_path, "*.txt")}
    raw = load_dataset(
        "text",
        data_files=data_files,
        cache_dir=cache_dir or "/tmp/hf_datasets_cache",
    )
    # filter empty or very short lines
    raw = raw.filter(lambda ex: bool(ex["text"].strip()))
    raw = raw.filter(lambda ex: len(ex["text"]) > 10)

    logging.info("Training/Updating tokenizer...")
    tokenizer = train_tokenizer(
        processed_data_path, tokenizer_dir, model_name, vocab_size
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def tokenize_fn(ex):
        tokens = tokenizer(
            ex["text"],
            truncation=True,
            max_length=block_size,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    logging.info("Tokenizing dataset...")
    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        num_proc=4,
    )

    tokenized = tokenized["train"].train_test_split(test_size=0.1)
    logging.info(f"Dataset size: {len(tokenized['train'])} examples")

    if len(tokenized["train"]) == 0:
        raise ValueError(
            f"No training examples were found in {processed_data_full_path}."
            " Ensure the directory contains non-empty .txt files before running"
            " training."
        )

    if len(tokenized["train"]) < 10:
        raise ValueError(
            "Training dataset must contain at least 10 examples to avoid "
            "unstable gradients. Add more data to processed_data/processed_text."
        )

    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0.")

    tokenized_datasets = tokenized

    # Optional sanity check for NaNs in the tokenized dataset
    import numpy as np

    for split_name in ["train", "test"]:
        ds = tokenized_datasets[split_name]
        for col in ["input_ids", "labels"]:
            # Flatten the list of token sequences so ``np.isnan`` can operate
            # on a one-dimensional NumPy array. ``ds[col]`` is a list of
            # variable-length lists which cannot be directly converted to a
            # NumPy array without specifying ``dtype=object``. Flattening
            # avoids shape issues and still allows detection of invalid values.
            flat_tokens = [t for seq in ds[col] for t in seq]
            arr = np.asarray(flat_tokens, dtype=float)
            if np.isnan(arr).any():
                raise ValueError(f"NaN detected in {split_name} dataset column {col}")

    # Datasets were already split earlier
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    import torch

    logging.info(f"Loading model: {model_name}")
    quant_args = {}
    if load_in_8bit and torch.cuda.is_available():
        quant_args = {"load_in_8bit": True, "device_map": "auto"}
        logging.info("Using 8-bit quantization for model loading.")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, ignore_mismatched_sizes=True, **quant_args
    )
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['q', 'v'],
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_cfg)
    if tokenizer.pad_token is not None and model.config.vocab_size < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    # HF Trainer setup using a causal language modeling collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a sequence-to-sequence LLM for Runyoro/Rutooro."
    )
    parser.add_argument(
        "--processed_data_path",
        type=str,
        required=True,
        help=(
            "Path to the directory containing processed text data (e.g., processed_data/processed_text). "
            "Each .txt file in this directory will be treated as a document."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/mt5-small",
        help="Pre-trained model name from Hugging Face Transformers (e.g., gpt2, distilgpt2, facebook/opt-125m).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/runyoro_llm_model",
        help="Directory to save the trained model and tokenizer.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="./tokenizer",
        help="Directory where the tokenizer will be saved.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory for intermediate training checkpoints (defaults to output_dir)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=5000,
        help="Vocabulary size for tokenizer training.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per device during training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size per device during evaluation.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Number of updates steps before saving checkpoint.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Total number of checkpoints to store.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Number of update steps between two logs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for the learning rate scheduler.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Number of update steps between two evaluations.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm (for gradient clipping).",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="Maximum token length for model inputs.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching downloaded models and datasets.",
    )
    parser.add_argument(
        "--cleanup_checkpoints",
        action="store_true",
        help="Whether to clean up intermediate checkpoints after training.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision training (no, fp16, bf16).",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model using 8-bit quantization if CUDA is available.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use Weights & Biases for logging.",
    )

    args = parser.parse_args()

    train_llm(
        processed_data_path=args.processed_data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        tokenizer_dir=args.tokenizer_dir,
        vocab_size=args.vocab_size,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        block_size=args.block_size,
        cache_dir=args.cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        cleanup_checkpoints=args.cleanup_checkpoints,
        mixed_precision=args.mixed_precision,
        load_in_8bit=args.load_in_8bit,
        use_wandb=args.use_wandb,
    )
