import os
import argparse
import logging
import shutil
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from scripts.tokenizer_utils import train_tokenizer
import math


class NanDetectionCallback(TrainerCallback):
    """Detect NaN values in training metrics and stop training.

    If a NaN is encountered, the callback optionally checks the training
    dataset for NaN values to help locate problematic examples.
    """

    def __init__(self, train_dataset=None):
        super().__init__()
        self.train_dataset = train_dataset

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
        if grad_norm is not None and isinstance(grad_norm, float) and math.isnan(grad_norm):
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
    save_steps: int = 5000,
    save_total_limit: int = 2,
    logging_steps: int = 10,
    learning_rate: float = 1e-5,
    warmup_steps: int = 5,
    weight_decay: float = 0.01,
    eval_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    cache_dir: str | None = None,
    checkpoint_dir: str | None = None,
    cleanup_checkpoints: bool = False,
    mixed_precision: str | None = "fp16",
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
    Mixed precision (``fp16``) is enabled by default for faster training on GPUs.

    ``warmup_steps`` now defaults to 5 so the scheduler quickly reaches the base
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

    try:
        dataset = load_dataset(
            "text",
            data_files=f"{processed_data_full_path}/*.txt",
            cache_dir=cache_dir or "/tmp/hf_datasets_cache",
            keep_in_memory=True,
        )["train"]
    except NotImplementedError as e:
        if "LocalFileSystem" in str(e):
            logging.warning(
                "In-memory loading not supported on this 'datasets' version; falling back to manual loading."
            )
            # Manually read the text files and create the Dataset in-memory.
            # Each non-empty line is treated as a separate training example so
            # we preserve the same semantics as ``load_dataset("text")``.
            import glob

            text_files = sorted(glob.glob(f"{processed_data_full_path}/*.txt"))
            texts = []
            for path in text_files:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            texts.append(line)

            dataset = Dataset.from_dict({"text": texts})
        else:
            raise
    logging.info(f"Dataset size: {len(dataset)} examples")

    if len(dataset) == 0:
        raise ValueError(
            f"No training examples were found in {processed_data_full_path}."
            " Ensure the directory contains non-empty .txt files before running"
            " training."
        )

    # Ensure we have a validation split
    dataset = dataset.train_test_split(test_size=0.1)

    logging.info("Training/Updating tokenizer...")
    tokenizer = train_tokenizer(
        processed_data_path, tokenizer_dir, model_name, vocab_size
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], truncation=True, max_length=512)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    logging.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=["text"],
    )

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
                raise ValueError(
                    f"NaN detected in {split_name} dataset column {col}"
                )

    # Datasets were already split earlier
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    logging.info(f"Loading model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, ignore_mismatched_sizes=True
    )
    if tokenizer.pad_token is not None and model.config.vocab_size < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    logging.info("Using DataCollatorForSeq2Seq for training.")

    training_output_dir = checkpoint_dir if checkpoint_dir else output_dir

    fp16 = False
    bf16 = False
    if mixed_precision == "fp16":
        fp16 = True
    elif mixed_precision == "bf16":
        bf16 = True

    # Build arguments for TrainingArguments but ensure compatibility with older
    # Transformers versions that may not support some parameters like
    # ``evaluation_strategy``.
    import inspect

    training_args_kwargs = {
        "output_dir": training_output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "logging_steps": logging_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "evaluation_strategy": "steps",
        "eval_steps": eval_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_grad_norm": max_grad_norm,
        "fp16": fp16,
        "bf16": bf16,
        "run_name": "runyoro_llm_v2",
        "report_to": ["wandb"] if use_wandb else None,
    }

    init_params = inspect.signature(TrainingArguments.__init__).parameters
    filtered_kwargs = {
        k: v for k, v in training_args_kwargs.items() if k in init_params
    }

    training_args = TrainingArguments(**filtered_kwargs)
    logging.info(f"Actual warmup_steps used by Trainer: {training_args.warmup_steps}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.add_callback(NanDetectionCallback(train_dataset=train_dataset))

    # --- Re-enabled Automated checkpoint resumption with robustness check ---
    latest_checkpoint = None
    if os.path.isdir(training_output_dir):
        checkpoints = [
            d for d in os.listdir(training_output_dir) if d.startswith("checkpoint-")
        ]
        if checkpoints:
            # Sort by checkpoint number to find the true latest
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            candidate_checkpoint = os.path.join(training_output_dir, checkpoints[-1])

            # --- ROBUSTNESS CHECK: Verify if the candidate checkpoint is actually valid ---
            # A valid checkpoint should at least contain the model weights
            # It could be 'model.safetensors' or 'pytorch_model.bin' depending on save format
            model_safetensors_exists = os.path.exists(
                os.path.join(candidate_checkpoint, "model.safetensors")
            )
            pytorch_model_exists = os.path.exists(
                os.path.join(candidate_checkpoint, "pytorch_model.bin")
            )

            if model_safetensors_exists or pytorch_model_exists:
                latest_checkpoint = candidate_checkpoint
                logging.info(
                    f"Found and validated latest checkpoint: {latest_checkpoint}"
                )
            else:
                logging.warning(
                    f"Found checkpoint folder '{candidate_checkpoint}' but it appears incomplete (missing model files). Starting fresh."
                )
                # If incomplete, treat as if no valid checkpoint was found
                latest_checkpoint = None
        else:
            logging.info(
                "No checkpoint folders found in output directory. Starting fresh."
            )
    else:
        logging.info(
            "Output directory does not exist. Starting fresh."
        )  # This might not happen often if you create it earlier

    # Proceed with training or resumption
    if latest_checkpoint:
        logging.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        try:
            trainer.train(resume_from_checkpoint=latest_checkpoint)
        except RuntimeError as e:
            logging.error(
                f"Failed to load checkpoint due to incompatible model parameters: {e}"
            )
            logging.info("Starting training from scratch instead.")
            trainer.train()
    else:
        logging.info("Starting training fresh (no valid checkpoint found).")
        trainer.train()

    logging.info(f"Saving final model to {training_output_dir}/final_model")
    trainer.save_model(f"{training_output_dir}/final_model")
    tokenizer.save_pretrained(f"{training_output_dir}/final_model")

    if training_output_dir != output_dir:
        final_dest = os.path.join(output_dir, "final_model")
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Copying final model to {final_dest}")
        shutil.copytree(
            os.path.join(training_output_dir, "final_model"),
            final_dest,
            dirs_exist_ok=True,
        )
        if cleanup_checkpoints:
            logging.info(f"Cleaning up checkpoint directory {training_output_dir}")
            shutil.rmtree(training_output_dir, ignore_errors=True)

    logging.info("Training complete.")


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
        "--save_total_limit",
        type=int,
        default=2,
        help="Total number of checkpoints to keep during training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Warmup steps for the scheduler.",
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
        help="Maximum gradient norm for clipping to avoid exploding gradients.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional directory for Hugging Face dataset cache.",
    )
    parser.add_argument(
        "--cleanup_checkpoints",
        action="store_true",
        help="Remove checkpoint_dir after copying the final model to output_dir.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["fp16", "bf16"],
        default="fp16",
        help=(
            "Enable mixed precision training (fp16 or bf16). If not set, training"
            " runs in full precision."
        ),
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log metrics to Weights & Biases during training.",
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
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        cache_dir=args.cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        save_total_limit=args.save_total_limit,
        cleanup_checkpoints=args.cleanup_checkpoints,
        mixed_precision=args.mixed_precision,
        use_wandb=args.use_wandb,
    )
