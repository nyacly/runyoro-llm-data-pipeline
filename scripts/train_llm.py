import os
import argparse
import logging
import shutil
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from scripts.tokenizer_utils import train_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_llm(
    processed_data_path: str,
    model_name: str,
    output_dir: str,
    tokenizer_dir: str,
    vocab_size: int = 5000,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    save_steps: int = 500,
    save_total_limit: int = 2,
    logging_steps: int = 50,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    eval_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    cache_dir: str | None = None,
    checkpoint_dir: str | None = None,
    cleanup_checkpoints: bool = False,
    mixed_precision: str | None = None,
):
    logging.info(f"Loading dataset from {processed_data_path}")

    processed_data_full_path = os.path.join(os.getcwd(), processed_data_path)
    logging.info(f"Attempting to load text files from: {processed_data_full_path}")

    if not os.path.isdir(processed_data_full_path):
        raise FileNotFoundError(
            f"Processed data directory not found: {processed_data_full_path}"
        )

    all_texts = []
    text_files_found = 0
    for filename in os.listdir(processed_data_full_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(processed_data_full_path, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    all_texts.extend(lines)
                text_files_found += 1
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")

    if text_files_found == 0:
        raise ValueError(
            f"No .txt files found in the specified processed data directory: {processed_data_full_path}"
        )
    else:
        logging.info(f"Successfully loaded text from {text_files_found} files.")

    dataset = Dataset.from_dict({"text": all_texts})
    logging.info(f"Dataset created with {len(dataset)} examples.")

    logging.info("Training/Updating tokenizer...")
    tokenizer = train_tokenizer(processed_data_path, tokenizer_dir, model_name, vocab_size)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    logging.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, num_proc=os.cpu_count(), remove_columns=["text"]
    )

    # Split dataset into train and validation
    split_tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)
    train_dataset = split_tokenized_datasets["train"]
    eval_dataset = split_tokenized_datasets["test"]

    logging.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is not None and model.config.vocab_size < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_output_dir = checkpoint_dir if checkpoint_dir else output_dir

    fp16 = False
    bf16 = False
    if mixed_precision == "fp16":
        fp16 = True
    elif mixed_precision == "bf16":
        bf16 = True

    training_args = TrainingArguments(
        output_dir=training_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=eval_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        bf16=bf16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


    # --- Automated checkpoint resumption ---
    latest_checkpoint = None
    if os.path.isdir(training_output_dir):
        checkpoints = [d for d in os.listdir(training_output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort by checkpoint number
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
            latest_checkpoint = os.path.join(training_output_dir, checkpoints[-1])
            logging.info(f"Resuming from latest checkpoint: {latest_checkpoint}")

    logging.info("Starting training...")
    if latest_checkpoint:
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
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
    parser = argparse.ArgumentParser(description="Train a causal LLM for Runyoro/Rutooro.")
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
        default="gpt2", # Start with a small, accessible model like GPT-2
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
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
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
        default=2e-5,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
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
        default=None,
        help=(
            "Enable mixed precision training (fp16 or bf16). If not set, training"
            " runs in full precision."
        ),
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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cache_dir=args.cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        save_total_limit=args.save_total_limit,
        cleanup_checkpoints=args.cleanup_checkpoints,
        mixed_precision=args.mixed_precision,
    )


