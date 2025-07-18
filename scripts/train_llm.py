import os
import argparse
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_llm(
    processed_data_path: str,
    model_name: str,
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    save_steps: int = 500,
    save_total_limit: int = 2,
    logging_steps: int = 50,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    eval_steps: int = 500,
    gradient_accumulation_steps: int = 1,
):
    logging.info(f"Loading dataset from {processed_data_path}")
    # Assuming processed_data_path points to a directory containing text files
    # Each text file is considered a document
    dataset = load_dataset("text", data_files=f"{processed_data_path}/*.txt")

    logging.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    logging.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, num_proc=os.cpu_count(), remove_columns=["text"]
    )

    # Split dataset into train and validation
    train_dataset = tokenized_datasets["train"].train_test_split(test_size=0.1)
    eval_dataset = train_dataset["test"]
    train_dataset = train_dataset["train"]

    logging.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is not None and model.config.vocab_size < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
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
        fp16=False, # Disable mixed precision for Apple Silicon/Mac
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
    if os.path.isdir(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort by checkpoint number
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
            latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
            logging.info(f"Resuming from latest checkpoint: {latest_checkpoint}")

    logging.info("Starting training...")
    if latest_checkpoint:
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()

    logging.info(f"Saving final model to {output_dir}/final_model")
    trainer.save_model(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")

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

    args = parser.parse_args()
    train_llm(
        processed_data_path=args.processed_data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


