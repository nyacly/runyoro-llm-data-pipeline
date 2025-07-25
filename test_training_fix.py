#!/usr/bin/env python3
"""
Test script to verify the training fix works with a small dataset.
"""

import os
import tempfile
import shutil
from scripts.train_llm import train_llm

def create_test_data():
    """Create a small test dataset."""
    test_dir = tempfile.mkdtemp()
    data_dir = os.path.join(test_dir, "processed_text")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create some sample text files (at least 10 examples)
    sample_texts = [
        "Runyoro ni rurimi rw'abantu b'omu Bunyoro.",
        "Abantu b'omu Bunyoro ni abairu.",
        "Omu Bunyoro hari ebintu bingi ebirungi.",
        "Runyoro rurimi rw'abantu b'omu kitongole kya Bunyoro.",
        "Abantu b'omu Bunyoro barikukora emirimu mingi.",
        "Ebyokurya by'omu Bunyoro birungi muno.",
        "Ente z'omu Bunyoro nungi muno.",
        "Omwaka gunu tugyenda kusoma Runyoro.",
        "Runyoro ni rurimi rwa Uganda.",
        "Ninyenda kusoma Runyoro buri eizooba.",
        "Ebitabo bya Runyoro biri bingi.",
        "Ninyenda kwongyera kumanya Runyoro.",
    ]
    
    for i, text in enumerate(sample_texts):
        with open(os.path.join(data_dir, f"sample_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
    
    return test_dir, data_dir

def test_training():
    """Test the training pipeline with minimal settings."""
    test_dir, data_dir = create_test_data()
    
    try:
        # Test with minimal settings
        train_llm(
            processed_data_path=data_dir,
            model_name="google/mt5-small",
            output_dir=os.path.join(test_dir, "output"),
            tokenizer_dir=os.path.join(test_dir, "tokenizer"),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            learning_rate=1e-4,
            warmup_steps=1,
            logging_steps=1,
            save_steps=10,
            eval_steps=10,
            mixed_precision=None,  # Disable mixed precision for testing
            use_wandb=False,
        )
        print("✅ Training test completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False
    finally:
        # Clean up
        shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    success = test_training()
    exit(0 if success else 1)


