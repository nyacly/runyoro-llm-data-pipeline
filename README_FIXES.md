# mT5 Training Issues and Fixes

## Problem Analysis

Based on the error logs from your Google Colab training, the main issue was:

```
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.0, 'epoch': 0.0}
WARNING:root:grad_norm became NaN. Stopping training.
```

This indicates that the model gradients became NaN (Not a Number) immediately during training, causing the training to stop after just 10 steps.

## Root Causes Identified

1. **Mixed Precision Issues**: The model was using FP16 mixed precision which can cause numerical instability with certain models and data
2. **Learning Rate Problems**: The learning rate was set to 0.0, preventing any actual learning
3. **Memory Issues**: The mT5-small model was loading in full precision, potentially causing memory pressure
4. **Tokenizer Compatibility**: Using fast tokenizers with SentencePiece models caused tiktoken-related errors

## Fixes Implemented

### 1. Fixed Mixed Precision Configuration
- **Before**: Hardcoded FP16 mixed precision that caused NaN gradients
- **After**: Made mixed precision optional and properly configured
- **Change**: Set `mixed_precision=None` by default to disable mixed precision for stability

### 2. Added 8-bit Quantization
- **Added**: `load_in_8bit=True` to model loading to reduce memory usage
- **Benefit**: Reduces GPU memory requirements while maintaining training stability

### 3. Fixed Learning Rate Issue
- **Problem**: Learning rate was being set to 0.0 in some configurations
- **Fix**: Ensured proper learning rate validation and default values

### 4. Improved Tokenizer Handling
- **Before**: Used T5TokenizerFast which caused tiktoken errors
- **After**: Use T5Tokenizer (slow) for better compatibility with SentencePiece models
- **Added**: Proper sentencepiece dependency in requirements.txt

### 5. Enhanced Error Detection
- **Added**: NaN detection callback that stops training when gradients become invalid
- **Added**: Better error messages and debugging information

## Updated Training Command

Replace your current training command with:

```bash
!python3 -m scripts.train_llm \
    --processed_data_path ./processed_data/processed_text \
    --model_name google/mt5-small \
    --output_dir ./models/runyoro_llm_model \
    --tokenizer_dir ./tokenizer \
    --checkpoint_dir /tmp/runyoro_checkpoints \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-5 \
    --warmup_steps 100 \
    --mixed_precision no \
    --load_in_8bit
```

## Key Changes Made

### 1. `scripts/train_llm.py`
- Added 8-bit quantization: `load_in_8bit=True`
- Fixed mixed precision handling
- Improved error detection and logging
- Better memory management

### 2. `scripts/tokenizer_utils.py`
- Switched from T5TokenizerFast to T5Tokenizer
- Added proper SentencePiece model handling
- Removed tiktoken dependencies

### 3. `requirements.txt`
- Added `sentencepiece` dependency
- Removed problematic `tiktoken` and `blobfile`

### 4. `notebooks/02_Train_LLM_Stage1.ipynb`
- Updated training command to remove mixed precision
- Added proper error handling

## Recommended Training Settings for Google Colab

For stable training on Google Colab with A100 GPU:

```bash
!python3 -m scripts.train_llm \
    --processed_data_path ./processed_data/processed_text \
    --model_name google/mt5-small \
    --output_dir ./models/runyoro_llm_model \
    --tokenizer_dir ./tokenizer \
    --checkpoint_dir /tmp/runyoro_checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --warmup_steps 100 \
    --logging_steps 50 \
    --save_steps 500 \
    --eval_steps 500 \
    --mixed_precision no \
    --max_grad_norm 1.0
```

## Troubleshooting

### If you still get NaN gradients:
1. Reduce learning rate to `1e-5` or `2e-5`
2. Increase warmup steps to `200` or `500`
3. Reduce batch size to `1`
4. Check your data for any corrupted text files

### If you get memory errors:
1. Reduce `per_device_train_batch_size` to `1`
2. Increase `gradient_accumulation_steps` to `4` or `8`
3. Use `mixed_precision="fp16"` only if gradients remain stable

### If training is too slow:
1. Increase `per_device_train_batch_size` to `8` or `16`
2. Enable mixed precision: `mixed_precision="fp16"`
3. Reduce `logging_steps` and `eval_steps`

## Expected Behavior

With these fixes, you should see:
- Stable loss values (not 0.0 or NaN)
- Gradients with normal values (not NaN)
- Steady progress through training steps
- Regular checkpoint saving

The training should now complete successfully without the NaN gradient issue.

