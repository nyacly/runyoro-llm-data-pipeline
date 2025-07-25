# Runyoro/Rutooro LLM Data Pipeline

This repository contains a data processing and training pipeline for developing language models for Runyoro/Rutooro, an endangered Bantu language spoken in Uganda.

## Issues Fixed

The following issues have been identified and resolved in this refactored version:

### 1. Tokenizer Issues
- **Problem**: The original code attempted to use fast tokenizers with mT5, which caused tiktoken-related errors
- **Solution**: Modified `tokenizer_utils.py` to use the slow T5Tokenizer and skip custom tokenizer training for SentencePiece models

### 2. Mixed Precision Configuration
- **Problem**: Mixed precision was hardcoded to be disabled, preventing GPU acceleration
- **Solution**: Implemented proper CUDA detection and mixed precision configuration based on hardware availability

### 3. Dependencies
- **Problem**: Missing dependencies and conflicting package versions
- **Solution**: Updated `requirements.txt` with proper dependencies including `sentencepiece` and `accelerate`

### 4. Data Processing
- **Problem**: Text iterator yielded entire files instead of individual lines
- **Solution**: Modified `_text_iterator` to yield individual lines for better tokenization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nyacly/runyoro-llm-data-pipeline.git
cd runyoro-llm-data-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Configure accelerate for distributed training:
```bash
accelerate config
```

## Usage

### Data Processing
Place your text data in the `processed_data/processed_text/` directory as `.txt` files.

### Training
Run the training script:
```bash
python3 -m scripts.train_llm \
    --processed_data_path processed_data/processed_text \
    --model_name google/mt5-small \
    --output_dir ./models/runyoro_llm_model \
    --tokenizer_dir ./tokenizer \
    --checkpoint_dir /tmp/runyoro_checkpoints \
    --mixed_precision no \
    --load_in_8bit \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4
```

### Jupyter Notebooks
The training can also be run using the provided Jupyter notebooks:
- `01_Process_Data.ipynb` - Data preprocessing
- `02_Train_LLM_Stage1.ipynb` - Model training
- `03_Test_LLM.ipynb` - Model testing

## Key Features

- **Robust Error Handling**: NaN detection and automatic training termination
- **Checkpoint Management**: Automatic checkpoint saving and resumption
- **Mixed Precision Training**: Automatic FP16/BF16 support based on hardware
- **Data Validation**: Automatic filtering of empty or invalid examples
- **Flexible Configuration**: Extensive command-line arguments for customization

## Configuration Options

Key training parameters:
- `--num_train_epochs`: Number of training epochs (default: 5)
- `--per_device_train_batch_size`: Batch size per device (default: 8)
  - `--learning_rate`: Learning rate (default: 1e-5)
  - `--mixed_precision`: Mixed precision mode (no/fp16/bf16)
  - `--load_in_8bit`: Enable 8-bit model loading to reduce GPU memory usage
  - `--gradient_accumulation_steps`: Gradient accumulation steps
  - `--max_grad_norm`: Gradient clipping threshold

## Hardware Requirements

- **Minimum**: 8GB RAM, CPU training supported
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM for mixed precision training
- **Optimal**: Multi-GPU setup with 16GB+ VRAM per GPU

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Use `fp16` mixed precision

2. **Tokenizer Errors**:
   - Ensure `sentencepiece` is installed
   - Use the slow tokenizer (T5Tokenizer) instead of fast tokenizer

3. **Training Instability**:
   - Check for NaN values in data
   - Reduce learning rate
   - Adjust `max_grad_norm` for gradient clipping

-### Performance Optimization

- Use mixed precision training (`--mixed_precision fp16`) when training is
  stable and a compatible GPU is available
- Enable gradient checkpointing for memory efficiency
- Use multiple GPUs with `accelerate launch`
- Optimize batch size and gradient accumulation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Runyoro/Rutooro language community
- Hugging Face Transformers library
- Google's mT5 model

