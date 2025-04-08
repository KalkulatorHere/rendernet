

# Sentiment Analysis on IMDB Reviews with Label Noise Robustness

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/your-username/your-notebook-name)

A transformer-based approach for sentiment analysis with experiments in handling label noise.

## Project Overview

### Key Components
1. **Content Filtering**: Uses `unitary/toxic-bert` and `Hate-speech-CNERG/dehatebert` models to remove violent/NSFW content
2. **Base Model**: DistilBERT-base-uncased for efficient training
3. **Noise Injection**: 10% label flipping experiment
4. **Robust Training**: Implements Symmetric Cross Entropy Loss for noisy labels
5. **Early Stopping**: Custom callback with patience=3

## Why Did We Fail? 🔍

The critical error occurred during model saving:
```bash
SafetensorError: No space left on device


**Root Causes**:
1. **Checkpoint Overload**: Default HuggingFace settings save all intermediate checkpoints
2. **Kaggle Storage Limits**: Free tier has ~20GB temporary storage
3. **Dataset Expansion**: Content filtering preserved most data (78% original size)
4. **Memory Leaks**: Potential CUDA memory fragmentation during multi-model filtering

## Prevention Strategies 🛡️

### 1. Storage Management
```python
# Limit checkpoints & enable FP16
TrainingArguments(
    save_total_limit=1,  # Keep only best model
    fp16=True,           # Reduce model size
)
```

### 2. Tempfile Usage
```python
import tempfile
PROCESSED_DATA_PATH = tempfile.mkdtemp()  # Use ephemeral storage
```

### 3. Model Cleanup
```python
import shutil
shutil.rmtree("./clean_model")  # Remove previous checkpoints
```

### 4. Resource Monitoring
```bash
# Add during training
!df -h  # Monitor disk space
!nvidia-smi  # GPU memory
```

### 5. Alternative Architectures
- Use smaller models (TinyBERT, MobileBERT)
- Implement gradient checkpointing
- Use quantization (`bitsandbytes`)

## Results Comparison 📊

| Metric       | Clean Model | Noisy Model |
|--------------|-------------|-------------|
| Accuracy     | 93.6%       | 89.7%       |
| F1-Score     | 92.9%       | 88.2%       |
| Precision    | 89.2%       | 82.3%       |
| Recall       | 97.1%       | 94.1%       |

**Noise Impact**: 3.9% absolute accuracy drop demonstrates label noise vulnerability

## Key Lessons Learned 🎓

1. **Kaggle Constraints Matter**: Storage limits require aggressive cleanup
2. **Early Stopping ≠ Safety**: Validation metrics can mask storage issues
3. **Batch Processing Risks**: Parallel filtering creates memory spikes
4. **Noise Needs Special Handling**: Standard CE loss fails with flipped labels

## Alternative Approaches 💡

1. **Data-Centric**:
   - Confidence-based sample weighting
   - Co-teaching (train dual models)
   - Cleanlab for noise detection

2. **Model-Centric**:
   - Label smoothing
   - Mixup augmentation
   - BERT-ASC (noise-robust variant)

3. **Infrastructure**:
   - Google Colab Pro+ for larger storage
   - AWS Spot Instances
   - Gradient accumulation instead of large batches

## How to Run

```bash
# 1. Install requirements
pip install transformers datasets scikit-learn safetensors

# 2. Reduce memory usage
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 3. Run with cleanup
python main.py --freeze_layers 4 --max_length 256
```

## References
- Original Dataset: [IMDB Dataset of 50K Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Toxic Content Model: [`unitary/toxic-bert`](https://huggingface.co/unitary/toxic-bert)
- Hate Speech Model: [`dehatebert`](https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english)

---

**Note**: Replace Kaggle notebook links with your actual notebook URL before deployment
```

