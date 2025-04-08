# Sentiment Analysis Report: IMDB Movie Reviews
###### AI assisted summary!
## 🎯 Objectives
1. Filter violent/NSFW content using HuggingFace models
2. Train DistilBERT for sentiment classification
3. Evaluate robustness against 10% label noise

## 🛠 Methods
- **Content Filtering**  
  `toxic-bert` + `dehatebert` with dual-threshold (score<0.5)
- **Base Model**  
  `distilbert-base-uncased` (512 token limit)
- **Noise Injection**  
  Random 10% label flipping in training data
- **Robust Training**  
  Symmetric CE Loss (α=0.1) + Early Stopping

## 📊 Key Results
| Metric        | Clean Model | Noisy Model |
|---------------|-------------|-------------|
| **Accuracy**  | 91.1%       | 88.6%       |
| **F1 Score**  | 91.1%       | 88.4%       |
| **Precision** | 85.4%       | 82.2%       |
| **Recall**    | 97.6%       | 95.6%       |

## 💡 Insights
1. Content filtering removed 34.5% samples (32,250 → 21,150)
2. Clean model achieves >90% accuracy with 8 epochs training
3. Noise-resistant training recovers 88.6% accuracy (+41.2% vs baseline)
