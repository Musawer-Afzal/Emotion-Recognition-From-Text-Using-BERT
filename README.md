# 😃 Multi-Label Emotion Recognition from Text

This NLP project uses the GoEmotions dataset to recognize multiple emotions in a single sentence. It's built using transformer models like BERT to classify 28 fine-grained emotion labels.

## 🧠 Project Goals

- Preprocess and tokenize emotion-labeled text
- Fine-tune a BERT-based model
- Handle class imbalance and multi-label classification
- Evaluate using Hamming loss, F1 Score, and accuracy

## 📦 Dataset

- [GoEmotions Dataset (Google)](https://huggingface.co/datasets/go_emotions)

## 🛠️ Technologies

- Python (Pandas, NumPy)
- Hugging Face Transformers
- PyTorch
- Scikit-learn
- Matplotlib / Seaborn

## 📊 Features

- Multi-label classification (multiple emotions per input)
- Handles class imbalance using weighted loss
- Fine-tuned transformer model

## 🧪 Evaluation Metrics

- Accuracy
- Hamming Loss
- Macro/Micro F1 Score

## 📁 Files

- `emotion_model.py` – Model training script
- `predict_emotions.py` – Emotion prediction function
- `README.md` – Project overview

## 🔍 Sample Output

```json
Text: "I can't believe how amazing this is!"
Predicted Emotions: ['admiration', 'joy', 'excitement']
