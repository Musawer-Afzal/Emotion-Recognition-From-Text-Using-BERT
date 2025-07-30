import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same model architecture used during training
class EmotionClassifier(nn.Module):
    def __init__(self, num_labels=7):
        super(EmotionClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)  # ✅ change 'classifier' → 'out'

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        return self.out(dropout_output)  # ✅ same here


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # folder path

# Load model
model = EmotionClassifier(num_labels=7)  # change if you have different number of labels
model.load_state_dict(torch.load(
    "D:\\Coding\\Internship\\Developers Hub Internship\\PDF 2\\Completed Projects\\Emotion Recognition Ai\\Trained model\\emotion_model_state.pt",
    map_location=device
))
model.to(device)
model.eval()

# List of emotion labels (based on GoEmotions)
EMOTION_LABELS = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion']

# Prediction function
def predict_emotions(text):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    # Get emotions with probability > 0.5 (threshold can be adjusted)
    result = {label: float(prob) for label, prob in zip(EMOTION_LABELS, probs) if prob > 0.5}

    return result if result else {"neutral": 1.0}

# Streamlit UI
st.title("🧠 Emotion Recognition from Text")
st.markdown("Enter a sentence and get its associated emotions.")

user_input = st.text_area("Enter text:", height=150)

if st.button("Analyze Emotion"):
    if user_input.strip():
        emotions = predict_emotions(user_input)
        st.subheader("Detected Emotions:")
        for emotion, score in emotions.items():
            st.write(f"**{emotion.capitalize()}**: {score:.2f}")
    else:
        st.warning("Please enter some text to analyze.")
