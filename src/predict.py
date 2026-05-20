import torch
from transformers import BertTokenizer, BertForSequenceClassification
from .preprocess import normalize_gujarati
from .config import MODEL_SAVE_DIR

def predict(text):
    tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_DIR)

    text = normalize_gujarati(text)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)

    pred = torch.argmax(outputs.logits).item()

    return "Sarcastic 😏" if pred == 1 else "Not Sarcastic 🙂"
