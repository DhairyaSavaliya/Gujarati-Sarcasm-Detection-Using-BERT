from transformers import BertTokenizer, BertForSequenceClassification
from .config import MODEL_NAME, MODEL_SAVE_DIR

def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    return tokenizer, model
