import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDIC_NLP_LIB = os.path.join(BASE_DIR, "indic_resources", "indic_nlp_library")
INDIC_NLP_RESOURCES = os.path.join(BASE_DIR, "indic_resources", "indic_nlp_resources")

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "sarcasm_data.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

MODEL_NAME = "google/muril-base-cased"
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "muril_sarcasm")
