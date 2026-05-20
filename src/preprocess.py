import os
import sys
import pandas as pd

from .config import INDIC_NLP_LIB, INDIC_NLP_RESOURCES, RAW_DATA_PATH, PROCESSED_DATA_PATH

sys.path.append(INDIC_NLP_LIB)
os.environ["INDIC_NLP_RESOURCES"] = INDIC_NLP_RESOURCES

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize

normalizer = IndicNormalizerFactory().get_normalizer("gu")

def normalize_gujarati(text):
    text = normalizer.normalize(text)
    tokens = indic_tokenize.trivial_tokenize(text)
    return " ".join(tokens)

def preprocess_dataset():
    df = pd.read_csv(RAW_DATA_PATH, names=["text", "label"])
    def normalize_label(x):
        if str(x).strip() in ["1", "sarcasm", "true", "yes"]:
            return 1
        if str(x).strip() in ["0", "not_sarcasm", "false", "no"]:
            return 0
        return None

    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["clean_text"] = df["text"].apply(normalize_gujarati)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    return df

