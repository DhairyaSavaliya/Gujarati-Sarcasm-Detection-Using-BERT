from datasets import Dataset
import pandas as pd
from .config import PROCESSED_DATA_PATH

def load_dataset():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["label"] = df["label"].astype(int)
    ds = Dataset.from_pandas(df)
    ds = ds.rename_column("label", "labels")
    return ds
