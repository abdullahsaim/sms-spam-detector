"""
preprocess.py — Text cleaning and feature engineering for SMS Spam Detection.
Author: abdullahsaim
"""

import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LEN = 100
TOKENIZER_PATH = "artifacts/tokenizer.pkl"


def clean_text(text: str) -> str:
    """
    Lowercase, remove punctuation/digits, strip extra whitespace.
    Keeps only alphabetic tokens — digits and symbols add noise for spam signals.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)       # remove non-alpha chars
    text = re.sub(r"\s+", " ", text).strip()   # collapse whitespace
    return text


def load_and_clean_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the UCI SMS Spam Collection dataset and perform cleaning steps:
      1. Drop unnamed/empty columns (artifact of CSV export).
      2. Rename columns to 'label' and 'text'.
      3. Encode labels: ham=0, spam=1.
      4. Apply text cleaning.
      5. Remove duplicate messages.
    """
    df = pd.read_csv(filepath, encoding="latin-1")

    # Step 1: Drop unnamed columns
    df = df[["v1", "v2"]].copy()

    # Step 2: Rename
    df.columns = ["label", "text"]

    # Step 3: Encode labels
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Step 4: Clean text
    df["text"] = df["text"].apply(clean_text)

    # Step 5: Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"[preprocess] Removed {before - len(df)} duplicate rows. Remaining: {len(df)}")

    return df


def split_dataset(df: pd.DataFrame, val_ratio=0.1, test_ratio=0.1, seed=SEED):
    """
    Stratified train/val/test split preserving class balance.
    Default: 80% train, 10% val, 10% test.
    """
    from sklearn.model_selection import train_test_split

    X = df["text"].values
    y = df["label"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), stratify=y, random_state=seed
    )
    val_fraction = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_fraction), stratify=y_temp, random_state=seed
    )

    print(f"[split] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_tokenizer(X_train, vocab_size=MAX_VOCAB_SIZE):
    """Fit tokenizer on training data only to prevent leakage."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    return tokenizer


def encode_texts(tokenizer, texts, maxlen=MAX_SEQUENCE_LEN):
    """Convert text to padded integer sequences."""
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=maxlen, padding="post", truncating="post")


def save_tokenizer(tokenizer, path=TOKENIZER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[preprocess] Tokenizer saved → {path}")


def load_tokenizer(path=TOKENIZER_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)
