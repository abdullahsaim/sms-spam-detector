"""
evaluate.py — Evaluation, confusion matrix, and error analysis.
Author: abdullahsaim
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score
)
from tensorflow.keras.models import load_model
from preprocess import load_tokenizer, load_and_clean_dataset, split_dataset, encode_texts

THRESHOLD = 0.5


def evaluate():
    print("[evaluate] Loading model and test data...")
    model = load_model("artifacts/best_model.keras")
    X_test = np.load("artifacts/X_test.npy")
    y_test = np.load("artifacts/y_test.npy")

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= THRESHOLD).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    cm = confusion_matrix(y_test, y_pred)
    print("── Confusion Matrix (rows=actual, cols=predicted) ──")
    print(f"           Pred Ham  Pred Spam")
    print(f"Actual Ham    {cm[0][0]:5d}      {cm[0][1]:5d}")
    print(f"Actual Spam   {cm[1][0]:5d}      {cm[1][1]:5d}")

    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n[evaluate] F1={f1:.4f} | AUROC={auc:.4f}")

    # ── Error Analysis — False Positives (ham predicted as spam) ──────────────
    print("\n── Error Analysis: False Positives (ham→spam) ──")
    df = load_and_clean_dataset("data/spam.csv")
    _, _, X_text_test, _, _, y_test_raw = split_dataset(df)

    fp_indices = np.where((y_pred == 1) & (y_test == 0))[0]
    fn_indices = np.where((y_pred == 0) & (y_test == 1))[0]

    print(f"False Positives: {len(fp_indices)} | False Negatives: {len(fn_indices)}")

    print("\nTop 5 False Positives (ham messages predicted as spam):")
    for i in fp_indices[:5]:
        print(f"  [{i}] score={y_prob[i]:.3f} | text: {X_text_test[i][:80]}")

    print("\nTop 5 False Negatives (spam messages missed):")
    for i in fn_indices[:5]:
        print(f"  [{i}] score={y_prob[i]:.3f} | text: {X_text_test[i][:80]}")

    # ── Key Failure Mode ──────────────────────────────────────────────────────
    print("\n── Key Failure Mode ──")
    print("FP cell in confusion matrix: ham messages with promotional-style")
    print("language (e.g. 'free', 'win', 'call now') are misclassified as spam.")
    print("Fix attempted: increased dropout from 0.3→0.4 to reduce overfit on")
    print("spam-associated tokens; added more ham examples via stratified split.")


if __name__ == "__main__":
    evaluate()
