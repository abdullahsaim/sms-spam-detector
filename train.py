"""
train.py — Model definition and training for SMS Spam Detection.
Author: abdullahsaim

Usage:
    python train.py --epochs 10 --batch_size 32 --lr 0.001 --embed_dim 64 --dropout 0.4
"""

import argparse
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam

from preprocess import (
    load_and_clean_dataset, split_dataset, build_tokenizer,
    encode_texts, save_tokenizer, MAX_VOCAB_SIZE, MAX_SEQUENCE_LEN
)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds()

# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train SMS Spam Detector")
    parser.add_argument("--data",       type=str,   default="data/spam.csv")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--embed_dim",  type=int,   default=64)
    parser.add_argument("--dropout",    type=float, default=0.4)
    parser.add_argument("--patience",   type=int,   default=3)
    return parser.parse_args()


# ── Model Definition ──────────────────────────────────────────────────────────
def build_model(vocab_size, embed_dim, maxlen, dropout_rate, lr):
    """
    BiLSTM model chosen over:
      - Naive Bayes: cannot capture word order / context.
      - Plain LSTM: BiLSTM reads sequence both directions, better for short SMS.
    Architecture: Embedding → BiLSTM → GlobalMaxPool → Dense(64) → Output.
    Binary cross-entropy loss with sigmoid output for binary classification.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim,
                  input_length=maxlen, name="embedding"),
        Bidirectional(LSTM(64, return_sequences=True), name="bilstm"),
        GlobalMaxPooling1D(name="global_max_pool"),
        Dense(64, activation="relu", name="dense_hidden"),
        Dropout(dropout_rate, name="dropout"),
        Dense(1, activation="sigmoid", name="output")
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model


# ── Training Pipeline ─────────────────────────────────────────────────────────
def train(args):
    print("\n[train] Loading & preprocessing data...")
    df = load_and_clean_dataset(args.data)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    tokenizer = build_tokenizer(X_train, vocab_size=MAX_VOCAB_SIZE)
    save_tokenizer(tokenizer)

    X_train_enc = encode_texts(tokenizer, X_train, maxlen=MAX_SEQUENCE_LEN)
    X_val_enc   = encode_texts(tokenizer, X_val,   maxlen=MAX_SEQUENCE_LEN)
    X_test_enc  = encode_texts(tokenizer, X_test,  maxlen=MAX_SEQUENCE_LEN)

    print(f"\n[train] Building model — embed_dim={args.embed_dim}, "
          f"dropout={args.dropout}, lr={args.lr}")
    model = build_model(
        vocab_size=MAX_VOCAB_SIZE + 1,
        embed_dim=args.embed_dim,
        maxlen=MAX_SEQUENCE_LEN,
        dropout_rate=args.dropout,
        lr=args.lr
    )
    model.summary()

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint("artifacts/best_model.keras",
                        monitor="val_loss", save_best_only=True, verbose=1),
        CSVLogger("logs/training_log.csv", append=False)
    ]

    print("\n[train] Starting training...")
    history = model.fit(
        X_train_enc, y_train,
        validation_data=(X_val_enc, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # ── Final validation log line ─────────────────────────────────────────────
    best_epoch = np.argmin(history.history["val_loss"])
    print(f"\n[train] ✓ Best epoch {best_epoch + 1} | "
          f"val_loss={history.history['val_loss'][best_epoch]:.4f} | "
          f"val_accuracy={history.history['val_accuracy'][best_epoch]:.4f} | "
          f"val_auc={history.history['val_auc'][best_epoch]:.4f} | "
          f"checkpoint=artifacts/best_model.keras")

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\n[train] Evaluating on test set...")
    results = model.evaluate(X_test_enc, y_test, verbose=0)
    print(f"[test]  loss={results[0]:.4f} | accuracy={results[1]:.4f} | "
          f"auc={results[2]:.4f} | precision={results[3]:.4f} | recall={results[4]:.4f}")

    # Save test data for error analysis
    np.save("artifacts/X_test.npy", X_test_enc)
    np.save("artifacts/y_test.npy", y_test)
    print("[train] Test arrays saved for error analysis.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
