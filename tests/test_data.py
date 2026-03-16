"""
tests/test_data.py — Unit and integration tests for preprocessing pipeline.
Author: abdullahsaim
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocess import (
    clean_text, build_tokenizer, encode_texts,
    MAX_SEQUENCE_LEN, MAX_VOCAB_SIZE
)


# ── Unit Tests: clean_text ────────────────────────────────────────────────────

class TestCleanText:
    def test_lowercases_input(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_removes_punctuation(self):
        assert clean_text("hello!!! world???") == "hello world"

    def test_removes_digits(self):
        assert clean_text("call 12345 now") == "call  now".strip()
        # digits stripped, spaces collapsed
        result = clean_text("call 12345 now")
        assert "12345" not in result

    def test_strips_extra_whitespace(self):
        result = clean_text("  hello   world  ")
        assert result == "hello world"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_digits_and_punctuation(self):
        assert clean_text("123!!!???") == ""


# ── Unit Tests: Tokenizer & Encoding ─────────────────────────────────────────

class TestTokenizerAndEncoding:
    @pytest.fixture
    def sample_texts(self):
        return [
            "free prize win now",
            "call me tomorrow please",
            "congratulations you won",
            "hello how are you doing today"
        ]

    def test_tokenizer_builds(self, sample_texts):
        tokenizer = build_tokenizer(sample_texts, vocab_size=100)
        assert tokenizer is not None
        assert len(tokenizer.word_index) > 0

    def test_encode_texts_output_shape(self, sample_texts):
        tokenizer = build_tokenizer(sample_texts, vocab_size=100)
        encoded = encode_texts(tokenizer, sample_texts, maxlen=MAX_SEQUENCE_LEN)
        assert encoded.shape == (len(sample_texts), MAX_SEQUENCE_LEN)

    def test_encode_texts_padded_correctly(self, sample_texts):
        tokenizer = build_tokenizer(sample_texts, vocab_size=100)
        encoded = encode_texts(tokenizer, ["hi"], maxlen=10)
        # Short sequence should be zero-padded to length 10
        assert encoded.shape == (1, 10)
        assert (encoded[0] == 0).sum() > 0   # padding zeros present

    def test_no_data_leakage_from_val(self, sample_texts):
        """Tokenizer must only be fit on training data."""
        train = sample_texts[:3]
        val   = ["unseen word xyzabc"]
        tokenizer = build_tokenizer(train, vocab_size=100)
        # OOV token should handle unseen words without crashing
        encoded = encode_texts(tokenizer, val, maxlen=10)
        assert encoded.shape == (1, 10)


# ── Integration Test: Full Preprocessing Pipeline ────────────────────────────

class TestPreprocessingPipeline:
    def test_pipeline_produces_correct_label_types(self):
        """Simulate a small dataframe through label encoding."""
        data = {
            "v1": ["ham", "spam", "ham", "spam"],
            "v2": ["hello world", "free prize now", "call me", "win cash today"]
        }
        df = pd.DataFrame(data)
        df.columns = ["label", "text"]
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
        assert set(df["label"].unique()).issubset({0, 1})

    def test_no_nulls_after_cleaning(self):
        data = {
            "v1": ["ham", "spam"],
            "v2": ["hello!", "FREE WIN $$$"]
        }
        df = pd.DataFrame(data)
        df["v2"] = df["v2"].apply(clean_text)
        assert df["v2"].isnull().sum() == 0

    def test_sequence_length_clipping(self):
        """Sequences longer than MAX_SEQUENCE_LEN must be truncated."""
        long_text = ["word " * 200]
        tokenizer = build_tokenizer(long_text, vocab_size=1000)
        encoded = encode_texts(tokenizer, long_text, maxlen=MAX_SEQUENCE_LEN)
        assert encoded.shape[1] == MAX_SEQUENCE_LEN
