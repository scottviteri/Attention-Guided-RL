"""
Pytest-compatible tests for tensor implementation.
"""

import pytest
import torch
from transformers import AutoTokenizer
from src.data import tokenize_article_into_chunks, create_key_value_pairs_from_chunks

@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

def test_tokenize_article_into_chunks(tokenizer):
    text = """
    This is a test article for verifying tensor implementation.
    We want to make sure that the tokenization and key-value pair creation
    works correctly with PyTorch tensors. This text should be long enough
    to produce multiple chunks for our test.
    """
    chunks = tokenize_article_into_chunks(
        text=text,
        tokenizer=tokenizer,
        chunk_size=10,
        max_chunks=5,
        skip_if_too_short=False
    )
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, torch.Tensor)
        assert chunk.shape[0] <= 10

def test_create_key_value_pairs_from_chunks(tokenizer):
    text = """
    This is a test article for verifying tensor implementation.
    We want to make sure that the tokenization and key-value pair creation
    works correctly with PyTorch tensors. This text should be long enough
    to produce multiple chunks for our test.
    """
    chunks = tokenize_article_into_chunks(
        text=text,
        tokenizer=tokenizer,
        chunk_size=10,
        max_chunks=5,
        skip_if_too_short=False
    )
    pairs = create_key_value_pairs_from_chunks(
        chunks=chunks,
        max_pairs=3,
        non_overlapping=True
    )
    assert len(pairs) > 0
    for key, value in pairs:
        assert isinstance(key, torch.Tensor)
        assert isinstance(value, torch.Tensor)
        assert key.shape[0] <= 10
        assert value.shape[0] <= 10 