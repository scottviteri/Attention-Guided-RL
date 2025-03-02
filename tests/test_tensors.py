"""
Pytest-compatible tests for tensor implementation.
"""

import pytest
import torch
from transformers import AutoTokenizer
from unittest.mock import patch
from src.data import chunk_tokens, split_key_value
from src.config import KEY_TOKEN_COUNT, VALUE_TOKEN_COUNT

@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

@pytest.fixture
def device():
    """Fixture to determine the device to use for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"

@patch('src.data.tokenize_articles')
def test_tokenize_into_chunks(mock_tokenize, tokenizer, device):
    """Test chunking tokens into fixed size chunks."""
    # Create a sufficiently long text by repeating a paragraph
    base_text = "This is a test article for tensor implementation verification. "
    # Repeat to ensure we have enough tokens
    text = base_text * 30  # Create a much longer text
    
    # Tokenize and set up mock return
    tokens = tokenizer.encode(text, add_special_tokens=False)
    mock_tokenize.return_value = iter([("Test Article", tokens)])
    
    # Run the chunking function with explicit device
    result = list(chunk_tokens(mock_tokenize(), device=device))
    
    # Verify results
    assert len(result) == 1  # One article
    title, chunks = result[0]
    assert title == "Test Article"
    assert len(chunks) > 0
    
    # All chunks should be full-sized and on the correct device
    chunk_size = KEY_TOKEN_COUNT + VALUE_TOKEN_COUNT
    for chunk in chunks:
        assert isinstance(chunk, torch.Tensor)
        assert chunk.shape[0] == chunk_size
        assert chunk.device.type == device  # Verify device

@patch('src.data.sample_chunks')
def test_create_key_value_pairs(mock_sample_chunks, device):
    """Test splitting chunks into key-value pairs."""
    # Create dummy chunk tensors with the correct size
    chunk_size = KEY_TOKEN_COUNT + VALUE_TOKEN_COUNT
    chunks = [
        torch.tensor(list(range(chunk_size)), dtype=torch.long, device=device),
        torch.tensor(list(range(chunk_size, chunk_size * 2)), dtype=torch.long, device=device)
    ]
    
    # Set up mock return
    mock_sample_chunks.return_value = iter([("Test Article", chunks)])
    
    # Split into key-value pairs
    result = list(split_key_value(mock_sample_chunks()))
    
    # Verify results
    assert len(result) == 1  # One article
    title, pairs = result[0]
    assert title == "Test Article"
    assert len(pairs) > 0
    
    # Verify key-value sizes and device
    for key, value in pairs:
        assert isinstance(key, torch.Tensor)
        assert isinstance(value, torch.Tensor)
        assert key.shape[0] == KEY_TOKEN_COUNT
        assert value.shape[0] == VALUE_TOKEN_COUNT
        assert key.device.type == device  # Verify device
        assert value.device.type == device  # Verify device 