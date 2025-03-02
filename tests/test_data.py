"""
Tests for the data module.
"""

import pytest
import numpy as np
import torch
from typing import List, Iterator
from unittest.mock import patch

# Import the module to test
from src.data import (
    KeyValuePair,
    tokenize_articles,
    chunk_tokens,
    sample_chunks,
    split_key_value,
    to_dataclass
)

@patch('src.data.load_dataset')
def test_tokenize_articles(mock_load_dataset):
    """Test article tokenization."""
    # Mock the dataset
    mock_load_dataset.return_value = iter([
        {"title": "Test1", "text": "Content1" * 100},  # Long enough content
        {"title": "", "text": "Content2"},             # Missing title, should skip
        {"title": "Test3", "text": ""}                 # Missing text, should skip
    ])
    
    # Mock the tokenizer
    with patch('src.data.AutoTokenizer') as mock_tokenizer_class:
        mock_tokenizer = mock_tokenizer_class.from_pretrained.return_value
        # Return 200 tokens for the first article (above MIN_ARTICLE_TOKENS)
        mock_tokenizer.encode.return_value = list(range(200))
        
        # Run the function
        result = list(tokenize_articles())
        
        # Verify result
        assert len(result) == 1
        assert result[0][0] == "Test1"
        assert result[0][1] == list(range(200))

@patch('src.data.tokenize_articles')
def test_chunk_tokens(mock_tokenize):
    """Test chunking tokens."""
    # Set up return value
    mock_tokenize.return_value = iter([
        ("Test1", list(range(100)))  # 100 tokens
    ])
    
    # Get preferred device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Run the function with explicit device
    result = list(chunk_tokens(mock_tokenize(), device=device))
    
    # Verify result
    assert len(result) == 1
    title, chunks = result[0]
    assert title == "Test1"
    
    # Calculate expected chunks (assuming KEY_TOKEN_COUNT + VALUE_TOKEN_COUNT = 64)
    chunk_size = 64
    expected_chunk_count = 100 // chunk_size
    assert len(chunks) == expected_chunk_count
    
    # Verify first chunk
    first_chunk = chunks[0]
    assert torch.equal(first_chunk, torch.tensor(list(range(chunk_size)), dtype=torch.long, device=device))
    # Verify device
    assert first_chunk.device.type == device

@patch('src.data.chunk_tokens')
def test_sample_chunks(mock_chunk_tokens):
    """Test sampling chunks."""
    # Get preferred device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create 20 mock chunks (more than MAX_PAIRS which is 10)
    all_chunks = [torch.tensor([i], device=device) for i in range(20)]
    mock_chunk_tokens.return_value = iter([
        ("Test1", all_chunks)
    ])
    
    # Run the function
    result = list(sample_chunks(mock_chunk_tokens()))
    
    # Verify result - should sample MAX_PAIRS (10) chunks
    assert len(result) == 1
    title, sampled = result[0]
    assert title == "Test1"
    assert len(sampled) == 10  # MAX_PAIRS is 10
    
    # Each chunk should be a tensor and should come from the original set
    for chunk in sampled:
        assert isinstance(chunk, torch.Tensor)
        assert any(torch.equal(chunk, orig) for orig in all_chunks)
        # Verify device
        assert chunk.device.type == device

@patch('src.data.sample_chunks')
def test_split_key_value(mock_sample_chunks):
    """Test splitting chunks."""
    # Get preferred device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create mock chunks of size 64 (KEY_TOKEN_COUNT + VALUE_TOKEN_COUNT)
    chunks = [
        torch.tensor(list(range(64)), dtype=torch.long, device=device),
        torch.tensor(list(range(64, 128)), dtype=torch.long, device=device)
    ]
    mock_sample_chunks.return_value = iter([
        ("Test1", chunks)
    ])
    
    # Run the function
    result = list(split_key_value(mock_sample_chunks()))
    
    # Verify result
    assert len(result) == 1
    title, pairs = result[0]
    assert title == "Test1"
    assert len(pairs) == 2
    
    # Verify pairs are split correctly
    key1, value1 = pairs[0]
    assert torch.equal(key1, torch.tensor(list(range(24)), dtype=torch.long, device=device))  # KEY_TOKEN_COUNT = 24
    assert torch.equal(value1, torch.tensor(list(range(24, 64)), dtype=torch.long, device=device))  # VALUE_TOKEN_COUNT = 40
    # Verify device
    assert key1.device.type == device
    assert value1.device.type == device

def test_to_dataclass():
    """Test converting to KeyValuePair dataclasses."""
    # Get preferred device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create test input
    article_embedded_pairs = iter([
        ("Test1", [
            (
                torch.tensor([1, 2], device=device),
                torch.tensor([3, 4], device=device),
                np.array([0.1, 0.2])
            )
        ])
    ])
    
    # Run the function
    result = list(to_dataclass(article_embedded_pairs))
    
    # Verify result
    assert len(result) == 1
    pairs = result[0]
    assert len(pairs) == 1
    
    # Check dataclass attributes
    kv_pair = pairs[0]
    assert isinstance(kv_pair, KeyValuePair)
    assert torch.equal(kv_pair.key_tokens, torch.tensor([1, 2], device=device))
    assert torch.equal(kv_pair.value_tokens, torch.tensor([3, 4], device=device))
    np.testing.assert_array_equal(kv_pair.key_embedding, np.array([0.1, 0.2]))
    assert kv_pair.key_id == "Test1_0" 