"""
Tests for the embeddings module.
"""

import os
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Import the module to test
from src.embeddings import (
    compute_embeddings,
    get_embeddings,
    compute_similarity,
    compute_embeddings_batch,
    process_hidden_states_batch
)
from src.model import LanguageModel

@pytest.fixture(scope="module")
def language_model():
    """Fixture to provide a language model for tests."""
    return LanguageModel(cache_dir="../model_cache", device="cpu")

@patch('src.embeddings.get_embeddings')
def test_compute_embeddings(mock_get_embeddings, language_model):
    """Test the compute_embeddings function."""
    # Setup mock return value
    mock_get_embeddings.return_value = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6])
    ]
    
    # Call function with default parameters and the pre-loaded model
    texts = ["Hello world", "Test text"]
    embeddings = compute_embeddings(texts, model=language_model)

    # Verify function was called correctly
    mock_get_embeddings.assert_called_once_with(
        texts=texts,
        model=language_model,
        are_queries=True,
        use_cache=True,
        cache_dir="embedding_cache",
        max_length=512,
        batch_size=16
    )

    # Verify results
    assert len(embeddings) == 2
    assert embeddings[0].tolist() == [0.1, 0.2, 0.3]
    assert embeddings[1].tolist() == [0.4, 0.5, 0.6]

@patch('src.embeddings.get_embeddings')
def test_compute_embeddings_empty_input(mock_get_embeddings, language_model):
    """Test compute_embeddings with empty input."""
    # Call with empty list and pre-loaded model
    result = compute_embeddings([], model=language_model)

    # Verify get_embeddings was not called
    mock_get_embeddings.assert_not_called()

    # Verify empty result
    assert result == []

@patch('src.embeddings.get_embeddings')
def test_compute_embeddings_with_query_flags(mock_get_embeddings, language_model):
    """Test compute_embeddings with mixed query flags."""
    # Setup mock return value
    mock_get_embeddings.return_value = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6])
    ]
    
    # Call with query flag set to False and pre-loaded model
    texts = ["Hello world", "Test text"]
    are_queries = False
    embeddings = compute_embeddings(texts, are_queries=are_queries, model=language_model)

    # Verify function was called correctly
    mock_get_embeddings.assert_called_with(
        texts=texts,
        model=language_model,
        are_queries=are_queries,
        use_cache=True,
        cache_dir="embedding_cache",
        max_length=512,
        batch_size=16
    )

    # Verify results
    assert len(embeddings) == 2

@patch('src.embeddings.get_embeddings')
def test_compute_embeddings_with_parameters(mock_get_embeddings, language_model):
    """Test compute_embeddings with custom parameters."""
    # Setup mock return value
    mock_get_embeddings.return_value = [np.array([0.1, 0.2, 0.3])]
    
    # Call with custom parameters and pre-loaded model
    texts = ["Hello world"]
    embeddings = compute_embeddings(
        texts,
        model=language_model,
        are_queries=False,
        use_cache=False,
        cache_dir="custom_cache",
        max_length=256,
        batch_size=8
    )

    # Verify function was called correctly
    mock_get_embeddings.assert_called_with(
        texts=texts,
        model=language_model,
        are_queries=False,
        use_cache=False,
        cache_dir="custom_cache",
        max_length=256,
        batch_size=8
    )

def test_compute_similarity():
    """Test the compute_similarity function."""
    # Create a query embedding
    query_embedding = np.array([0.5, 0.5, 0.7071])  # Not normalized
    
    # Create key embeddings
    key_embeddings = {
        "key1": np.array([1.0, 0.0, 0.0]),  # Normalized
        "key2": np.array([0.0, 1.0, 0.0]),  # Normalized
        "key3": np.array([0.0, 0.0, 2.0])   # Not normalized
    }
    
    # Compute similarities
    similarities = compute_similarity(query_embedding, key_embeddings)
    
    # Check results - after normalization, the dot products should be:
    # query · key1 ≈ 0.5/sqrt(0.5² + 0.5² + 0.7071²) ≈ 0.5 / 1.0 = 0.5
    # query · key2 ≈ 0.5/sqrt(0.5² + 0.5² + 0.7071²) ≈ 0.5 / 1.0 = 0.5
    # query · key3 ≈ 0.7071/sqrt(0.5² + 0.5² + 0.7071²) ≈ 0.7071 / 1.0 = 0.7071
    assert len(similarities) == 3
    assert "key1" in similarities
    assert "key2" in similarities
    assert "key3" in similarities
    
    # Check approximate values (allowing for floating point precision)
    assert abs(similarities["key1"] - 0.5) < 0.01
    assert abs(similarities["key2"] - 0.5) < 0.01
    assert abs(similarities["key3"] - 0.7071) < 0.01
    
    # Test with empty key_embeddings
    empty_similarities = compute_similarity(query_embedding, {})
    assert empty_similarities == {}

@patch('torch.no_grad')
@patch('src.embeddings.tokenize_text_batch')
def test_compute_embeddings_batch(mock_tokenize, mock_no_grad, language_model):
    """Test compute_embeddings_batch function with mocked dependencies."""
    # Create mock inputs and outputs
    mock_inputs = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]])
    }
    mock_tokenize.return_value = mock_inputs
    
    # Create mock model outputs
    mock_hidden_states = [torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                                       [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]])]
    
    # Mock the model's forward pass
    mock_outputs = MagicMock()
    mock_outputs.hidden_states = mock_hidden_states
    language_model.model = MagicMock()
    language_model.model.return_value = mock_outputs
    
    # Call the function
    texts = ["Text 1", "Text 2"]
    result = compute_embeddings_batch(texts, language_model)
    
    # Verify the function was called correctly
    mock_tokenize.assert_called_once_with(texts, language_model.tokenizer, 512, language_model.device)
    language_model.model.assert_called_once()
    
    # Verify result structure
    assert len(result) == 2
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)

def test_process_hidden_states_batch():
    """Test process_hidden_states_batch function."""
    # Create synthetic hidden states and attention mask
    hidden_states = [
        torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # Batch item 1
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]  # Batch item 2
        ])
    ]
    attention_mask = torch.tensor([
        [1, 1, 0],  # Batch item 1 - only first 2 tokens are valid
        [1, 1, 1]   # Batch item 2 - all 3 tokens are valid
    ])
    
    # Process hidden states
    result = process_hidden_states_batch(hidden_states, attention_mask)
    
    # Expected result:
    # Batch item 1: average of first two tokens (since third is masked): [[1,2], [3,4]] -> [2, 3]
    # Batch item 2: average of all three tokens: [[7,8], [9,10], [11,12]] -> [9, 10]
    expected_item1 = np.array([2.0, 3.0])  # (1+3)/2, (2+4)/2
    expected_item2 = np.array([9.0, 10.0])  # (7+9+11)/3, (8+10+12)/3
    
    # Check results (with tolerance for floating point)
    assert result.shape == (2, 2)  # 2 batch items, 2 dims each
    np.testing.assert_allclose(result[0], expected_item1, rtol=1e-5)
    np.testing.assert_allclose(result[1], expected_item2, rtol=1e-5) 