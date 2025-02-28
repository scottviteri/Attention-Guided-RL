"""
Tests for the embeddings module.
"""

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module to test
from src.embeddings import (
    compute_embeddings,
    LlamaEmbeddingService
)
from src.model import LanguageModel

@pytest.fixture(scope="module")
def language_model():
    """Fixture to provide a language model for tests."""
    return LanguageModel(cache_dir="../model_cache", device="cpu")

@patch('src.embeddings.LlamaEmbeddingService')
def test_compute_embeddings(mock_service_class, language_model):
    """Test the compute_embeddings function."""
    # Configure the mock service
    mock_service = MagicMock(spec=LlamaEmbeddingService)
    mock_service.get_embeddings.return_value = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6])
    ]
    mock_service_class.return_value = mock_service
    
    # Call function with default parameters and the pre-loaded model
    texts = ["Hello world", "Test text"]
    embeddings = compute_embeddings(texts, model=language_model)

    # Verify service was created and called correctly
    mock_service_class.assert_called_once_with(model=language_model, use_cache=True)
    mock_service.get_embeddings.assert_called_with(
        texts, 
        are_queries=[True, True],
        normalize=False
    )

    # Verify results
    assert len(embeddings) == 2
    assert embeddings[0].tolist() == [0.1, 0.2, 0.3]
    assert embeddings[1].tolist() == [0.4, 0.5, 0.6]

@patch('src.embeddings.LlamaEmbeddingService')
def test_compute_embeddings_empty_input(mock_service_class, language_model):
    """Test compute_embeddings with empty input."""
    # Setup mock
    mock_service = MagicMock(spec=LlamaEmbeddingService)
    mock_service_class.return_value = mock_service
    
    # Call with empty list and pre-loaded model
    result = compute_embeddings([], model=language_model)

    # Verify no service instantiation
    mock_service_class.assert_not_called()

    # Verify empty result
    assert result == []

@patch('src.embeddings.LlamaEmbeddingService')
def test_compute_embeddings_with_query_flags(mock_service_class, language_model):
    """Test compute_embeddings with mixed query flags."""
    # Setup mock
    mock_service = MagicMock(spec=LlamaEmbeddingService)
    mock_service.get_embeddings.return_value = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6])
    ]
    mock_service_class.return_value = mock_service
    
    # Call with mixed flags and pre-loaded model
    texts = ["Hello world", "Test text"]
    are_queries = [True, False]
    embeddings = compute_embeddings(texts, are_queries=are_queries, model=language_model)

    # Verify service was called correctly
    mock_service.get_embeddings.assert_called_with(
        texts, 
        are_queries=are_queries,
        normalize=False
    )

    # Verify results
    assert len(embeddings) == 2

@patch('src.embeddings.LlamaEmbeddingService')
def test_compute_embeddings_with_normalization(mock_service_class, language_model):
    """Test compute_embeddings with normalization."""
    # Setup mock
    mock_service = MagicMock(spec=LlamaEmbeddingService)
    mock_service.get_embeddings.return_value = [np.array([0.1, 0.2, 0.3])]
    mock_service_class.return_value = mock_service
    
    # Call with normalization and pre-loaded model
    texts = ["Hello world"]
    embeddings = compute_embeddings(texts, normalize=True, model=language_model)

    # Verify service was called correctly
    mock_service.get_embeddings.assert_called_with(
        texts, 
        are_queries=[True],
        normalize=True
    ) 