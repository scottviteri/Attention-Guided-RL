"""
Tests for the Llama-based embedding implementation.
"""

import pytest
import time
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from src.model import LanguageModel
from src.embeddings import compute_embeddings, get_embeddings, get_embedding

@pytest.fixture(scope="module")
def language_model():
    """Fixture to provide a language model for tests."""
    return LanguageModel(cache_dir="../model_cache", device="cpu")

def test_get_embedding(language_model):
    """Test getting a single embedding."""
    text = "What is artificial intelligence?"
    
    embedding = get_embedding(text, model=language_model)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] > 0
    assert np.linalg.norm(embedding) > 0

def test_get_batch_embeddings(language_model):
    """Test getting batch embeddings."""
    texts = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?",
    ]
    
    embeddings = get_embeddings(texts, model=language_model, are_queries=True)
    
    assert len(embeddings) == len(texts)
    for emb in embeddings:
        assert isinstance(emb, np.ndarray)
        assert emb.shape[0] > 0
        assert np.linalg.norm(emb) > 0

@patch('src.embeddings.compute_llama_embedding')
def test_embedding_cache(mock_compute_llama, language_model):
    """Test that the cache works correctly."""
    text = "What is artificial intelligence?"
    
    # Get the embedding dimensions from the model
    hidden_size = language_model.model.model.embed_tokens.embedding_dim
    
    # Set up mock with appropriate dimensions
    mock_vector = np.ones(hidden_size)
    mock_compute_llama.return_value = mock_vector
    
    # First call should compute the embedding
    first_embedding = get_embedding(text, model=language_model, use_cache=True)
    
    # Second call should use the cache
    with patch('src.embeddings.load_from_cache') as mock_load_cache:
        mock_load_cache.return_value = mock_vector
        second_embedding = get_embedding(text, model=language_model, use_cache=True)
        # Check that load_from_cache was called
        mock_load_cache.assert_called_once()
    
    # The embeddings should be identical (but we will use the actual shape from the first embedding)
    np.testing.assert_array_equal(second_embedding, mock_vector)

@patch('src.embeddings.compute_llama_embedding')
def test_batch_embeddings_cache(mock_compute_llama, language_model):
    """Test batch embeddings with cache."""
    texts = [
        "What is artificial intelligence?",
        "How do neural networks work?",
        "Explain machine learning concepts."
    ]
    
    # Get the embedding dimensions from the model
    hidden_size = language_model.model.model.embed_tokens.embedding_dim
    
    # Set up mock with appropriate dimensions
    mock_vector = np.ones(hidden_size)
    mock_compute_llama.return_value = mock_vector
    
    # Cache should be checked for each text
    with patch('src.embeddings.load_from_cache') as mock_load_cache:
        mock_load_cache.return_value = None  # Cache miss for all
        batch_embeddings = get_embeddings(texts, model=language_model, are_queries=True, use_cache=True)
        assert mock_load_cache.call_count == len(texts)
    
    # All returned embeddings should have the same shape
    first_shape = batch_embeddings[0].shape
    for embedding in batch_embeddings:
        assert embedding.shape == first_shape
        assert np.linalg.norm(embedding) > 0

@patch('src.embeddings.get_embeddings')
def test_compute_embeddings_integration(mock_get_embeddings, language_model):
    """Integration test for the compute_embeddings function."""
    text = "What is artificial intelligence?"
    
    # Set up mock return value with correct dimensions
    hidden_size = language_model.model.model.embed_tokens.embedding_dim
    mock_embedding = np.random.rand(hidden_size)
    mock_get_embeddings.return_value = [mock_embedding]
    
    # Test with compute_embeddings directly using pre-loaded model
    embedding = compute_embeddings([text], model=language_model)[0]
    
    # Verify result
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] > 0
    
    # Verify the function was called correctly (using any_order=True to ignore parameter order)
    mock_get_embeddings.assert_called_once()
    call_args = mock_get_embeddings.call_args
    assert call_args[1]['texts'] == [text]
    assert call_args[1]['model'] == language_model
    assert call_args[1]['are_queries'] == True
    assert call_args[1]['use_cache'] == True
    assert 'cache_dir' in call_args[1]
    assert 'max_length' in call_args[1]

def test_similarity_between_embeddings(language_model):
    """Test similarity calculations between embeddings."""
    text1 = "What is artificial intelligence?"
    text2 = "AI is the simulation of human intelligence by machines."
    text3 = "The capital of France is Paris."
    
    # Get embeddings and normalize them manually
    embedding1 = get_embedding(text1, model=language_model)
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    
    embedding2 = get_embedding(text2, model=language_model)
    # Ensure embedding2 is 1D if it's 2D
    if embedding2.ndim > 1:
        embedding2 = embedding2.squeeze()
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    
    embedding3 = get_embedding(text3, model=language_model)
    # Ensure embedding3 is 1D if it's 2D
    if embedding3.ndim > 1:
        embedding3 = embedding3.squeeze()
    embedding3 = embedding3 / np.linalg.norm(embedding3)
    
    # Compute similarities
    sim_1_2 = np.dot(embedding1, embedding2)
    sim_1_3 = np.dot(embedding1, embedding3)
    sim_2_3 = np.dot(embedding2, embedding3)
    
    # Check that similarity between related concepts is higher
    assert sim_1_2 > sim_1_3, f"Expected similarity between AI questions to be higher than unrelated topics, but got {sim_1_2} <= {sim_1_3}"
    assert sim_1_2 > sim_2_3, f"Expected similarity between AI questions to be higher than unrelated topics, but got {sim_1_2} <= {sim_2_3}" 