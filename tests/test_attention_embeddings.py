"""
Tests for attention-based embeddings.
"""

import logging
import pytest
import numpy as np
import torch
from src.model import LanguageModel
from src.embeddings import LlamaEmbeddingService

@pytest.fixture(scope="module")
def language_model():
    """Fixture to provide a language model for tests."""
    return LanguageModel("meta-llama/Llama-3.2-3B-Instruct", cache_dir="../model_cache")

@pytest.fixture(scope="module")
def embedding_service(language_model):
    """Fixture to provide an embedding service for tests."""
    return LlamaEmbeddingService(language_model)

def test_language_model_structure(language_model):
    """Verify the structure of the language model."""
    model = language_model
    
    # Check basic structure
    assert hasattr(model, 'model')
    assert hasattr(model.model, 'model')
    
    # Check layers
    assert hasattr(model.model.model, 'layers')
    assert len(model.model.model.layers) > 0
    
    # Check first layer's attention mechanism
    first_layer = model.model.model.layers[0]
    assert hasattr(first_layer, 'self_attn')
    
    # Check key and query projections
    assert hasattr(first_layer.self_attn, 'k_proj')
    assert hasattr(first_layer.self_attn, 'q_proj')
    assert isinstance(first_layer.self_attn.k_proj.weight, torch.Tensor)
    assert isinstance(first_layer.self_attn.q_proj.weight, torch.Tensor)

def test_standard_embedding(embedding_service):
    """Test standard embedding computation."""
    query_text = "What is artificial intelligence?"
    
    standard_embedding = embedding_service.get_embedding(query_text)
    
    assert isinstance(standard_embedding, np.ndarray)
    assert standard_embedding.shape[0] > 0
    assert np.linalg.norm(standard_embedding) > 0

def test_attention_key_embedding(embedding_service):
    """Test attention key embedding computation."""
    query_text = "What is artificial intelligence?"
    
    key_embedding = embedding_service._compute_attention_key_embedding(query_text)
    
    assert isinstance(key_embedding, np.ndarray)
    assert key_embedding.shape[0] > 0
    assert np.linalg.norm(key_embedding) > 0
    
    # Test non-normalized version too
    raw_key = embedding_service._compute_attention_key_embedding(query_text, normalize=False)
    assert isinstance(raw_key, np.ndarray)
    assert np.linalg.norm(raw_key) > 0

def test_attention_query_embedding(embedding_service):
    """Test attention query embedding computation."""
    query_text = "What is artificial intelligence?"
    
    query_embedding = embedding_service._compute_attention_query_embedding(query_text)
    
    assert isinstance(query_embedding, np.ndarray)
    assert query_embedding.shape[0] > 0
    assert np.linalg.norm(query_embedding) > 0
    
    # Test non-normalized version too
    raw_query = embedding_service._compute_attention_query_embedding(query_text, normalize=False)
    assert isinstance(raw_query, np.ndarray)
    assert np.linalg.norm(raw_query) > 0

def test_embedding_similarities(embedding_service):
    """Test similarities between different embedding types."""
    query_text = "What is artificial intelligence?"
    
    standard_embedding = embedding_service.get_embedding(query_text)
    key_embedding = embedding_service._compute_attention_key_embedding(query_text)
    query_embedding = embedding_service._compute_attention_query_embedding(query_text)
    
    # Check that all embeddings have the same dimensionality
    assert standard_embedding.shape == key_embedding.shape
    assert standard_embedding.shape == query_embedding.shape
    
    # Normalize standard embedding for similarity calculation
    normalized_standard = standard_embedding / np.linalg.norm(standard_embedding)
    
    # Compute similarities
    key_sim = np.dot(normalized_standard, key_embedding)
    query_sim = np.dot(normalized_standard, query_embedding)
    key_query_sim = np.dot(key_embedding, query_embedding)
    
    # We don't assert specific values, but the similarities should be between -1 and 1
    assert -1.0 <= key_sim <= 1.0
    assert -1.0 <= query_sim <= 1.0
    assert -1.0 <= key_query_sim <= 1.0 