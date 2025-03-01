"""
Tests for attention-based embeddings.
"""

import logging
import pytest
import numpy as np
import torch
from src.model import LanguageModel
from src.embeddings import (
    get_embedding, 
    compute_llama_embedding,
    compute_attention_key_embedding,
    compute_attention_query_embedding
)

@pytest.fixture(scope="module")
def language_model(language_model_cpu):
    """Fixture to provide a language model for tests."""
    # Use the shared session fixture from conftest.py
    return language_model_cpu

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

def test_standard_embedding(language_model):
    """Test standard embedding computation."""
    query_text = "What is artificial intelligence?"
    
    standard_embedding = compute_llama_embedding(query_text, model=language_model)
    
    assert isinstance(standard_embedding, np.ndarray)
    assert standard_embedding.shape[0] > 0
    assert np.linalg.norm(standard_embedding) > 0

def test_attention_key_embedding(language_model):
    """Test attention key embedding computation."""
    query_text = "What is artificial intelligence?"
    
    key_embedding = compute_attention_key_embedding(query_text, model=language_model)
    
    assert isinstance(key_embedding, np.ndarray)
    assert key_embedding.shape[0] > 0
    assert np.linalg.norm(key_embedding) > 0

def test_attention_query_embedding(language_model):
    """Test attention query embedding computation."""
    query_text = "What is artificial intelligence?"
    
    query_embedding = compute_attention_query_embedding(query_text, model=language_model)
    
    assert isinstance(query_embedding, np.ndarray)
    assert query_embedding.shape[0] > 0
    assert np.linalg.norm(query_embedding) > 0
    
    # We no longer test normalized vs non-normalized versions since we've
    # standardized on using scaled dot product attention without normalization

def test_embedding_similarities(language_model):
    """Test similarities between different embedding types."""
    query_text = "What is artificial intelligence?"
    
    standard_embedding = compute_llama_embedding(query_text, model=language_model)
    key_embedding = compute_attention_key_embedding(query_text, model=language_model)
    query_embedding = compute_attention_query_embedding(query_text, model=language_model)
    
    # Check that embeddings have reasonable dimensions
    assert standard_embedding.shape[0] > 0
    assert key_embedding.shape[0] > 0
    assert query_embedding.shape[0] > 0
    
    # Note: We no longer expect embeddings to have the same dimensionality
    # The standard embedding comes from the full model (3072 dims)
    # Key and query embeddings might be from attention projections with different dimensions
    # Their dimensions are implementation-dependent
    
    # We can't directly compare key and query embeddings if they have different dimensions
    if key_embedding.shape == query_embedding.shape:
        # Only compute similarity if dimensions match
        key_query_sim = np.dot(key_embedding, query_embedding)
        assert -1.0 <= key_query_sim <= 1.0 