"""
Tests for attention-based embeddings.
"""

import logging
import pytest
import numpy as np
import torch
from src.model import LanguageModel
from src.embeddings import (
    compute_attention_key_embeddings,
    compute_attention_query_embeddings
)

@pytest.fixture(scope="module")
def language_model(language_model_cuda):
    """Fixture to provide a language model for tests. Using CUDA version for better performance."""
    # Use the CUDA model fixture from conftest.py
    return language_model_cuda

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
    
    # Verify model is on CUDA
    assert language_model.device == "cuda"
    assert next(language_model.model.parameters()).is_cuda

def test_attention_key_embedding(language_model):
    """Test attention key embedding computation."""
    query_text = "What is artificial intelligence?"
    
    # Tokenize the text first
    tokens = language_model.tokenizer(query_text, return_tensors="pt")["input_ids"]
    
    # Get embedding from tokens directly
    key_embedding = compute_attention_key_embeddings(tokens, model=language_model)
    
    # Check that embedding is non-empty
    assert key_embedding.shape[0] > 0
    
    # Check that the model is on the correct device during operation
    assert language_model.device == "cuda"
    assert next(language_model.model.parameters()).is_cuda

def test_attention_query_embedding(language_model):
    """Test attention query embedding computation."""
    query_text = "What is artificial intelligence?"
    
    # Tokenize the text first
    tokens = language_model.tokenizer(query_text, return_tensors="pt")["input_ids"]
    
    # Get embedding from tokens directly
    query_embedding = compute_attention_query_embeddings(tokens, model=language_model)
    
    # Check that embedding is non-empty
    assert query_embedding.shape[0] > 0
    
    # Check that the model is on the correct device during operation
    assert language_model.device == "cuda"
    assert next(language_model.model.parameters()).is_cuda
    
    # We no longer test normalized vs non-normalized versions since we've
    # standardized on using scaled dot product attention without normalization

def test_embedding_similarities(language_model):
    """Test similarities between different embedding types."""
    query_text = "What is artificial intelligence?"
    
    # Verify model is on CUDA before computing embeddings
    assert language_model.device == "cuda"
    assert next(language_model.model.parameters()).is_cuda
    
    # Tokenize the text first
    tokens = language_model.tokenizer(query_text, return_tensors="pt")["input_ids"]
    
    # Get embeddings directly from tokens
    key_embedding = compute_attention_key_embeddings(tokens, model=language_model)
    query_embedding = compute_attention_query_embeddings(tokens, model=language_model)
    
    # Move to CPU for easier calculations
    key_embedding = key_embedding.cpu()
    query_embedding = query_embedding.cpu()
    
    # Check that embeddings have reasonable dimensions
    assert key_embedding.shape[0] > 0
    assert query_embedding.shape[0] > 0
    
    # Note: We no longer expect embeddings to have the same dimensionality
    # Key and query embeddings might be from attention projections with different dimensions
    # Their dimensions are implementation-dependent
    
    # We can't directly compare key and query embeddings if they have different dimensions
    if key_embedding.shape == query_embedding.shape:
        # Only compute similarity if dimensions match
        # Flatten the embeddings to 1D tensors
        key_embedding_flat = key_embedding.view(-1)
        query_embedding_flat = query_embedding.view(-1)
        
        # Normalize for cosine similarity
        key_norm = torch.norm(key_embedding_flat)
        query_norm = torch.norm(query_embedding_flat)
        
        if key_norm > 0 and query_norm > 0:
            key_normalized = key_embedding_flat / key_norm
            query_normalized = query_embedding_flat / query_norm
            
            # Compute cosine similarity
            key_query_sim = torch.dot(key_normalized, query_normalized).item()
            assert -1.0 <= key_query_sim <= 1.0 