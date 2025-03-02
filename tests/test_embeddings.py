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
    get_embeddings,
    compute_similarity,
    extract_attention_activations,
    compute_multihead_similarity
)
from src.model import LanguageModel

@pytest.fixture(scope="module")
def language_model(language_model_cpu):
    """Fixture to provide a language model for tests."""
    # Use the shared session fixture from conftest.py
    return language_model_cpu

@patch('src.embeddings.extract_attention_activations')
def test_get_embeddings(mock_extract_activations, language_model):
    """Test the get_embeddings function."""
    # Setup mock return values
    mock_extract_activations.return_value = torch.tensor([
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],  # Batch item 1
        [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]   # Batch item 2
    ])
    
    # Mock tokenizer
    language_model.tokenizer = MagicMock()
    language_model.tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])  # All ones for valid tokens
    }
    
    # Call function with the pre-loaded model and input_ids only
    token_dict = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])
    }
    
    embeddings = get_embeddings(
        tokens=token_dict["input_ids"],
        model=language_model,
        are_queries=True
    )

    # Verify results
    assert embeddings.shape[0] == 2  # batch size
    assert embeddings.shape[1] > 0  # embedding dimension

def test_compute_similarity():
    """Test the compute_similarity function."""
    # Create a query embedding with batch dimension
    query_embedding = torch.tensor([[0.5, 0.5, 0.7071]])  # Add batch dimension
    
    # Create key embeddings with batch dimension
    key_embeddings = torch.stack([
        torch.tensor([1.0, 0.0, 0.0]),  # key1 - Normalized
        torch.tensor([0.0, 1.0, 0.0]),  # key2 - Normalized
        torch.tensor([0.0, 0.0, 2.0])   # key3 - Not normalized
    ]).unsqueeze(0)  # Add batch dimension [1, 3, 3]
    
    # Compute similarities
    similarities = compute_similarity(query_embedding, key_embeddings)
    
    # Check results - using the actual formula: dot(query, key) / sqrt(d)
    # For these vectors with d=3:
    # query · key1 = (0.5*1.0 + 0.5*0.0 + 0.7071*0.0) / sqrt(3) ≈ 0.5 / 1.732 ≈ 0.2887
    # query · key2 = (0.5*0.0 + 0.5*1.0 + 0.7071*0.0) / sqrt(3) ≈ 0.5 / 1.732 ≈ 0.2887
    # query · key3 = (0.5*0.0 + 0.5*0.0 + 0.7071*2.0) / sqrt(3) ≈ 1.4142 / 1.732 ≈ 0.8165
    
    # Check dimensions
    assert similarities.shape == (1, 3)  # batch_size=1, num_keys=3
    
    # Check values
    assert abs(similarities[0, 0].item() - 0.2887) < 0.01
    assert abs(similarities[0, 1].item() - 0.2887) < 0.01
    assert abs(similarities[0, 2].item() - 0.8165) < 0.01
    
    # We can't test with empty key_embeddings anymore since we assert it's not empty
    with pytest.raises(AssertionError):
        compute_similarity(query_embedding, torch.zeros((1, 0, 3)))

def test_extract_attention_activations():
    """Test the extract_attention_activations function."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.device = "cpu"
    
    # Create a mock layer that will be targeted by the hook
    mock_q_proj = MagicMock()
    mock_q_proj.register_forward_hook.return_value = MagicMock()
    
    # Set up the model structure
    mock_model.model = MagicMock()
    mock_model.model.model = MagicMock()
    mock_model.model.model.layers = [MagicMock()]
    mock_model.model.model.layers[-1].self_attn = MagicMock()
    mock_model.model.model.layers[-1].self_attn.q_proj = mock_q_proj
    mock_model.model.model.layers[-1].self_attn.k_proj = mock_q_proj  # Reuse same mock
    
    # Set up the expected output tensor
    expected_output = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
    
    # Mock the forward hook to set the activations
    def side_effect(*args, **kwargs):
        # Set the output for the model call - doesn't matter what it is
        return MagicMock()
    
    # Configure the hook registration to capture the hook function
    def register_hook_side_effect(hook_fn):
        # Call the hook function with dummy inputs to set activations
        hook_fn(None, None, expected_output)
        return MagicMock()  # Return a removable hook
    
    mock_q_proj.register_forward_hook.side_effect = register_hook_side_effect
    mock_model.model.side_effect = side_effect
    
    # Call the function
    tokens = torch.tensor([[1, 2]])
    result = extract_attention_activations(tokens, mock_model, activation_type="query")
    
    # Verify the result matches the expected output
    assert torch.equal(result, expected_output)
    
    # Verify the hook was registered and removed
    mock_q_proj.register_forward_hook.assert_called_once()

def test_compute_multihead_similarity():
    """Test the compute_multihead_similarity function."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.model = MagicMock()
    mock_model.model.config = MagicMock()
    mock_model.model.config.num_attention_heads = 12
    mock_model.model.config.num_key_value_heads = 4  # Using grouped attention
    
    batch_size = 2
    num_heads = 12
    head_dim = 64
    num_kv_groups = 4
    
    # Create query embeddings with batch dimension
    # [batch_size, num_heads * head_dim]
    query_embedding = torch.randn(batch_size, num_heads * head_dim)
    
    # Create key embeddings with batch dimension
    # [batch_size, num_keys, num_kv_groups * head_dim]
    num_keys = 3
    key_embeddings = torch.randn(batch_size, num_keys, num_kv_groups * head_dim)
    
    # Call the function
    avg_similarities = compute_multihead_similarity(
        query_embedding=query_embedding,
        key_embeddings=key_embeddings,
        model=mock_model
    )
    
    # Check dimensions
    assert avg_similarities.shape == (batch_size, num_keys)  # [batch_size, num_keys] 