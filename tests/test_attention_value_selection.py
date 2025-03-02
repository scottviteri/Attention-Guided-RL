"""
Tests for attention-based value selection functions.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import math

from src.model import LanguageModel
from src.embeddings import (
    compute_similarity,
    extract_attention_activations,
    compute_multihead_similarity
)
from src.data import KeyValuePair
from src.rl_agent import select_value_with_attention

@pytest.fixture(scope="module")
def language_model(language_model_cuda):
    """Fixture to provide a language model for tests."""
    model = language_model_cuda
    # Verify the model is on CUDA if available
    if torch.cuda.is_available():
        assert next(model.model.parameters()).is_cuda, "Model should be on CUDA"
        print(f"Model is on device: {model.device}")
    return model

def test_extract_attention_activations(language_model):
    """Test extracting attention activations."""
    # Create test input
    test_text = "This is a test input"
    inputs = language_model.tokenizer(test_text, return_tensors="pt")
    
    # Move inputs to device
    inputs = {k: v.to(language_model.device) for k, v in inputs.items()}
    
    # Extract key activations
    key_activations = extract_attention_activations(
        inputs["input_ids"], 
        language_model, 
        activation_type="key"
    )
    
    # Check basic properties
    assert isinstance(key_activations, torch.Tensor)
    assert key_activations.ndim >= 2  # At least 2D tensor (batch, embedding)
    assert key_activations.shape[0] == 1  # Batch size
    
    # Verify activations are on the correct device
    if torch.cuda.is_available():
        assert key_activations.is_cuda, "Key activations should be on CUDA"
        # Print GPU memory after key activations
        print(f"GPU memory after key activations: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

@pytest.fixture
def key_value_pairs(language_model):
    """Create sample key-value pairs for testing."""
    # Get the device from the language model
    device = language_model.device
    
    # Create 5 pairs with mock tensors
    pairs = []
    for i in range(5):
        key_tokens = torch.ones(1, 10, device=device) * i  # Simple distinguishable pattern
        value_tokens = torch.ones(1, 15, device=device) * (i + 10)  # Different pattern for values
        key_embedding = torch.ones(96, device=device) * (i + 5)  # Mock embedding
        
        # Move to CPU for storage in numpy array
        key_embedding_cpu = key_embedding.cpu()
        
        pair = KeyValuePair(
            key_tokens=key_tokens,
            value_tokens=value_tokens,
            key_embedding=key_embedding_cpu.numpy(),  # Convert to numpy array
            key_id=f"key_{i}"
        )
        pairs.append(pair)
    return pairs

def test_compute_similarity():
    """Test the compute_similarity function."""
    # Create a query embedding with batch dimension
    query_embedding = torch.tensor([[0.5, 0.5, 0.7071]])  # Add batch dimension
    
    # Create key embeddings with batch dimension
    key_embeddings = torch.stack([
        torch.tensor([1.0, 0.0, 0.0]),  # key1
        torch.tensor([0.0, 1.0, 0.0]),  # key2
        torch.tensor([0.0, 0.0, 2.0])   # key3
    ])
    key_embeddings = key_embeddings.unsqueeze(0)  # Add batch dimension
    
    # Compute similarities
    similarities = compute_similarity(query_embedding, key_embeddings)
    
    # Check results
    assert similarities.shape == (1, 3)  # Batch size 1, 3 keys
    assert abs(similarities[0, 0].item() - 0.2887) < 0.01
    assert abs(similarities[0, 1].item() - 0.2887) < 0.01
    assert abs(similarities[0, 2].item() - 0.8165) < 0.01

def test_select_value_with_attention(language_model, key_value_pairs):
    """Test selecting a value using attention-based selection."""
    # Verify CUDA is being used if available
    if torch.cuda.is_available():
        print(f"CUDA is available for select_value_with_attention test, using device: {language_model.device}")
        # Print initial GPU memory usage
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
    # Create a mock query
    query = "What is machine learning?"
    
    # Create a mock language model with controlled similarity scores
    mock_model = MagicMock(spec=LanguageModel)
    # Use the same device as the real language model instead of hardcoding CPU
    mock_model.device = language_model.device
    mock_model.tokenizer = language_model.tokenizer
    
    # Print the device we're using
    print(f"Mock model device: {mock_model.device}")
    
    # Mock the extract_attention_activations function to return a fixed embedding
    # Create tensor on the same device as the model
    query_embedding = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float, device=mock_model.device)
    
    # Verify our tensor is on the correct device
    if torch.cuda.is_available():
        assert query_embedding.is_cuda, "Query embedding should be on CUDA"
    
    with patch('src.rl_agent.extract_attention_activations', return_value=query_embedding), \
         patch('src.rl_agent.compute_similarity') as mock_similarity, \
         patch('src.rl_agent.select_key') as mock_select_key:
        
        # Set up similarity scores to prefer the 3rd key-value pair
        # Return a tensor of similarity scores with batch dimension on same device
        mock_scores = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], device=mock_model.device)
        
        # Verify our mock scores are on the correct device
        if torch.cuda.is_available():
            assert mock_scores.is_cuda, "Mock scores should be on CUDA"
            
        mock_similarity.return_value = mock_scores
        
        # Mock the select_key function to return predictable results
        mock_select_key.return_value = ("Selected Key", "Selected Value", key_value_pairs[:4])
        
        # Mock softmax and sample functions
        with patch('src.rl_agent.softmax') as mock_softmax, \
             patch('src.rl_agent.sample_from_distribution') as mock_sample:
                
            # Set up the mocks
            mock_softmax.return_value = np.array([0.1, 0.2, 0.3, 0.1, 0.3])
            mock_sample.return_value = 4  # Select the last item
            
            # Call the function
            key, value, remaining_db = select_value_with_attention(
                query=query,
                database=key_value_pairs,
                model=mock_model,
                temperature=1.0,
                verbose=False,
                use_multihead=False  # Use standard similarity for easier testing
            )
            
            # Verify the result
            assert key == "Selected Key"
            assert value == "Selected Value"
            assert len(remaining_db) == 4
            
            # Verify the mocks were called correctly
            mock_similarity.assert_called_once()
            mock_select_key.assert_called_once_with(key_value_pairs, "key_4", mock_model.tokenizer)
            mock_softmax.assert_called_once()
            mock_sample.assert_called_once()
            
            # Force some computation on GPU to make sure it's being used
            if torch.cuda.is_available():
                # Create a large tensor and do a simple operation to trigger GPU memory usage
                large_tensor = torch.rand(3000, 3000, device=mock_model.device)
                result = large_tensor @ large_tensor
                # Print GPU memory after large tensor operation
                print(f"GPU memory after large tensor operation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

def test_select_value_with_multihead_attention(language_model, key_value_pairs):
    """Test selecting a value using multi-head attention-based selection."""
    # Verify CUDA is being used if available
    if torch.cuda.is_available():
        print(f"CUDA is available for multihead test, using device: {language_model.device}")
        # Print initial GPU memory usage
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
    # Create a mock query
    query = "What is machine learning?"
    
    # Create a mock language model with controlled similarity scores
    mock_model = MagicMock(spec=LanguageModel)
    # Use the same device as the real language model instead of hardcoding CPU
    mock_model.device = language_model.device
    mock_model.tokenizer = language_model.tokenizer
    
    # Print the device we're using
    print(f"Mock model device for multihead test: {mock_model.device}")
    
    # Mock the extract_attention_activations function to return a fixed embedding
    # Create tensor on the same device as the model
    query_embedding = torch.tensor([[1.0, 2.0, 3.0] * 32], dtype=torch.float, device=mock_model.device)
    
    # Verify our tensor is on the correct device
    if torch.cuda.is_available():
        assert query_embedding.is_cuda, "Query embedding should be on CUDA"
    
    with patch('src.rl_agent.extract_attention_activations', return_value=query_embedding), \
         patch('src.rl_agent.compute_multihead_similarity') as mock_multihead_similarity, \
         patch('src.rl_agent.select_key') as mock_select_key:
        
        # Set up similarity scores to prefer the 3rd key-value pair with batch dimensions
        device = mock_model.device
        
        # avg_similarities: [batch_size, num_keys] on same device as model
        avg_similarities = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], device=device)
        
        # Verify our tensor is on the correct device
        if torch.cuda.is_available():
            assert avg_similarities.is_cuda, "Average similarities should be on CUDA"
        
        mock_multihead_similarity.return_value = avg_similarities
        
        # Mock the select_key function to return predictable results
        mock_select_key.return_value = ("Selected Key", "Selected Value", key_value_pairs[:4])
        
        # Mock softmax and sample functions
        with patch('src.rl_agent.softmax') as mock_softmax, \
             patch('src.rl_agent.sample_from_distribution') as mock_sample:
                
            # Set up the mocks
            mock_softmax.return_value = np.array([0.1, 0.2, 0.1, 0.1, 0.5])
            mock_sample.return_value = 4  # Select the last item
            
            # Call the function
            key, value, remaining_db = select_value_with_attention(
                query=query,
                database=key_value_pairs,
                model=mock_model,
                temperature=1.0,
                verbose=False,
                use_multihead=True  # Use multi-head similarity
            )
            
            # Verify the result
            assert key == "Selected Key"
            assert value == "Selected Value"
            assert len(remaining_db) == 4
            
            # Verify the mocks were called correctly
            mock_multihead_similarity.assert_called_once()
            mock_select_key.assert_called_once_with(key_value_pairs, "key_4", mock_model.tokenizer)
            mock_softmax.assert_called_once()
            mock_sample.assert_called_once()
            
            # Force some computation on GPU to make sure it's being used
            if torch.cuda.is_available():
                # Create a large tensor and do a simple operation to trigger GPU memory usage
                large_tensor = torch.rand(4000, 4000, device=device)
                result = large_tensor @ large_tensor
                # Print GPU memory after large tensor operation
                print(f"GPU memory after large tensor operation in multihead test: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

def test_compute_multihead_similarity():
    """Test the compute_multihead_similarity function with grouped query attention."""
    # Create query embedding with batch dimension
    query_embedding = torch.ones(1, 96)  # [1, 96] - Add batch dimension
    
    # Create key embeddings with batch dimension
    key_embeddings = torch.stack([
        torch.ones(32),      
        torch.ones(32) * 0.5,
        torch.ones(32) * 0.25
    ])
    key_embeddings = key_embeddings.unsqueeze(0)  # [1, 3, 32] - Add batch dimension
    
    # Mock model
    mock_model = MagicMock()
    mock_model.model = MagicMock()
    mock_model.model.config = MagicMock()
    mock_model.model.config.num_attention_heads = 24
    mock_model.model.config.num_key_value_heads = 8
    
    # Compute similarities
    similarities = compute_multihead_similarity(
        query_embedding=query_embedding,
        key_embeddings=key_embeddings,
        model=mock_model
    )
    
    # Check result
    assert similarities.shape == (1, 3)  # Batch size 1, 3 keys

def compute_multihead_similarity(
    query_embedding: torch.Tensor,
    key_embeddings: torch.Tensor,
    model: LanguageModel,
    verbose: bool = False
) -> torch.Tensor:
    """
    Compute similarity between query embedding and key embeddings using multi-head attention.
    
    Args:
        query_embedding: Query embedding tensor [batch_size, q_embed_dim]
        key_embeddings: Key embeddings tensor [batch_size, num_keys, k_embed_dim]
        model: LanguageModel to extract head configuration
        verbose: Whether to print debug information
        
    Returns:
        Tensor of average similarity scores [batch_size, num_keys]
    """
    # Ensure query_embedding has a batch dimension
    if query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)  # Add batch dimension if missing
    
    batch_size, num_keys = key_embeddings.shape[0], key_embeddings.shape[1]
    assert num_keys > 0, "key_embeddings cannot be empty"
    
    # Ensure inputs are on the same device
    device = query_embedding.device
    key_embeddings = key_embeddings.to(device)
    
    # Extract model parameters
    llama_config = model.model.config
    num_heads = getattr(llama_config, 'num_attention_heads', getattr(llama_config, 'num_heads', 24))
    num_kv_groups = getattr(llama_config, 'num_key_value_heads', getattr(llama_config, 'num_kv_heads', 8))
    
    if verbose:
        logger.info(f"Using num_heads={num_heads}, num_kv_groups={num_kv_groups}")
    
    # Calculate head dimensions
    q_embedding_dim = query_embedding.shape[1]
    k_embedding_dim = key_embeddings.shape[2]
    head_dim = min(q_embedding_dim // num_heads, k_embedding_dim // num_kv_groups)
    
    # Calculate heads per group ratio for grouped query attention
    heads_per_group = num_heads // num_kv_groups
    
    # Reshape for multi-head processing
    query_heads = query_embedding.view(batch_size, num_heads, head_dim)
    key_heads = key_embeddings.view(batch_size, num_keys, num_kv_groups, head_dim)
    
    # Initialize output tensor
    per_head_similarities = torch.zeros((batch_size, num_heads, num_keys), device=device)
    
    # Vectorize this part if possible in future improvements
    for b in range(batch_size):
        for h in range(num_heads):
            # Determine which KV group this head should attend to
            kv_group_idx = h // heads_per_group
            
            # Get the query head and corresponding key heads
            q_head = query_heads[b, h]
            k_heads = key_heads[b, :, kv_group_idx]
            
            # Compute dot product and scale
            per_head_similarities[b, h] = torch.matmul(q_head, k_heads.transpose(0, 1)) / math.sqrt(head_dim)
    
    # Average across heads
    average_similarities = per_head_similarities.mean(dim=1)
    
    return average_similarities 