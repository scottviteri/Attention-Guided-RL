"""
Functions for computing embeddings using transformer models.
"""
import logging
import torch
import math
from typing import Dict, List, Union, Optional, Tuple
from torchtyping import TensorType

from src.model import LanguageModel

# Set up logging
logger = logging.getLogger(__name__)

def compute_similarity(
    query_embedding: TensorType["batch_size", "embed_dim"],
    key_embeddings: TensorType["batch_size", "num_keys", "embed_dim"]
) -> TensorType["batch_size", "num_keys"]:
    """
    Compute similarity between query embedding and multiple key embeddings using scaled dot product.
    Supports batched processing.
    
    Args:
        query_embedding: Query embedding tensor [batch_size, embed_dim]
        key_embeddings: Key embeddings tensor [batch_size, num_keys, embed_dim]
    
    Returns:
        Tensor of similarity scores [batch_size, num_keys]
    """
    batch_size, num_keys = key_embeddings.shape[0], key_embeddings.shape[1]
    assert num_keys > 0, "key_embeddings cannot be empty"
    
    # Get embedding dimension for scaling
    dim = query_embedding.shape[1]
    scaling_factor = torch.sqrt(torch.tensor(dim, dtype=torch.float32))
    
    # Initialize output tensor
    similarities = torch.zeros((batch_size, num_keys), dtype=torch.float32, device=query_embedding.device)
    
    # Process each batch item
    for b in range(batch_size):
        # [embed_dim] · [num_keys, embed_dim]ᵀ = [num_keys]
        batch_query = query_embedding[b]  # [embed_dim]
        batch_keys = key_embeddings[b]    # [num_keys, embed_dim]
        similarities[b] = torch.matmul(batch_query, batch_keys.t()) / scaling_factor
    
    return similarities

def compute_attention_key_embeddings(
    tokens: TensorType["batch_size", "seq_len"], 
    model: LanguageModel
) -> TensorType["batch_size", "embed_dim"]:
    """
    Compute key embeddings from attention mechanism for a batch of tokens.
    
    Args:
        tokens: Input token IDs [batch_size, seq_len]
        model: LanguageModel instance
        
    Returns:
        Tensor containing key embeddings [batch_size, embed_dim]
    """
    assert tokens.size(0) > 0, "Received empty input batch"
    
    # Extract key activations using hooks
    activations = extract_attention_activations(tokens, model, activation_type="key")
    
    # Since all tokens are valid, we can simply average across the sequence dimension
    embeddings = activations.mean(dim=1)
    
    return embeddings

def compute_attention_query_embeddings(
    tokens: TensorType["batch_size", "seq_len"], 
    model: LanguageModel
) -> TensorType["batch_size", "embed_dim"]:
    """
    Compute query embeddings from attention mechanism for a batch of tokens.
    
    Args:
        tokens: Input token IDs [batch_size, seq_len]
        model: LanguageModel instance
        
    Returns:
        Tensor containing query embeddings [batch_size, embed_dim]
    """
    assert tokens.size(0) > 0, "Received empty input batch"
    
    # Extract query activations using hooks
    activations = extract_attention_activations(tokens, model, activation_type="query")
    
    # Since all tokens are valid, we can simply average across the sequence dimension
    embeddings = activations.mean(dim=1)
    
    return embeddings

def extract_attention_activations(
    tokens: TensorType["batch_size", "seq_len"],
    model: LanguageModel,
    activation_type: str = "query"
) -> TensorType["batch_size", "seq_len", "embed_dim"]:
    """
    Extract query or key activations from the model using hooks.
    
    Args:
        tokens: Input token IDs [batch_size, seq_len]
        model: LanguageModel instance
        activation_type: Type of activation to extract ("query" or "key")
        
    Returns:
        Activations tensor [batch_size, seq_len, embed_dim]
    """
    # Make sure the tokens are on the same device as the model
    device = model.device
    tokens = tokens.to(device)
    
    # Define hook to capture activations
    activations = None
    
    def hook_fn(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    # Place hook on the appropriate layer
    if activation_type == "query":
        # Target the query projection in the last attention layer
        target_module = model.model.model.layers[-1].self_attn.q_proj
    elif activation_type == "key":
        # Target the key projection in the last attention layer
        target_module = model.model.model.layers[-1].self_attn.k_proj
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")
    
    # Register the hook
    hook = target_module.register_forward_hook(hook_fn)
    
    # Run forward pass to trigger hook
    with torch.no_grad():
        # Create attention mask (all 1s since we're asserting there's no padding)
        attention_mask = torch.ones_like(tokens, device=device)
        inputs = {
            "input_ids": tokens,
            "attention_mask": attention_mask
        }
        model.model(**inputs)
    
    # Remove the hook
    hook.remove()
    
    # Verify activations are on the correct device (should be by default since model is on that device)
    if activations is not None and activations.device != device:
        logger.warning(f"Activations were on {activations.device}, moving to {device}")
        activations = activations.to(device)
    
    return activations

def get_embeddings(
    tokens: TensorType["batch_size", "seq_len"],
    model: LanguageModel,
    are_queries: Union[bool, List[bool]] = True
) -> TensorType["batch_size", "embed_dim"]:
    """
    Get embeddings for batched token inputs based on whether they are queries or keys.
    
    Args:
        tokens: Tensor of token IDs [batch_size, seq_len]
        model: LanguageModel instance
        are_queries: Whether tokens represent queries (True) or content (False)
                    Can be a single boolean or a list matching batch size
        
    Returns:
        Tensor of embeddings [batch_size, embed_dim]
    """
    # Handle empty input case
    assert tokens.size(0) > 0, "tokens tensor cannot be empty"
    
    # Check if all tokens are queries or all are keys
    all_queries = isinstance(are_queries, bool) and are_queries
    all_keys = isinstance(are_queries, bool) and not are_queries
    
    # Compute embeddings based on token type
    if all_queries:
        # All tokens are queries - use query attention mechanism
        activations = extract_attention_activations(tokens, model, activation_type="query")
        embeddings = activations.mean(dim=1)  # Simple mean across sequence dimension
    elif all_keys:
        # All tokens are keys - use key attention mechanism
        activations = extract_attention_activations(tokens, model, activation_type="key")
        embeddings = activations.mean(dim=1)  # Simple mean across sequence dimension
    else:
        # Mixed tokens - handle separately
        raise NotImplementedError("Mixed query/key batches not yet supported")
    
    return embeddings

def compute_multihead_similarity(
    query_embedding: TensorType["batch_size", "q_embed_dim"],
    key_embeddings: TensorType["batch_size", "num_keys", "k_embed_dim"],
    model: LanguageModel,
    verbose: bool = False
) -> TensorType["batch_size", "num_keys"]:
    """
    Compute similarity between query embedding and key embeddings using multi-head attention.
    Supports batched processing.
    
    Args:
        query_embedding: Query embedding tensor [batch_size, q_embed_dim]
        key_embeddings: Key embeddings tensor [batch_size, num_keys, k_embed_dim]
        model: LanguageModel to extract head configuration
        verbose: Whether to print debug information
        
    Returns:
        Tensor of average similarity scores [batch_size, num_keys]
    """
    batch_size, num_keys = key_embeddings.shape[0], key_embeddings.shape[1]
    assert num_keys > 0, "key_embeddings cannot be empty"
    
    # Ensure inputs are on the same device
    device = query_embedding.device
    if key_embeddings.device != device:
        if verbose:
            logger.warning(f"Moving key_embeddings from {key_embeddings.device} to {device}")
        key_embeddings = key_embeddings.to(device)
    
    # Extract model parameters
    llama_config = model.model.config
    # Try to get num_heads and num_kv_groups from model config
    num_heads = getattr(llama_config, 'num_attention_heads', 
                      getattr(llama_config, 'num_heads', 24))
    
    num_kv_groups = getattr(llama_config, 'num_key_value_heads', 
                          getattr(llama_config, 'num_kv_heads', 8))
    
    if verbose:
        logger.info(f"Using num_heads={num_heads}, num_kv_groups={num_kv_groups}")
    
    # Calculate head dimension
    q_embedding_dim = query_embedding.shape[1]
    head_dim = q_embedding_dim // num_heads
    
    assert q_embedding_dim % num_heads == 0, f"Query embedding dimension {q_embedding_dim} not divisible by num_heads {num_heads}"
    
    # Determine key embedding dimension
    k_embedding_dim = key_embeddings.shape[2]
    key_head_dim = k_embedding_dim // num_kv_groups
    
    assert k_embedding_dim % num_kv_groups == 0, f"Key embedding dimension {k_embedding_dim} not divisible by num_kv_groups {num_kv_groups}"
    
    if head_dim != key_head_dim:
        if verbose:
            logger.warning(f"Head dimensions don't match: query={head_dim}, key={key_head_dim}. Using min value.")
        head_dim = min(head_dim, key_head_dim)
    
    # Calculate heads per group ratio (how many query heads attend to each KV group)
    heads_per_group = num_heads // num_kv_groups
    
    if verbose:
        logger.info(f"Query embedding dim: {q_embedding_dim}, Key embedding dim: {k_embedding_dim}")
        logger.info(f"Heads: {num_heads}, KV groups: {num_kv_groups}, Heads per group: {heads_per_group}")
        logger.info(f"Head dim: {head_dim}, Key head dim: {key_head_dim}")
    
    # Reshape query embedding into heads for each item in batch
    # [batch_size, q_embed_dim] -> [batch_size, num_heads, head_dim]
    query_heads = query_embedding.reshape(batch_size, num_heads, head_dim)
    
    # Reshape key embeddings into groups for all keys
    # [batch_size, num_keys, k_embed_dim] -> [batch_size, num_keys, num_kv_groups, key_head_dim]
    key_groups_all = key_embeddings.reshape(batch_size, num_keys, num_kv_groups, key_head_dim)
    
    # Initialize results tensor on the same device
    # [batch_size, num_heads, num_keys]
    per_head_similarities = torch.zeros((batch_size, num_heads, num_keys), dtype=torch.float, device=device)
    
    # Calculate per-head similarities efficiently using broadcasting
    for batch_idx in range(batch_size):
        for h in range(num_heads):
            # Determine which KV group this head should attend to
            kv_group_idx = h // heads_per_group
            
            # Extract corresponding query head [head_dim]
            query_head = query_heads[batch_idx, h, :head_dim]
            
            # Extract corresponding key groups for all keys [num_keys, head_dim]
            key_groups = key_groups_all[batch_idx, :, kv_group_idx, :head_dim]
            
            # Calculate scaled dot product for all keys at once
            # [head_dim] · [num_keys, head_dim]ᵀ = [num_keys]
            similarities = torch.matmul(query_head, key_groups.transpose(0, 1)) / math.sqrt(head_dim)
            per_head_similarities[batch_idx, h] = similarities
    
    # Calculate average similarity across heads for each batch item
    # [batch_size, num_heads, num_keys] -> [batch_size, num_keys]
    average_similarities = torch.mean(per_head_similarities, dim=1)
    
    return average_similarities 