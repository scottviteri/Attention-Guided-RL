"""
Functions for computing embeddings of text.
"""

import os
import json
import time
import hashlib
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path

from src.config import MODEL_NAME
from src.utils import ensure_dir
from src.model import load_language_model

# Avoid circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from src.model import LanguageModel
else:
    # Forward reference for runtime type hinting
    LanguageModel = "LanguageModel"

# Set up logging
logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = "embedding_cache"

# Functional embedding API

def get_cache_path(text: str, cache_dir: str = DEFAULT_CACHE_DIR) -> Path:
    """
    Get the cache path for a text embedding.
    
    Args:
        text: Text to hash
        cache_dir: Cache directory
        
    Returns:
        Path object for the cache file
    """
    # Create hash of the text for filename
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Ensure cache directory exists
    ensure_dir(cache_dir)
    
    # Return path
    return Path(cache_dir) / f"{text_hash}.json"

def load_from_cache(text: str, cache_dir: str = DEFAULT_CACHE_DIR) -> Optional[np.ndarray]:
    """
    Load embedding from cache if available.
    
    Args:
        text: Text to load embedding for
        cache_dir: Cache directory
        
    Returns:
        Embedding array if found, None otherwise
    """
    cache_path = get_cache_path(text, cache_dir)
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
            return np.array(cache_data["embedding"])
    except Exception as e:
        logger.warning(f"Error loading from cache: {e}")
        return None

def save_to_cache(text: str, embedding: np.ndarray, cache_dir: str = DEFAULT_CACHE_DIR) -> None:
    """
    Save embedding to cache.
    
    Args:
        text: Text the embedding is for
        embedding: The embedding to save
        cache_dir: Cache directory
    """
    cache_path = get_cache_path(text, cache_dir)
    
    try:
        with open(cache_path, 'w') as f:
            json.dump({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "embedding": embedding.tolist(),
                "timestamp": time.time()
            }, f)
    except Exception as e:
        logger.warning(f"Error saving to cache: {e}")

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize embedding to unit length.
    
    Args:
        embedding: Embedding vector
        
    Returns:
        Normalized embedding vector
    """
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding

def tokenize_text(text: str, tokenizer, max_length: int = 512, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Tokenize text and prepare for model input.
    
    Args:
        text: Text to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum token length
        device: Device for tensors
        
    Returns:
        Dictionary with input_ids and attention_mask
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    
    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device)
    }

def tokenize_text_batch(texts: List[str], tokenizer, max_length: int = 512, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Tokenize multiple texts in a batch and prepare for model input.
    
    Args:
        texts: List of texts to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum token length
        device: Device for tensors
        
    Returns:
        Dictionary with input_ids and attention_mask
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    
    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device)
    }

def extract_hidden_states(inputs: Dict[str, torch.Tensor], model: 'LanguageModel') -> List[torch.Tensor]:
    """
    Extract hidden states from model layers.
    
    Args:
        inputs: Tokenized inputs
        model: Language model
        
    Returns:
        List of hidden state tensors from each layer
    """
    with torch.no_grad():
        outputs = model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )
    
    return outputs.hidden_states

def process_hidden_states(hidden_states: List[torch.Tensor], attention_mask: torch.Tensor) -> np.ndarray:
    """
    Process hidden states to create embeddings.
    
    Args:
        hidden_states: List of hidden states
        attention_mask: Attention mask tensor
        
    Returns:
        Embedding array
    """
    # Stack and average hidden states from all layers
    all_layers = torch.stack(hidden_states)
    avg_hidden_states = torch.mean(all_layers, dim=0)
    
    # Apply attention mask and average over tokens
    expanded_mask = attention_mask.unsqueeze(-1).expand(avg_hidden_states.size())
    masked_hidden = avg_hidden_states * expanded_mask
    sum_hidden = torch.sum(masked_hidden, dim=1)
    token_count = torch.sum(attention_mask, dim=1, keepdim=True)
    mean_hidden = sum_hidden / token_count
    
    # Convert to numpy and return
    return mean_hidden.cpu().numpy()[0]  # Return first (and only) item

def process_hidden_states_batch(hidden_states: List[torch.Tensor], attention_mask: torch.Tensor) -> np.ndarray:
    """
    Process hidden states to create embeddings for a batch.
    
    Args:
        hidden_states: List of hidden states
        attention_mask: Attention mask tensor
        
    Returns:
        Embedding array for each item in the batch
    """
    # Stack and average hidden states from all layers
    all_layers = torch.stack(hidden_states)
    avg_hidden_states = torch.mean(all_layers, dim=0)
    
    # Apply attention mask and average over tokens
    expanded_mask = attention_mask.unsqueeze(-1).expand(avg_hidden_states.size())
    masked_hidden = avg_hidden_states * expanded_mask
    sum_hidden = torch.sum(masked_hidden, dim=1)
    token_count = torch.sum(attention_mask, dim=1, keepdim=True)
    mean_hidden = sum_hidden / token_count
    
    # Convert to numpy and return all batch embeddings
    return mean_hidden.cpu().numpy()

def get_embedding(
    text: str,
    model: LanguageModel,
    is_query: bool = True,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    max_length: int = 512
) -> np.ndarray:
    """
    Get embedding for a single text string, with optional caching.
    
    Args:
        text: Text to embed
        model: Language model
        is_query: Whether this is a query (True) or a key (False)
        use_cache: Whether to use cache
        cache_dir: Directory for cache
        max_length: Maximum token length
        
    Returns:
        Embedding array
    """
    if use_cache:
        # Try to load from cache first
        cached = load_from_cache(text, cache_dir)
        if cached is not None:
            return cached
            
    # Compute embedding based on type
    if is_query:
        embedding = compute_attention_query_embedding(text, model, max_length)
    else:
        # For keys, use the standard embedding
        embedding = compute_llama_embedding(text, model, max_length)
        
    # Save to cache if enabled
    if use_cache:
        save_to_cache(text, embedding, cache_dir)
        
    return embedding

def compute_embeddings_batch(texts: List[str], model: LanguageModel, max_length: int = 512) -> List[np.ndarray]:
    """
    Compute embeddings for multiple texts in a single batch.
    
    Args:
        texts: List of texts to embed
        model: Language model
        max_length: Maximum token length
        
    Returns:
        List of embedding arrays
    """
    if not texts:
        return []
    
    # Tokenize all texts in a batch
    inputs = tokenize_text_batch(texts, model.tokenizer, max_length, model.device)
    
    # Extract hidden states in a single forward pass
    with torch.no_grad():
        outputs = model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )
    
    hidden_states = outputs.hidden_states
    
    # Process hidden states for the batch
    batch_embeddings = process_hidden_states_batch(hidden_states, inputs["attention_mask"])
    
    # Convert to list of numpy arrays
    return [batch_embeddings[i] for i in range(len(texts))]

def get_embeddings(
    texts: List[str],
    model: LanguageModel,
    are_queries: Union[bool, List[bool]] = True,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    max_length: int = 512,
    batch_size: int = 16
) -> List[np.ndarray]:
    """
    Get embeddings for multiple texts with proper batching.
    
    Args:
        texts: List of texts to embed
        model: Language model
        are_queries: Whether texts are queries (vs keys) - either a single bool or list of bools per text
        use_cache: Whether to use cache
        cache_dir: Cache directory
        max_length: Maximum token length
        batch_size: Size of batches to process at once
        
    Returns:
        List of embedding arrays
    """
    if not texts:
        return []
    
    # Handle single boolean for all texts
    if isinstance(are_queries, bool):
        are_queries = [are_queries] * len(texts)
    
    results = []
    to_compute = []
    to_compute_indices = []
    
    # First check cache for all texts if cache is enabled
    if use_cache:
        for i, text in enumerate(texts):
            if not text:
                results.append(np.array([]))
                continue
            
            # Determine cache subdirectory based on text type
            is_query = are_queries[i]
            text_type_dir = os.path.join(cache_dir, "queries" if is_query else "keys")
            
            # Check cache
            cached_embedding = load_from_cache(text, text_type_dir)
            if cached_embedding is not None:
                results.append(cached_embedding)
            else:
                # Add to list for batch computation
                to_compute.append(text)
                to_compute_indices.append(i)
    else:
        # If cache is disabled, compute all embeddings
        to_compute = texts
        to_compute_indices = list(range(len(texts)))
        results = [None] * len(texts)  # Placeholder list
    
    # Compute embeddings in batches if needed
    if to_compute:
        for i in range(0, len(to_compute), batch_size):
            batch_texts = to_compute[i:i+batch_size]
            
            # Compute batch embeddings
            batch_embeddings = compute_embeddings_batch(batch_texts, model, max_length)
            
            # Store results and optionally cache
            for j, embedding in enumerate(batch_embeddings):
                original_idx = to_compute_indices[i+j]
                
                # Save to cache if enabled
                if use_cache:
                    text = to_compute[i+j]
                    is_query = are_queries[original_idx]
                    text_type_dir = os.path.join(cache_dir, "queries" if is_query else "keys")
                    save_to_cache(text, embedding, text_type_dir)
                
                # Store in results
                if use_cache:
                    results.append(embedding)
                else:
                    results[original_idx] = embedding
    
    return results

def compute_embeddings(
    texts: List[str], 
    model: LanguageModel,
    are_queries: Union[bool, List[bool]] = True,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    max_length: int = 512,
    batch_size: int = 16
) -> List[np.ndarray]:
    """
    Compute embeddings for a list of texts using the language model.
    
    Args:
        texts: List of texts to embed
        model: Language model (required)
        are_queries: Whether the texts are queries (vs keys)
        use_cache: Whether to use embedding cache
        cache_dir: Directory for embedding cache
        max_length: Maximum token length
        batch_size: Size of batches to process at once
        
    Returns:
        List of embeddings as numpy arrays
    """
    # Handle empty input
    if not texts:
        return []
    
    # Use get_embeddings for batched processing
    return get_embeddings(
        texts=texts,
        model=model,
        are_queries=are_queries,
        use_cache=use_cache,
        cache_dir=cache_dir,
        max_length=max_length,
        batch_size=batch_size
    )

def compute_similarity(
    query_embedding: np.ndarray,
    key_embeddings: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compute dot product similarity between query embedding and key embeddings,
    scaled by sqrt(d) where d is the embedding dimension.
    
    Args:
        query_embedding: Query embedding vector (not normalized)
        key_embeddings: Dictionary of key IDs to embedding vectors (not normalized)
        
    Returns:
        Dictionary of key IDs to similarity scores
    """
    # Compute similarity for each key
    similarities = {}
    
    # Flatten query embedding if it's multi-dimensional
    if query_embedding.ndim > 1:
        query_embedding = query_embedding.flatten()
    
    # Get the dimension for scaling
    d = query_embedding.shape[0]
    scaling_factor = np.sqrt(d)
    
    for key_id, key_embedding in key_embeddings.items():
        # Flatten key embedding if it's multi-dimensional
        if key_embedding.ndim > 1:
            key_embedding = key_embedding.flatten()
            
        # Compute scaled dot product (no normalization)
        similarity = np.dot(query_embedding, key_embedding) / scaling_factor
        
        # Convert to float for serialization
        similarities[key_id] = float(similarity)
    
    return similarities

def extract_attention_activations(
    inputs: Dict[str, torch.Tensor],
    model: 'LanguageModel', 
    activation_type: str = "query"
) -> List[torch.Tensor]:
    """
    Extract attention activations (query/key vectors) from model layers.
    
    Args:
        inputs: Tokenized inputs
        model: Language model
        activation_type: Type of activation to extract ('query' or 'key')
        
    Returns:
        List of activation tensors from each layer
    """
    if activation_type not in ["query", "key"]:
        raise ValueError("activation_type must be either 'query' or 'key'")
    
    # Get the underlying Llama model
    llama_model = model.model.model
    
    # Get number of layers
    num_layers = len(llama_model.layers)
    
    # Use only the deeper half of the layers
    start_layer = num_layers // 2
    
    # Prepare position ids for RoPE (rotary positional embeddings)
    seq_length = inputs["input_ids"].shape[1]
    position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs["input_ids"].device)
    position_ids = position_ids.unsqueeze(0).expand_as(inputs["input_ids"])
    
    # Store activations from each layer
    activations = []
    
    with torch.no_grad():
        # Process through model and capture internal activations
        hidden_states = llama_model.embed_tokens(inputs["input_ids"])
        
        # Properly format attention mask for the Llama model
        # Convert from [batch_size, seq_len] to causal 4D format needed by Llama 3.2
        attn_mask = inputs["attention_mask"]
        if attn_mask.dim() == 2:
            # Convert to 4D mask
            extended_attention_mask = attn_mask[:, None, None, :]
            # Create causal mask
            seq_len = inputs["input_ids"].shape[1]
            causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=inputs["input_ids"].device), diagonal=1)
            causal_mask = causal_mask[None, None, :, :]
            # Combine the masks
            combined_attention_mask = torch.logical_or(~extended_attention_mask.bool(), causal_mask)
            # Convert to float mask
            extended_attention_mask = torch.where(combined_attention_mask, torch.finfo(torch.float16).min, 0.0)
        else:
            extended_attention_mask = attn_mask
        
        # Register hooks for the q_proj and k_proj modules
        activation_hooks = []
        activation_outputs = []
        
        def create_hook_fn(layer_idx, proj_type):
            def hook_fn(module, input_tensor, output_tensor):
                if layer_idx >= start_layer:
                    activation_outputs.append(output_tensor)
            return hook_fn
            
        # Register hooks on all layers
        for i, layer in enumerate(llama_model.layers):
            if activation_type == "query":
                hook = layer.self_attn.q_proj.register_forward_hook(create_hook_fn(i, "query"))
            else:
                hook = layer.self_attn.k_proj.register_forward_hook(create_hook_fn(i, "key"))
            activation_hooks.append(hook)
            
        # Forward pass through the model
        outputs = model.model(
            input_ids=inputs["input_ids"],
            attention_mask=attn_mask,
            position_ids=position_ids,
            output_hidden_states=False,
            return_dict=True
        )
        
        # Remove the hooks
        for hook in activation_hooks:
            hook.remove()
        
        # The activations collected via hooks are our result
        activations = activation_outputs
    
    return activations

def process_attention_activations(
    activations: List[torch.Tensor], 
    attention_mask: torch.Tensor
) -> np.ndarray:
    """
    Process attention activations to create embeddings.
    Average over layers, token positions (using attention mask).
    
    Args:
        activations: List of attention activation tensors from multiple layers
        attention_mask: Attention mask tensor
        
    Returns:
        Embedding array for each item in the batch, shape (batch_size, embedding_dim)
    """
    if not activations:
        raise ValueError("No activations were captured. Check that the hook registration worked correctly.")
    
    # Handle shape of the activation outputs from hooks
    processed_activations = []
    
    for activation in activations:
        # If it's the raw output of q_proj or k_proj, we need to reshape it
        # The outputs of our hooks are the direct outputs of the linear projections
        batch_size = activation.shape[0]
        
        # Average over the sequence length dimension to get one vector per batch item
        # First, mask out padded positions
        expanded_mask = attention_mask.unsqueeze(-1)
        while expanded_mask.dim() < activation.dim():
            expanded_mask = expanded_mask.unsqueeze(1)
        expanded_mask = expanded_mask.expand_as(activation)
        
        # Apply mask and get mean
        masked_activation = activation * expanded_mask
        sequence_dim = 1  # Usually the sequence is the second dimension
        token_count = attention_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
        mean_activation = masked_activation.sum(dim=sequence_dim) / token_count
        
        processed_activations.append(mean_activation)
    
    # Stack and average over layers
    if processed_activations:
        all_layers = torch.stack(processed_activations)
        avg_activations = torch.mean(all_layers, dim=0)
    else:
        # If no processed activations, return empty tensor with appropriate shape
        avg_activations = torch.zeros((attention_mask.shape[0], 1), device=attention_mask.device)
    
    # Convert to numpy
    result = avg_activations.cpu().detach().numpy()
    
    # Handle case where result is 3D (reshape to 2D)
    if result.ndim == 3:
        batch_size = result.shape[0]
        # Flatten all dimensions after the batch dimension
        flat_dim = np.prod(result.shape[1:]).astype(int)
        result = result.reshape(batch_size, flat_dim)
    
    return result

def compute_llama_embedding(text: str, model: 'LanguageModel', max_length: int = 512) -> np.ndarray:
    """
    Compute embedding using Llama model's hidden states.
    
    Args:
        text: Text to embed
        model: Language model
        max_length: Maximum token length
        
    Returns:
        Embedding array
    """
    # Tokenize input
    inputs = tokenize_text(text, model.tokenizer, max_length, model.device)
    
    # Get hidden states
    hidden_states = extract_hidden_states(inputs, model)
    
    # Process hidden states
    return process_hidden_states(hidden_states, inputs["attention_mask"])

def compute_attention_key_embedding(text: str, model: 'LanguageModel', max_length: int = 512) -> np.ndarray:
    """
    Compute embedding for an attention key.
    Extracts key projection activations from the model's attention layers.
    
    Args:
        text: Text to embed
        model: Language model
        max_length: Maximum token length
        
    Returns:
        Embedding array
    """
    # Tokenize input
    inputs = tokenize_text(text, model.tokenizer, max_length, model.device)
    
    # Extract key activations
    activations = extract_attention_activations(inputs, model, "key")
    
    # Process activations
    return process_attention_activations(activations, inputs["attention_mask"])

def compute_attention_query_embedding(text: str, model: 'LanguageModel', max_length: int = 512) -> np.ndarray:
    """
    Compute embedding for an attention query.
    Extracts query projection activations from the model's attention layers.
    
    Args:
        text: Text to embed
        model: Language model
        max_length: Maximum token length
        
    Returns:
        Embedding array
    """
    # Tokenize input
    inputs = tokenize_text(text, model.tokenizer, max_length, model.device)
    
    # Extract query activations
    activations = extract_attention_activations(inputs, model, "query")
    
    # Process activations
    embedding = process_attention_activations(activations, inputs["attention_mask"])
        
    return embedding

def extract_attention_activations_batch(
    inputs: Dict[str, torch.Tensor],
    model: 'LanguageModel', 
    activation_type: str = "query"
) -> List[torch.Tensor]:
    """
    Extract attention activations (query/key vectors) from model layers for a batch of inputs.
    
    Args:
        inputs: Tokenized inputs for a batch
        model: Language model
        activation_type: Type of activation to extract ('query' or 'key')
        
    Returns:
        List of activation tensors from each layer
    """
    if activation_type not in ["query", "key"]:
        raise ValueError("activation_type must be either 'query' or 'key'")
    
    # Get the underlying Llama model
    llama_model = model.model.model
    
    # Get number of layers
    num_layers = len(llama_model.layers)
    
    # Use only the deeper half of the layers
    start_layer = num_layers // 2
    
    # Prepare position ids for RoPE (rotary positional embeddings)
    seq_length = inputs["input_ids"].shape[1]
    position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs["input_ids"].device)
    position_ids = position_ids.unsqueeze(0).expand_as(inputs["input_ids"])
    
    # Store activations from each layer
    activations = []
    
    # Register hooks to capture activations
    activation_hooks = []
    activation_outputs = []
    
    def create_hook_fn(layer_idx, proj_type):
        def hook_fn(module, input_tensor, output_tensor):
            if layer_idx >= start_layer:
                activation_outputs.append(output_tensor)
        return hook_fn
    
    # Register hooks on all layers
    for i, layer in enumerate(llama_model.layers):
        if activation_type == "query":
            hook = layer.self_attn.q_proj.register_forward_hook(create_hook_fn(i, "query"))
        else:
            hook = layer.self_attn.k_proj.register_forward_hook(create_hook_fn(i, "key"))
        activation_hooks.append(hook)
    
    with torch.no_grad():
        # Run forward pass to collect activations through hooks
        outputs = model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=position_ids,
            output_hidden_states=False,
            return_dict=True
        )
        
        # Remove the hooks
        for hook in activation_hooks:
            hook.remove()
    
    # The activations collected via hooks are our result
    return activation_outputs

def compute_attention_key_embeddings_batch(texts: List[str], model: 'LanguageModel', max_length: int = 512) -> List[np.ndarray]:
    """
    Compute attention key embeddings for multiple texts in a single batch.
    
    Args:
        texts: List of texts to embed
        model: Language model
        max_length: Maximum token length
        
    Returns:
        List of key embedding arrays
    """
    if not texts:
        return []
    
    # Tokenize all texts in a batch
    inputs = tokenize_text_batch(texts, model.tokenizer, max_length, model.device)
    
    # Extract key activations
    activations = extract_attention_activations_batch(inputs, model, "key")
    
    # Process activations
    batch_embeddings = process_attention_activations(activations, inputs["attention_mask"])
    
    # Convert to list of numpy arrays
    return [batch_embeddings[i] for i in range(len(texts))]

def compute_attention_query_embeddings_batch(
    texts: List[str], 
    model: 'LanguageModel', 
    max_length: int = 512
) -> List[np.ndarray]:
    """
    Compute embeddings for a batch of attention queries.
    Extracts query projection activations from the model's attention layers.
    
    Args:
        texts: List of texts to embed
        model: Language model
        max_length: Maximum token length
        
    Returns:
        List of embedding arrays
    """
    # Tokenize inputs
    inputs = tokenize_text_batch(texts, model.tokenizer, max_length, model.device)
    
    # Extract query activations
    activations = extract_attention_activations_batch(inputs, model, "query")
    
    # Process activations 
    # If we have a list of activations, process each one and take the mean
    if activations and isinstance(activations, list):
        # Process and average each activation tensor
        processed_activations = []
        for activation in activations:
            # Apply attention mask and get mean over tokens
            batch_size = activation.shape[0]
            
            # Reshape if needed to match attention mask
            if activation.dim() > 3:
                # Flatten the attention heads dimension
                activation = activation.reshape(batch_size, activation.shape[1], -1)
                
            # Apply attention mask
            expanded_mask = inputs["attention_mask"].unsqueeze(-1).expand(-1, -1, activation.shape[-1])
            token_count = inputs["attention_mask"].sum(dim=1, keepdim=True).unsqueeze(-1)
            masked_activation = activation * expanded_mask
            mean_activation = masked_activation.sum(dim=1) / token_count
            
            processed_activations.append(mean_activation)
            
        # Stack and average over layers
        all_layers = torch.stack(processed_activations)
        avg_embeddings = torch.mean(all_layers, dim=0)
        batch_embeddings = avg_embeddings.cpu().numpy()
    elif activations:
        # We received a single activation tensor, process it directly
        activation = activations[-1] if isinstance(activations, list) else activations
        
        # Apply attention mask
        batch_size = activation.shape[0]
        expanded_mask = inputs["attention_mask"].unsqueeze(-1).expand(-1, -1, activation.shape[-1])
        token_count = inputs["attention_mask"].sum(dim=1, keepdim=True).unsqueeze(-1)
        masked_activation = activation * expanded_mask
        mean_activation = masked_activation.sum(dim=1) / token_count
        
        batch_embeddings = mean_activation.cpu().numpy()
    else:
        # No activations captured, return empty arrays
        batch_embeddings = np.zeros((len(texts), 1))
    
    # Ensure each embedding is 1D
    result = []
    for i in range(len(texts)):
        # Flatten if necessary to ensure 1D
        if batch_embeddings[i].ndim > 1:
            result.append(batch_embeddings[i].flatten())
        else:
            result.append(batch_embeddings[i])
            
    return result 