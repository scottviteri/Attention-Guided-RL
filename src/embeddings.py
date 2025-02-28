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
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.config import EMBEDDING_MODEL, MODEL_NAME
from src.utils import ensure_dir
from src.model import load_language_model

# Set up logging
logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = "embedding_cache"

# Embedding service for llama model


class LlamaEmbeddingService:
    """
    Service for computing embeddings using Llama model activations.
    This avoids using external API calls by extracting activation vectors 
    directly from the model.
    """
    
    def __init__(
        self, 
        model: 'LanguageModel',
        cache_dir: str = DEFAULT_CACHE_DIR,
        use_cache: bool = True,
        max_length: int = 512
    ):
        """
        Initialize the Llama embedding service.
        
        Args:
            model: The Llama language model
            cache_dir: Directory for caching embeddings
            use_cache: Whether to use caching
            max_length: Maximum token length for each embedding request
        """
        self.model = model
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.max_length = max_length
        
        # Create cache directory if needed
        if self.use_cache:
            ensure_dir(self.cache_dir)
        
        # Determine hidden size from model configuration
        self.hidden_size = self.model.model.config.hidden_size
        
        logger.info(f"Initialized LlamaEmbeddingService with model: {model.__class__.__name__}")
        logger.debug(f"Hidden size: {self.hidden_size}, Max length: {self.max_length}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text string using Llama model activations.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as numpy array
        """
        # Check cache first if enabled
        if self.use_cache:
            cached_embedding = self._load_from_cache(text)
            if cached_embedding is not None:
                logger.debug(f"Cache HIT for text: {text[:50]}...")
                return cached_embedding
            else:
                logger.debug(f"Cache MISS for text: {text[:50]}...")
        
        # Compute embedding using Llama
        try:
            logger.debug(f"Computing Llama embedding for text: {text[:50]}...")
            start_time = time.time()
            
            embedding = self._compute_llama_embedding(text)
            
            elapsed = time.time() - start_time
            logger.debug(f"Llama embedding computed in {elapsed:.2f}s")
            
            # Cache the result if enabled
            if self.use_cache:
                self._save_to_cache(text, embedding)
                logger.debug(f"Saved embedding to cache for text: {text[:50]}...")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error computing Llama embedding: {e}")
            # Get the model's hidden size for zero vector
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
                hidden_size = self.model.model.config.hidden_size
            else:
                hidden_size = 3072  # Default for Llama models
                
            # Return zero vector as fallback
            return np.zeros(hidden_size)
    
    def _compute_llama_embedding(self, text: str) -> np.ndarray:
        """
        Compute an embedding from Llama hidden states by averaging across all layers.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector
        """
        try:
            # Tokenize input
            tokenizer = self.model.tokenizer
            inputs = tokenizer(text, return_tensors="pt").to(self.model.device)
            
            # Forward pass with output_hidden_states=True
            with torch.no_grad():
                self.model.model.eval()
                outputs = self.model.model(**inputs, output_hidden_states=True)
            
            # Get hidden states (tuple of tensors, one per layer)
            hidden_states = outputs.hidden_states
            
            # Average across all layers and tokens
            # Shape of each hidden state: [batch_size, seq_len, hidden_size]
            all_hidden_states = torch.stack(hidden_states)  # [num_layers, batch_size, seq_len, hidden_size]
            
            # Remove batch dimension (batch_size=1)
            all_hidden_states = all_hidden_states.squeeze(1)  # [num_layers, seq_len, hidden_size]
            
            # Create attention mask to ignore padding tokens
            attention_mask = inputs["attention_mask"]  # [batch_size, seq_len]
            attention_mask = attention_mask.squeeze(0)  # [seq_len]
            
            # Expand mask for broadcasting
            expanded_mask = attention_mask.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            
            # Apply mask
            masked_states = all_hidden_states * expanded_mask
            
            # Sum across tokens and divide by token count
            token_count = attention_mask.sum().item()
            sum_hidden_states = masked_states.sum(dim=1)  # [num_layers, hidden_size]
            
            # Average across layers
            mean_hidden_state = sum_hidden_states.mean(dim=0) / token_count  # [hidden_size]
            
            # Convert to numpy and normalize
            embedding = mean_hidden_state.cpu().numpy()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error computing Llama embedding: {e}")
            # Get the model's hidden size for zero vector
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
                hidden_size = self.model.model.config.hidden_size
            else:
                hidden_size = 3072  # Default for Llama models
                
            # Return zero vector as fallback
            return np.zeros(hidden_size)
    
    def _compute_attention_key_embedding(self, text: str) -> np.ndarray:
        """
        Compute embedding for text using attention key matrices.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as numpy array with shape matching the key dimension (1024)
        """
        try:
            tokenized_input = self.model.tokenizer(
                text, 
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            ).to(self.model.device)
            
            attention_mask = tokenized_input.attention_mask
            
            with torch.inference_mode():
                outputs = self.model.model.forward(
                    input_ids=tokenized_input.input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                key_vectors = []
                hidden_states = outputs.hidden_states

                for layer_idx in range(len(hidden_states) - 1):
                    layer = self.model.model.model.layers[layer_idx]
                    layer_hidden_states = hidden_states[layer_idx]

                    with torch.no_grad():
                        key_proj = torch.matmul(layer_hidden_states, layer.self_attn.k_proj.weight.T)
                        masked_proj = key_proj * attention_mask.unsqueeze(-1)
                        token_count = attention_mask.sum().item()

                        if token_count > 0:
                            key_sum = masked_proj.sum(dim=1) / token_count
                            key_vector = key_sum[0].cpu().numpy()
                            key_vectors.append(key_vector)

            if key_vectors:
                combined_vector = np.mean(key_vectors, axis=0)
                return combined_vector
            else:
                logger.warning("No valid key vectors were extracted")
                return np.zeros(layer.self_attn.k_proj.weight.shape[0])
                
        except Exception as e:
            logger.error(f"Error computing attention key embedding: {str(e)}")
            return np.zeros(layer.self_attn.k_proj.weight.shape[0])

    def _compute_attention_query_embedding(self, text: str, normalize: bool = False) -> np.ndarray:
        """
        Compute embedding for text using attention query matrices.
        
        Args:
            text: Text to embed
            normalize: Whether to normalize the output vector
            
        Returns:
            Embedding as numpy array
        """
        try:
            # Tokenize input
            tokenized_input = self.model.tokenizer(
                text, 
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            ).to(self.model.device)
            
            # Get attention mask
            attention_mask = tokenized_input.attention_mask
            
            with torch.inference_mode():
                outputs = self.model.model.forward(
                    input_ids=tokenized_input.input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Extract query vectors from all attention blocks
                query_vectors = []
                
                # Get all hidden states
                all_hidden_states = outputs.hidden_states
                
                # Clone hidden states outside of inference mode to avoid autograd errors
                # We need to detach and clone to remove the inference mode restriction
                hidden_states = [hs.detach().clone() for hs in all_hidden_states]
            
            # Get all layers except the last one - do this outside inference mode
            for layer_idx in range(len(hidden_states) - 1):
                try:
                    # Access the layer
                    layer = self.model.model.model.layers[layer_idx]
                    
                    # Get hidden states for this layer
                    layer_hidden_states = hidden_states[layer_idx]
                    
                    # Apply query projection - use the layer's query projection matrix
                    with torch.no_grad():  # Use no_grad to avoid gradient computation
                        query_proj = layer.self_attn.q_proj(layer_hidden_states)
                    
                    # Apply attention mask and average
                    mask = attention_mask.unsqueeze(-1)  # Add feature dimension
                    query_sum = torch.sum(query_proj * mask, dim=1)
                    token_count = torch.sum(attention_mask, dim=1).item()
                    
                    if token_count > 0:
                        # Move to CPU before converting to numpy
                        avg_query = (query_sum / token_count).cpu().numpy()
                        query_vectors.append(avg_query[0])  # Extract from batch dimension
                except Exception as e:
                    logger.error(f"Error in layer {layer_idx}: {str(e)}")
            
            # Combine query vectors from all layers
            if query_vectors:
                combined_vector = np.mean(query_vectors, axis=0)
                
                # Log the raw vector norm for debugging
                raw_norm = np.linalg.norm(combined_vector)
                logger.debug(f"Raw query vector norm: {raw_norm}")
                
                # Normalize if requested
                if normalize and raw_norm > 0:
                    combined_vector = combined_vector / raw_norm
                    
                return combined_vector
            else:
                logger.warning("No valid query vectors were extracted")
                return np.zeros(self.hidden_size)
                
        except Exception as e:
            logger.error(f"Error computing attention query embedding: {str(e)}")
            return np.zeros(self.hidden_size)
    
    def get_key_embedding(self, text: str, normalize: bool = False) -> np.ndarray:
        """
        Get a key embedding for the given text using attention key matrices.
        Uses cache if enabled.
        
        Args:
            text: Text to embed
            normalize: Whether to normalize the output vector
            
        Returns:
            Key embedding vector
        """
        # Check cache first if enabled
        if self.use_cache:
            # Use a different cache key to differentiate from regular embeddings
            cache_key = f"key_{text}"
            if normalize:
                cache_key = f"key_norm_{text}"
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                logger.debug(f"Cache HIT for key embedding: {text[:50]}...")
                return cached_embedding
            logger.debug(f"Cache MISS for key embedding: {text[:50]}...")
            
        # Compute embedding
        try:
            start_time = time.time()
            embedding = self._compute_attention_key_embedding(text)
            elapsed = time.time() - start_time
            logger.debug(f"Key embedding computed in {elapsed:.2f}s")
            
            # Cache the result if enabled
            if self.use_cache:
                cache_key = f"key_{text}"
                if normalize:
                    cache_key = f"key_norm_{text}"
                self._save_to_cache(cache_key, embedding)
                logger.debug(f"Saved key embedding to cache for text: {text[:50]}...")
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error in key embedding computation: {e}")
            # Get the model's hidden size for zero vector
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
                hidden_size = self.model.model.config.hidden_size
            else:
                hidden_size = 3072  # Default for Llama models
                
            # Return zero vector as fallback
            return np.zeros(hidden_size)
            
    def get_query_embedding(self, text: str, normalize: bool = False) -> np.ndarray:
        """
        Get a query embedding for the given text using attention query matrices.
        Uses cache if enabled.
        
        Args:
            text: Text to embed
            normalize: Whether to normalize the output vector
            
        Returns:
            Query embedding vector
        """
        # Check cache first if enabled
        if self.use_cache:
            # Use a different cache key to differentiate from regular embeddings
            cache_key = f"query_{text}"
            if normalize:
                cache_key = f"query_norm_{text}"
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                logger.debug(f"Cache HIT for query embedding: {text[:50]}...")
                return cached_embedding
            logger.debug(f"Cache MISS for query embedding: {text[:50]}...")
            
        # Compute embedding
        try:
            start_time = time.time()
            embedding = self._compute_attention_query_embedding(text, normalize=normalize)
            elapsed = time.time() - start_time
            logger.debug(f"Query embedding computed in {elapsed:.2f}s")
            
            # Cache the result if enabled
            if self.use_cache:
                cache_key = f"query_{text}"
                if normalize:
                    cache_key = f"query_norm_{text}"
                self._save_to_cache(cache_key, embedding)
                logger.debug(f"Saved query embedding to cache for text: {text[:50]}...")
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error in query embedding computation: {e}")
            # Get the model's hidden size for zero vector
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
                hidden_size = self.model.model.config.hidden_size
            else:
                hidden_size = 3072  # Default for Llama models
                
            # Return zero vector as fallback
            return np.zeros(hidden_size)
    
    def get_embeddings(self, texts: List[str], are_queries: bool = True, normalize: bool = False) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts using Llama model activations.
        
        Args:
            texts: List of texts to embed
            are_queries: Whether the texts are queries (vs keys)
            normalize: Whether to normalize the embeddings
            
        Returns:
            List of embeddings as numpy arrays
        """
        # Check cache first if enabled
        if self.use_cache:
            # Try to load all from cache
            cached_embeddings = []
            cache_misses = []
            cache_miss_indices = []
            
            for i, text in enumerate(texts):
                cached_embedding = self._load_from_cache(text)
                if cached_embedding is not None:
                    cached_embeddings.append((i, cached_embedding))
                else:
                    cache_misses.append(text)
                    cache_miss_indices.append(i)
            
            # If all were cached, return them
            if len(cached_embeddings) == len(texts):
                logger.debug(f"All {len(texts)} embeddings found in cache")
                # Sort by original index
                sorted_embeddings = sorted(cached_embeddings, key=lambda x: x[0])
                return [emb for _, emb in sorted_embeddings]
        else:
            # If cache disabled, all are misses
            cache_misses = texts
            cache_miss_indices = list(range(len(texts)))
            cached_embeddings = []
        
        # Compute embeddings for cache misses
        computed_embeddings = []
        if cache_misses:
            logger.info(f"Computing {len(cache_misses)} embeddings (cache misses)")
            
            if are_queries:
                # Compute embeddings for queries - use attention mechanism
                computed_embeddings = self._compute_attention_query_batch_embeddings(cache_misses, normalize)
            else:
                # Regular embedding computation for keys
                computed_embeddings = self._compute_llama_batch_embeddings(cache_misses)
            
            # Cache the computed embeddings
            if self.use_cache:
                for text, embedding in zip(cache_misses, computed_embeddings):
                    self._save_to_cache(text, embedding)
        
        # Combine cached and computed embeddings
        if cached_embeddings:
            # Create result array with correct size
            result = [None] * len(texts)
            
            # Fill in cached embeddings
            for idx, emb in cached_embeddings:
                result[idx] = emb
            
            # Fill in computed embeddings
            for i, idx in enumerate(cache_miss_indices):
                result[idx] = computed_embeddings[i]
            
            return result
        else:
            # All were computed
            return computed_embeddings
    
    def _compute_llama_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Compute embeddings for multiple texts in a single forward pass.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        try:
            # Tokenize inputs with padding
            tokenizer = self.model.tokenizer
            encoded_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(self.model.device)
            
            # Forward pass with output_hidden_states=True
            with torch.no_grad():
                self.model.model.eval()
                outputs = self.model.model(**encoded_inputs, output_hidden_states=True)
            
            # Get hidden states (tuple of tensors, one per layer)
            hidden_states = outputs.hidden_states
            
            # Stack all layers and get attention mask
            all_hidden_states = torch.stack(hidden_states)  # [num_layers, batch_size, seq_len, hidden_size]
            attention_mask = encoded_inputs["attention_mask"]  # [batch_size, seq_len]
            
            # Create embeddings list
            embeddings = []
            
            # Process each text in the batch
            for batch_idx in range(len(texts)):
                # Get attention mask for this text
                mask = attention_mask[batch_idx]  # [seq_len]
                seq_len = mask.sum().item()
                
                # Skip empty sequences
                if seq_len == 0:
                    hidden_size = all_hidden_states.size(-1)
                    embeddings.append(np.zeros(hidden_size))
                    continue
                
                # Extract states for this batch item
                # Shape: [num_layers, seq_len, hidden_size]
                batch_states = all_hidden_states[:, batch_idx, :, :]
                
                # Apply mask to ignore padding
                # First expand mask to match dimensions
                expanded_mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
                
                # Multiply by mask
                masked_batch_states = batch_states * expanded_mask
                
                # Sum across token positions and normalize by token count
                sum_states = masked_batch_states.sum(dim=1)  # [num_layers, hidden_size]
                
                # Average across layers
                mean_state = sum_states.mean(dim=0) / seq_len  # [hidden_size]
                
                # Convert to numpy and normalize
                embedding = mean_state.cpu().numpy()
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error computing batch embeddings: {e}")
            # Return empty list or zero vectors as fallback
            if len(texts) > 0:
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
                    hidden_size = self.model.model.config.hidden_size
                else:
                    hidden_size = 3072  # Default for Llama models
                
                return [np.zeros(hidden_size) for _ in texts]
            return []
            
    def _compute_attention_query_batch_embeddings(self, texts: List[str], normalize: bool = False) -> List[np.ndarray]:
        """
        Compute query embeddings for multiple texts using attention query matrices.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to normalize the output vectors
            
        Returns:
            List of embeddings as numpy arrays
        """
        if not texts:
            return []
            
        try:
            tokenized_inputs = self.model.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.model.device)
            
            with torch.inference_mode():
                outputs = self.model.model.forward(
                    input_ids=tokenized_inputs.input_ids,
                    attention_mask=tokenized_inputs.attention_mask,
                    output_hidden_states=True
                )
                
                attention_mask = tokenized_inputs.attention_mask
                all_hidden_states = outputs.hidden_states
                
                # Clone hidden states outside of inference mode to avoid autograd errors
                hidden_states = [hs.detach().clone() for hs in all_hidden_states]
            
            # Process each text in the batch
            batch_size = len(texts)
            embeddings = []
            
            for batch_idx in range(batch_size):
                # Extract attention mask for this text
                mask = attention_mask[batch_idx]
                
                # Skip empty sequences
                if not torch.any(mask):
                    embeddings.append(np.zeros(self.hidden_size))
                    continue
                
                # Extract query vectors from all attention blocks
                query_vectors = []
                
                # Get all layers except the last one
                for layer_idx in range(len(hidden_states) - 1):
                    try:
                        # Access the layer
                        layer = self.model.model.model.layers[layer_idx]
                        
                        # Get hidden states for this text and layer
                        text_hidden_states = hidden_states[layer_idx][batch_idx].unsqueeze(0)  # Add batch dimension
                        
                        # Apply query projection with no_grad to avoid gradient computation
                        with torch.no_grad():
                            query_proj = layer.self_attn.q_proj(text_hidden_states).squeeze(0)  # Remove batch dimension
                        
                        # Apply attention mask and average
                        query_sum = torch.sum(query_proj * mask.unsqueeze(-1), dim=0)
                        token_count = torch.sum(mask).item()
                        
                        if token_count > 0:
                            # Move to CPU before converting to numpy
                            avg_query = (query_sum / token_count).cpu().numpy()
                            query_vectors.append(avg_query)
                    except Exception as e:
                        logger.error(f"Error in layer {layer_idx} for batch {batch_idx}: {str(e)}")
                
                # Combine query vectors from all layers
                if query_vectors:
                    combined_vector = np.mean(query_vectors, axis=0)
                    
                    # Normalize if requested
                    if normalize:
                        norm = np.linalg.norm(combined_vector)
                        if norm > 0:
                            combined_vector = combined_vector / norm
                else:
                    combined_vector = np.zeros(self.hidden_size)
                    
                embeddings.append(combined_vector)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error computing batch query embeddings: {str(e)}")
            # Return empty vectors on error
            return [np.zeros(self.hidden_size) for _ in range(len(texts))]
    
    def compute_similarity(
        self, 
        query_embedding: np.ndarray,
        key_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute cosine similarity between query embedding and key embeddings.
        
        Args:
            query_embedding: Query embedding
            key_embeddings: Dictionary of key IDs to embeddings
            
        Returns:
            Dictionary of key IDs to similarity scores
        """
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Compute cosine similarity for each key
        similarities = {}
        
        for key_id, key_embedding in key_embeddings.items():
            # Normalize key embedding
            key_norm = np.linalg.norm(key_embedding)
            if key_norm > 0:
                key_embedding = key_embedding / key_norm
            
            # Compute dot product (cosine similarity for normalized vectors)
            similarity = np.dot(query_embedding, key_embedding)
            
            # Ensure value is in valid range
            similarity = max(0.0, min(1.0, float(similarity)))
            
            similarities[key_id] = similarity
        
        return similarities
    
    def _get_cache_path(self, text: str) -> Path:
        """Get file path for cached embedding."""
        # Create a hash of the text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Create file path
        return Path(self.cache_dir) / f"llama_{text_hash}.json"
    
    def _load_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Load embedding from cache if available."""
        cache_path = self._get_cache_path(text)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                
                return np.array(data["embedding"])
            except Exception as e:
                logger.warning(f"Error loading from cache: {e}")
                return None
        
        return None
    
    def _save_to_cache(self, text: str, embedding: np.ndarray) -> None:
        """Save embedding to cache."""
        cache_path = self._get_cache_path(text)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "embedding": embedding.tolist(),
                    "timestamp": time.time()
                }, f)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")


def compute_embeddings(
    texts: List[str], 
    are_queries: Union[bool, List[bool]] = True, 
    normalize: bool = False,
    model: Optional['LanguageModel'] = None
) -> List[np.ndarray]:
    """
    Compute embeddings for a list of texts using the Llama model.
    
    Args:
        texts: List of texts to embed
        are_queries: Whether the texts are queries (vs keys) or a list specifying for each text
        normalize: Whether to normalize the embeddings
        model: Optional pre-loaded language model to use
        
    Returns:
        List of embeddings as numpy arrays
    """
    # Handle empty input
    if not texts:
        return []
    
    # Check if we have query flags for each text
    if isinstance(are_queries, list):
        if len(are_queries) != len(texts):
            raise ValueError(f"Length of are_queries ({len(are_queries)}) must match length of texts ({len(texts)})")
    else:
        # Use the same flag for all texts
        are_queries = [are_queries] * len(texts)
    
    # Create model and service just for this embedding operation
    if model is None:
        model = load_language_model()
    
    service = LlamaEmbeddingService(model=model, use_cache=True)
        
    logger.debug(f"Using Llama model for {len(texts)} embeddings")
    return service.get_embeddings(texts, are_queries=are_queries, normalize=normalize) 