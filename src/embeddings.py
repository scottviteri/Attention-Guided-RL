"""
Functions for computing text embeddings using OpenAI's ADA model.
"""

import os
import json
import hashlib
import numpy as np
import time
import logging
from typing import Dict, List, Union, Optional, Any
from openai import OpenAI
from pathlib import Path

from src.config import EMBEDDING_MODEL
from src.utils import ensure_dir

# Set up logging
logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for computing and caching text embeddings using OpenAI's API.
    """
    
    def __init__(
        self, 
        model: str = EMBEDDING_MODEL, 
        cache_dir: str = "embeddings_cache",
        use_cache: bool = True,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the embedding service.
        
        Args:
            model: Name of the OpenAI embedding model
            cache_dir: Directory to store cached embeddings
            use_cache: Whether to use embedding caching
            openai_api_key: OpenAI API key (defaults to env variable)
        """
        self.model = model
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # Create cache directory if it doesn't exist
        if self.use_cache:
            ensure_dir(self.cache_dir)
        
        # Get API key from environment if not provided
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                logger.warning("OPENAI_API_KEY not found in environment variables. Make sure it's set before making API calls.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
        
        logger.info(f"Initialized EmbeddingService with model: {model}")
    
    def _get_cache_path(self, text: str) -> Path:
        """
        Get the cache file path for a text.
        
        Args:
            text: Input text
            
        Returns:
            Path to the cache file
        """
        # Create a hash of the text to use as filename
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return Path(self.cache_dir) / f"{text_hash}.json"
    
    def _load_from_cache(self, text: str) -> Optional[List[float]]:
        """
        Load embeddings from cache if available.
        
        Args:
            text: Input text
            
        Returns:
            Cached embedding or None if not found
        """
        if not self.use_cache:
            return None
        
        cache_path = self._get_cache_path(text)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if the cached embedding is for the same model
                if cache_data.get('model') == self.model:
                    logger.debug(f"Loaded embedding from cache: {cache_path}")
                    return cache_data['embedding']
            except Exception as e:
                logger.warning(f"Error loading from cache: {e}")
        
        return None
    
    def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        """
        Save embeddings to cache.
        
        Args:
            text: Input text
            embedding: Computed embedding
        """
        if not self.use_cache:
            return
        
        cache_path = self._get_cache_path(text)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'text': text[:100] + '...' if len(text) > 100 else text,  # Store truncated text for debugging
                    'model': self.model,
                    'embedding': embedding,
                    'timestamp': time.time()
                }, f)
            logger.debug(f"Saved embedding to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text, using cache if available.
        
        Args:
            text: Input text
            
        Returns:
            Normalized embedding as numpy array
        """
        # Check cache first
        cached_embedding = self._load_from_cache(text)
        if cached_embedding is not None:
            return np.array(cached_embedding)
        
        # If not in cache, compute embedding
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Save to cache
            self._save_to_cache(text, embedding)
            
            return np.array(embedding)
        
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return zeros as fallback (not ideal, but prevents crashes)
            # In production, might want to retry or raise the exception
            return np.zeros(1536)  # ADA embeddings are 1536-dimensional
    
    def get_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of normalized embeddings
        """
        embeddings = []
        
        # First check which texts are in cache
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached_embedding = self._load_from_cache(text)
            if cached_embedding is not None:
                embeddings.append(np.array(cached_embedding))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # If there are uncached texts, compute them in batch
        if uncached_texts:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=uncached_texts
                )
                
                # Process the response
                for i, embedding_data in enumerate(response.data):
                    embedding = embedding_data.embedding
                    text_idx = uncached_indices[i]
                    
                    # Save to cache
                    self._save_to_cache(uncached_texts[i], embedding)
                    
                    # Insert at the correct position
                    if text_idx >= len(embeddings):
                        embeddings.append(np.array(embedding))
                    else:
                        embeddings.insert(text_idx, np.array(embedding))
            
            except Exception as e:
                logger.error(f"Error getting batch embeddings: {e}")
                # Return zeros for failed embeddings
                for _ in uncached_texts:
                    embeddings.append(np.zeros(1536))
        
        return embeddings
    
    def compute_similarity(self, query_embedding: np.ndarray, key_embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute similarity scores between a query embedding and multiple key embeddings.
        
        Args:
            query_embedding: Embedding of the query
            key_embeddings: Dictionary of key IDs to embeddings
            
        Returns:
            Dictionary of key IDs to similarity scores
        """
        similarities = {}
        
        for key_id, key_embedding in key_embeddings.items():
            # Compute dot product
            similarity = np.dot(query_embedding, key_embedding)
            similarities[key_id] = float(similarity)
        
        return similarities

# Create a global instance for easy access
embedding_service = None

def initialize_embedding_service(
    model: str = EMBEDDING_MODEL,
    cache_dir: str = "embeddings_cache",
    use_cache: bool = True
) -> EmbeddingService:
    """
    Initialize the global embedding service.
    
    Args:
        model: Name of the OpenAI embedding model
        cache_dir: Directory to store cached embeddings
        use_cache: Whether to use embedding caching
        
    Returns:
        The embedding service instance
    """
    global embedding_service
    embedding_service = EmbeddingService(model=model, cache_dir=cache_dir, use_cache=use_cache)
    return embedding_service

def get_embedding_service() -> EmbeddingService:
    """
    Get the global embedding service, initializing if necessary.
    
    Returns:
        The embedding service instance
    """
    global embedding_service
    if embedding_service is None:
        embedding_service = initialize_embedding_service()
    return embedding_service

def ada_embedding(text: str) -> np.ndarray:
    """
    Get the ADA embedding for a text.
    
    Args:
        text: Input text
        
    Returns:
        Normalized embedding as numpy array
    """
    service = get_embedding_service()
    return service.get_embedding(text)

def batch_ada_embeddings(texts: List[str]) -> List[np.ndarray]:
    """
    Get ADA embeddings for multiple texts.
    
    Args:
        texts: List of input texts
        
    Returns:
        List of normalized embeddings
    """
    service = get_embedding_service()
    return service.get_batch_embeddings(texts) 