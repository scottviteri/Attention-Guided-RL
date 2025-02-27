"""
Functions for loading and processing Wikipedia articles and key-value pairs.
"""

import os
import json
import requests
import logging
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Iterator
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset

from src.config import (
    WIKI_ARTICLE_TITLE, 
    MAX_PAIRS, 
    KEY_TOKEN_COUNT, 
    VALUE_TOKEN_COUNT,
    QUERY_TOKEN_COUNT,
    MODEL_NAME
)
from src.utils import ensure_dir
from src.embeddings import ada_embedding, batch_ada_embeddings

# Set up logging
logger = logging.getLogger(__name__)

class WikiArticleProcessor:
    """
    Class for processing Wikipedia articles into key-value pairs.
    """
    
    def __init__(
        self, 
        model_name: str = MODEL_NAME,
        cache_dir: str = "data_cache"
    ):
        """
        Initialize the WikiArticleProcessor.
        
        Args:
            model_name: Name of the language model to use for tokenization
            cache_dir: Directory to store cached data
        """
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        ensure_dir(self.cache_dir)
        
        logger.info(f"Initialized WikiArticleProcessor with model: {model_name}")
    
    def fetch_wikipedia_article(self, title: str) -> str:
        """
        Fetch a Wikipedia article by title.
        
        Args:
            title: Title of the Wikipedia article
            
        Returns:
            Text content of the article
        """
        # Fetch from Huggingface datasets
        try:
            # Load the wikipedia dataset
            wiki_dataset = load_dataset(
                "wikipedia", 
                "20220301.en", 
                split="train", 
                streaming=True
            )
            
            # Filter to find the article with the matching title (case insensitive)
            article_found = False
            content = ""
            
            # Convert title to lowercase for case-insensitive comparison
            title_lower = title.lower()
            
            # Search through the dataset
            for article in wiki_dataset:
                if article["title"].lower() == title_lower:
                    content = article["text"]
                    article_found = True
                    break
            
            # If article not found in the stream, fallback to Wikipedia API
            if not article_found:
                logger.warning(f"Article not found in Huggingface dataset: {title}. Falling back to Wikipedia API.")
                return self._fetch_from_wikipedia_api(title)
            
            return content
        
        except Exception as e:
            logger.error(f"Error fetching article from Huggingface dataset: {e}")
            # Fallback to Wikipedia API
            return self._fetch_from_wikipedia_api(title)
    
    def _fetch_from_wikipedia_api(self, title: str) -> str:
        """
        Fetch a Wikipedia article from the Wikipedia API.
        
        Args:
            title: Title of the Wikipedia article
            
        Returns:
            Text content of the article
        """
        # If Huggingface dataset fails, use Wikipedia API as fallback
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "redirects": 1
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            # Extract the page content
            pages = data["query"]["pages"]
            page_id = list(pages.keys())[0]
            content = pages[page_id]["extract"]
            
            return content
        
        except Exception as e:
            logger.error(f"Error fetching Wikipedia article from API: {e}")
            return ""
    
    def tokenize_article_into_chunks(
        self, 
        text: str,
        chunk_size: int,
        max_chunks: int = MAX_PAIRS * 2  # We need twice as many chunks as pairs
    ) -> List[List[int]]:
        """
        Tokenize the entire article and split it into fixed-length chunks.
        
        Args:
            text: Article text
            chunk_size: Size of each chunk in tokens
            max_chunks: Maximum number of chunks to extract
            
        Returns:
            List of token chunks
        """
        # Tokenize the entire article
        all_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Split into chunks
        chunks = []
        for i in range(0, len(all_tokens), chunk_size):
            if len(chunks) >= max_chunks:
                break
                
            chunk = all_tokens[i:i+chunk_size]
            
            # Pad if necessary
            if len(chunk) < chunk_size:
                chunk = chunk + [self.tokenizer.pad_token_id] * (chunk_size - len(chunk))
                
            chunks.append(chunk)
        
        logger.info(f"Split article into {len(chunks)} chunks of size {chunk_size}")
        return chunks
    
    def create_key_value_pairs_from_chunks(
        self, 
        chunks: List[List[int]],
        max_pairs: int = MAX_PAIRS
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Create key-value pairs from consecutive chunks of tokens.
        
        Args:
            chunks: List of token chunks
            max_pairs: Maximum number of pairs to create
            
        Returns:
            List of (key_tokens, value_tokens) pairs
        """
        pairs = []
        
        # Create pairs from consecutive chunks
        for i in range(len(chunks) - 1):
            if len(pairs) >= max_pairs:
                break
                
            key_tokens = chunks[i]
            value_tokens = chunks[i + 1]
            
            pairs.append((key_tokens, value_tokens))
        
        logger.info(f"Created {len(pairs)} key-value pairs from chunks")
        return pairs
    
    def process_article(
        self, 
        title: str = WIKI_ARTICLE_TITLE,
        max_pairs: int = MAX_PAIRS,
        compute_embeddings: bool = True
    ) -> Dict[str, Any]:
        """
        Process a Wikipedia article into key-value pairs with embeddings.
        
        Args:
            title: Title of the Wikipedia article
            max_pairs: Maximum number of key-value pairs
            compute_embeddings: Whether to compute embeddings for keys
            
        Returns:
            Dictionary with key-value pairs and metadata
        """
        logger.info(f"Processing Wikipedia article: {title}")
        
        # Fetch the article
        article_text = self.fetch_wikipedia_article(title)
        if not article_text:
            logger.error(f"Failed to fetch article: {title}")
            return {
                "title": title,
                "pairs": [],
                "key_embeddings": {}
            }
        
        # Tokenize the article into chunks
        chunks = self.tokenize_article_into_chunks(
            article_text, 
            chunk_size=KEY_TOKEN_COUNT,  # Use the key token count as the chunk size
            max_chunks=max_pairs * 2
        )
        
        # Create key-value pairs from chunks
        tokenized_pairs = self.create_key_value_pairs_from_chunks(chunks, max_pairs)
        
        # Convert token IDs back to strings for readability
        readable_pairs = []
        for key_tokens, value_tokens in tokenized_pairs:
            key_text = self.tokenizer.decode(key_tokens)
            value_text = self.tokenizer.decode(value_tokens)
            readable_pairs.append((key_text, value_text))
        
        # Compute embeddings for keys if requested
        key_embeddings = {}
        if compute_embeddings:
            logger.info("Computing embeddings for keys")
            keys = [pair[0] for pair in readable_pairs]
            key_embedding_list = batch_ada_embeddings(keys)
            
            key_embeddings = {
                f"key_{i}": embedding.tolist()
                for i, embedding in enumerate(key_embedding_list)
            }
        
        # Return the processed data
        return {
            "title": title,
            "pairs": readable_pairs,
            "key_embeddings": key_embeddings
        }

class KeyValueDatabase:
    """
    Database for storing and managing key-value pairs during training.
    """
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize the key-value database.
        
        Args:
            data: Processed article data with pairs and key embeddings
        """
        self.title = data["title"]
        self.pairs = data["pairs"]
        
        # Convert key embeddings from lists to numpy arrays
        self.key_embeddings = {}
        for key_id, embedding in data["key_embeddings"].items():
            self.key_embeddings[key_id] = np.array(embedding)
        
        # Create a mapping from key IDs to pairs
        self.key_to_pair = {
            key_id: self.pairs[int(key_id.split('_')[1])]
            for key_id in self.key_embeddings.keys()
        }
        
        # Track available keys
        self.available_keys = list(self.key_embeddings.keys())
        
        logger.info(f"Initialized KeyValueDatabase with {len(self.pairs)} pairs")
    
    def get_available_key_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get embeddings for all available keys.
        
        Returns:
            Dictionary of key IDs to embeddings
        """
        return {
            key_id: self.key_embeddings[key_id]
            for key_id in self.available_keys
        }
    
    def select_key(self, key_id: str) -> Tuple[str, str]:
        """
        Select a key-value pair and remove it from available keys.
        
        Args:
            key_id: ID of the key to select
            
        Returns:
            Selected (key, value) pair
        """
        if key_id not in self.available_keys:
            raise ValueError(f"Key ID not available: {key_id}")
        
        # Remove the key from available keys
        self.available_keys.remove(key_id)
        
        # Return the corresponding pair
        return self.key_to_pair[key_id]
    
    def reset(self) -> None:
        """
        Reset the database to make all keys available again.
        """
        self.available_keys = list(self.key_embeddings.keys())
        logger.debug("Reset KeyValueDatabase")
    
    def is_empty(self) -> bool:
        """
        Check if all keys have been selected.
        
        Returns:
            True if no keys are available
        """
        return len(self.available_keys) == 0

# Helper functions for easier access
def load_wikipedia_article(
    title: str = WIKI_ARTICLE_TITLE,
    max_pairs: int = MAX_PAIRS
) -> KeyValueDatabase:
    """
    Load a Wikipedia article and convert it to a KeyValueDatabase.
    
    Args:
        title: Title of the Wikipedia article
        max_pairs: Maximum number of key-value pairs
        
    Returns:
        KeyValueDatabase for the article
    """
    processor = WikiArticleProcessor()
    
    # Process the article
    data = processor.process_article(title, max_pairs)
    
    return KeyValueDatabase(data) 