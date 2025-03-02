"""
Functions for loading and processing Wikipedia articles and key-value pairs.
"""
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Iterator
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from dataclasses import dataclass
import random

from src.config import (
    MAX_PAIRS, 
    KEY_TOKEN_COUNT, 
    VALUE_TOKEN_COUNT,
    MODEL_NAME,
    MIN_ARTICLE_TOKENS
)
from src.embeddings import get_embeddings
from src.model import load_language_model

# Set up logging
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class KeyValuePair:
    """Key-value pair with embeddings."""
    key_tokens: torch.Tensor
    value_tokens: torch.Tensor
    key_embedding: np.ndarray
    key_id: str = ""
    key: str = ""
    value: str = ""

def tokenize_articles(tokenizer_name: str = MODEL_NAME) -> Iterator[Tuple[str, List[int]]]:
    """Tokenize articles and filter by minimum token length."""
    # Verify MIN_ARTICLE_TOKENS is greater than chunk size
    chunk_size = KEY_TOKEN_COUNT + VALUE_TOKEN_COUNT
    assert MIN_ARTICLE_TOKENS >= chunk_size, f"MIN_ARTICLE_TOKENS ({MIN_ARTICLE_TOKENS}) must be >= chunk_size ({chunk_size})"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    
    for article in wiki_dataset:
        title, text = article.get("title", ""), article.get("text", "")
        if not title or not text:
            continue
            
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= MIN_ARTICLE_TOKENS:
            yield (title, tokens)
        else:
            logger.debug(f"Skipping short article '{title}': {len(tokens)} tokens")

def chunk_tokens(tokenized_articles: Iterator[Tuple[str, List[int]]], device: str = None) -> Iterator[Tuple[str, List[torch.Tensor]]]:
    """Convert token lists into chunks of fixed size."""
    chunk_size = KEY_TOKEN_COUNT + VALUE_TOKEN_COUNT
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for title, tokens in tokenized_articles:
        tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        
        # Create non-overlapping chunks
        chunks = []
        for i in range(0, len(tokens_tensor) - chunk_size + 1, chunk_size):
            chunk = tokens_tensor[i:i+chunk_size]
            if len(chunk) == chunk_size:  # Ensure full chunks
                chunks.append(chunk)
        
        assert chunks, f"No chunks created for '{title}' (length: {len(tokens)})"
        yield (title, chunks)

def sample_chunks(chunked_articles: Iterator[Tuple[str, List[torch.Tensor]]]) -> Iterator[Tuple[str, List[torch.Tensor]]]:
    """Sample up to max_pairs chunks from each article."""
    for title, chunks in chunked_articles:
        if len(chunks) > MAX_PAIRS:
            sampled_chunks = random.sample(chunks, MAX_PAIRS)
        else:
            sampled_chunks = chunks
        yield (title, sampled_chunks)

def split_key_value(sampled_articles: Iterator[Tuple[str, List[torch.Tensor]]]) -> Iterator[Tuple[str, List[Tuple[torch.Tensor, torch.Tensor]]]]:
    """Split chunks into (key, value) pairs."""
    for title, chunks in sampled_articles:
        pairs = []
        for chunk in chunks:
            key_tokens = chunk[:KEY_TOKEN_COUNT]
            value_tokens = chunk[KEY_TOKEN_COUNT:]
            pairs.append((key_tokens, value_tokens))
        yield (title, pairs)

def add_embeddings(article_pairs: Iterator[Tuple[str, List[Tuple[torch.Tensor, torch.Tensor]]]],
                  device: str = None) -> Iterator[Tuple[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]]:
    """Add embeddings to key parts of pairs."""
    # Determine device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    model = load_language_model(device=device)
    
    for title, pairs in article_pairs:
        # Extract just the keys for embedding - ensure they're on the right device
        keys = [key.to(device) for key, _ in pairs]
        
        # Stack keys into a single tensor for batch processing
        if keys:
            # Create a single batch of tokens
            stacked_keys = torch.stack(keys)
            # Create corresponding attention mask (all 1s since these are full tokens)
            attention_mask = torch.ones(stacked_keys.shape[0], stacked_keys.shape[1], device=device)
            
            # Compute embeddings for all keys at once
            key_embeddings = get_embeddings(
                tokens=stacked_keys, 
                attention_mask=attention_mask,
                model=model, 
                are_queries=False
            )
            
            # Add embeddings to pairs - move tensors back to CPU for easier storage/transfer
            pairs_with_embeddings = [(k.cpu(), v.cpu(), e.cpu()) for (k, v), e in zip(pairs, key_embeddings)]
        else:
            # Empty case - should rarely happen
            pairs_with_embeddings = []
            
        yield (title, pairs_with_embeddings)

def to_dataclass(article_embedded_pairs: Iterator[Tuple[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]]) -> Iterator[List[KeyValuePair]]:
    """Convert to KeyValuePair dataclasses."""
    for title, triplets in article_embedded_pairs:
        result = []
        for i, (key_tokens, value_tokens, embedding) in enumerate(triplets):
            result.append(KeyValuePair(
                key_tokens=key_tokens,
                value_tokens=value_tokens,
                key_embedding=embedding,
                key_id=f"{title}_{i}"
            ))
        yield result

def get_wikipedia_kv_stream(device: str = None) -> Iterator[List[KeyValuePair]]:
    """
    Process Wikipedia articles into KeyValuePair objects.
    
    Args:
        device: Device to load the model on (cuda or cpu)
        
    Returns:
        Iterator of lists of KeyValuePair objects, one list per article
    """
    # Determine device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return to_dataclass(
        add_embeddings(
            split_key_value(
                sample_chunks(
                    chunk_tokens(
                        tokenize_articles(),
                        device=device
                    )
                )
            ),
            device=device
        )
    ) 