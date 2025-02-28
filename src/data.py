"""
Functions for loading and processing Wikipedia articles and key-value pairs.
"""

import os
import json
import requests
import logging
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Iterator, Generator, Iterable
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import dataclasses
from dataclasses import dataclass, field

from src.config import (
    WIKI_ARTICLE_TITLE, 
    MAX_PAIRS, 
    KEY_TOKEN_COUNT, 
    VALUE_TOKEN_COUNT,
    QUERY_TOKEN_COUNT,
    MODEL_NAME
)
from src.utils import ensure_dir
from src.embeddings import ada_embedding, batch_ada_embeddings, batch_compute_embeddings

# Set up logging
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class KeyValuePair:
    """
    A frozen dataclass representing a key-value pair with embeddings.
    
    Attributes:
        key_tokens: Tokenized key content as a PyTorch tensor
        value_tokens: Tokenized value content as a PyTorch tensor
        key_embedding: Embedding vector for the key
        key_id: Optional identifier for the key
    """
    key_tokens: torch.Tensor
    value_tokens: torch.Tensor
    key_embedding: np.ndarray
    key_id: str = ""

@dataclass(frozen=True)
class WikiArticle:
    """
    A frozen dataclass representing a Wikipedia article with its tokenized chunks.
    
    Attributes:
        title: The title of the article
        chunks: List of tokenized chunks from the article as PyTorch tensors
        text: The original article text (optional)
    """
    title: str
    chunks: List[torch.Tensor]
    text: Optional[str] = None

@dataclass(frozen=True)
class BatchedWikiArticle:
    """
    A frozen dataclass representing a batch of Wikipedia articles.
    
    Attributes:
        titles: List of article titles
        chunks_list: List of tokenized chunks for each article as PyTorch tensors
        texts: List of original article texts (optional)
    """
    titles: List[str]
    chunks_list: List[List[torch.Tensor]]
    texts: Optional[List[str]] = None

def create_wiki_dataset_iterator(streaming: bool = True) -> Iterator[Dict[str, Any]]:
    """
    Create an iterator over the Wikipedia dataset.
    
    Args:
        streaming: Whether to stream the dataset (recommended for large datasets)
        
    Returns:
        Iterator over Wikipedia articles
    """
    try:
        # Load the wikipedia dataset
        wiki_dataset = load_dataset(
            "wikipedia", 
            "20220301.en", 
            split="train", 
            streaming=streaming
        )
        return iter(wiki_dataset)
    except Exception as e:
        logger.error(f"Error loading Wikipedia dataset: {e}")
        # Return empty iterator in case of failure
        return iter([])

def fetch_from_wikipedia_api(title: str) -> str:
    """
    Fetch a Wikipedia article from the Wikipedia API.
    
    Args:
        title: Title of the Wikipedia article
        
    Returns:
        Text content of the article
    """
    # Use Wikipedia API as fallback
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

def next_article(wiki_iterator: Iterator[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Get the next article from the Wikipedia dataset iterator.
    
    Args:
        wiki_iterator: Iterator over Wikipedia articles
        
    Returns:
        Tuple of (title, text)
    """
    try:
        article = next(wiki_iterator)
        return article["title"], article["text"]
    except StopIteration:
        logger.warning("No more articles in the dataset")
        return "", ""
    except Exception as e:
        logger.error(f"Error getting next article: {e}")
        return "", ""

def tokenize_article_into_chunks(
    text: str,
    tokenizer,
    chunk_size: int,
    max_chunks: int = None,
    skip_if_too_short: bool = True
) -> List[torch.Tensor]:
    """
    Tokenize article text into chunks of consistent token size.
    Now returns a list of PyTorch tensor chunks rather than a generator.
    
    Args:
        text: Article text
        tokenizer: Tokenizer to use
        chunk_size: Size of each chunk in tokens
        max_chunks: Maximum number of chunks to return (None for all)
        skip_if_too_short: If True, raise ValueError if article is too short for max_chunks
        
    Returns:
        List of token chunks as PyTorch tensors of consistent size
    """
    logger.debug(f"Tokenizing article: {text[:50]}...")
    
    # Tokenize the entire article
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except Exception as e:
        logger.error(f"Error tokenizing article: {e}")
        return []
        
    # Check if we have enough tokens
    if max_chunks is not None and len(tokens) < chunk_size * max_chunks and skip_if_too_short:
        logger.warning(f"Article too short: {len(tokens)} tokens, need {chunk_size * max_chunks}")
        raise ValueError("Article too short")
    
    # Create chunks
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i+chunk_size]
        
        # Skip partial chunks at the end
        if len(chunk) < chunk_size:
            continue
            
        # Convert chunk to a PyTorch tensor
        tensor_chunk = torch.tensor(chunk, dtype=torch.long)
        chunks.append(tensor_chunk)
        
        # Stop if we have enough chunks
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
    
    logger.debug(f"Created {len(chunks)} chunks")
    return chunks

def create_key_value_pairs_from_chunks(
    chunks: List[torch.Tensor], 
    max_pairs: int = MAX_PAIRS,
    non_overlapping: bool = True,
    random_selection: bool = True
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create key-value pairs from chunks of tokens.
    
    Args:
        chunks: List of token chunks as PyTorch tensors
        max_pairs: Maximum number of pairs to create
        non_overlapping: If True, create non-overlapping pairs [(a,b), (c,d)...]
                         If False, create overlapping pairs [(a,b), (b,c), (c,d)...]
        random_selection: If True, randomly select which consecutive pairs to use
                          If False, select consecutive pairs from the beginning
    
    Returns:
        List of (key_tokens, value_tokens) tuples where tokens are PyTorch tensors
    """
    pairs = []
    
    # Create pairs
    if random_selection:
        # Select random CONSECUTIVE pairs from the article
        import random
        
        # First determine all possible consecutive pairs based on non_overlapping setting
        possible_pairs = []
        if non_overlapping:
            # Non-overlapping pairs: (0,1), (2,3), (4,5), ...
            for i in range(0, len(chunks) - 1, 2):
                if i + 1 < len(chunks):
                    possible_pairs.append((i, i + 1))
        else:
            # Overlapping pairs: (0,1), (1,2), (2,3), ...
            for i in range(len(chunks) - 1):
                possible_pairs.append((i, i + 1))
        
        # Need at least 1 possible pair
        if not possible_pairs:
            logger.warning(f"Not enough chunks to create pairs: {len(chunks)} chunks")
            return pairs
        
        # Shuffle the possible pairs
        random.shuffle(possible_pairs)
        
        # Take up to max_pairs pairs
        selected_pairs = possible_pairs[:max_pairs]
        
        # Create the actual pairs using the selected indices
        for i, j in selected_pairs:
            pairs.append((chunks[i], chunks[j]))
    else:
        # Original consecutive pairs logic
        if non_overlapping:
            # Create non-overlapping pairs: (chunk[0], chunk[1]), (chunk[2], chunk[3]), ...
            for i in range(0, len(chunks) - 1, 2):
                if len(pairs) >= max_pairs:
                    break
                if i + 1 < len(chunks):
                    pairs.append((chunks[i], chunks[i + 1]))
        else:
            # Create overlapping pairs: (chunk[0], chunk[1]), (chunk[1], chunk[2]), ...
            for i in range(len(chunks) - 1):
                if len(pairs) >= max_pairs:
                    break
                pairs.append((chunks[i], chunks[i + 1]))
    
    logger.info(f"Created {len(pairs)} key-value pairs from chunks")
    return pairs

def process_article_text(
    article_text: str,
    tokenizer,
    title: str = "Wikipedia Article",
    max_pairs: int = MAX_PAIRS,
    compute_embeddings: bool = True
) -> Dict[str, Any]:
    """
    Process article text into key-value pairs.
    
    Args:
        article_text: Text of the article
        tokenizer: Tokenizer to use
        title: Title of the article
        max_pairs: Maximum number of key-value pairs to create
        compute_embeddings: Whether to compute key embeddings
        
    Returns:
        Dictionary with title, key-value pairs, and key embeddings
    
    Raises:
        ValueError: If the article is too short to create chunks
    """
    logger.info("Processing article text into key-value pairs")
    
    # Generate chunks
    chunks = tokenize_article_into_chunks(
        article_text, 
        tokenizer, 
        chunk_size=KEY_TOKEN_COUNT,
        max_chunks=max_pairs * 2  # Need twice as many chunks as pairs
    )
    
    # Create WikiArticle object
    wiki_article = WikiArticle(title=title, chunks=chunks, text=article_text)
    
    # Create key-value pairs
    pairs = create_key_value_pairs_from_chunks(
        chunks=wiki_article.chunks, 
        max_pairs=max_pairs, 
        random_selection=True  # Explicitly use random selection
    )
    
    if not pairs:
        logger.warning("No key-value pairs could be created from the article")
        return {
            "title": title,
            "pairs": [],
            "key_embeddings": {}
        }
    
    # Decode tokens to text
    text_pairs = []
    key_texts = []
    
    for key_tokens, value_tokens in pairs:
        key_text = tokenizer.decode(key_tokens)
        value_text = tokenizer.decode(value_tokens)
        
        text_pairs.append((key_text, value_text))
        key_texts.append(key_text)
    
    # Compute embeddings for keys if requested
    key_embeddings = {}
    if compute_embeddings and key_texts:
        logger.info(f"Computing embeddings for {len(key_texts)} keys")
        embeddings = batch_compute_embeddings(key_texts, are_queries=False)
        
        for i, embedding in enumerate(embeddings):
            key_embeddings[f"key_{i}"] = embedding
    
    # Return processed data
    result = {
        "title": title,
        "pairs": text_pairs,
        "key_embeddings": key_embeddings
    }
    
    logger.info(f"Article processed with {len(text_pairs)} key-value pairs")
    return result

def process_next_article(
    wiki_iterator: Iterator[Dict[str, Any]],
    tokenizer_name: str = MODEL_NAME,
    max_pairs: int = MAX_PAIRS
) -> Dict[str, Any]:
    """
    Process the next article from the Wikipedia dataset.
    
    Args:
        wiki_iterator: Iterator over Wikipedia articles
        tokenizer_name: Name of the tokenizer model to use
        max_pairs: Maximum number of key-value pairs
        
    Returns:
        Processed article data
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Get next article
    title, text = next_article(wiki_iterator)
    
    if not text:
        logger.error("Failed to get next article")
        return {
            "title": title or "Unknown",
            "pairs": [],
            "key_embeddings": {}
        }
    
    # Process the article
    data = process_article_text(text, tokenizer, title, max_pairs)
    data["title"] = title  # Use the actual title
    
    return data

def load_wikipedia_article(
    title: str = WIKI_ARTICLE_TITLE,
    max_pairs: int = MAX_PAIRS,
    tokenizer_name: str = MODEL_NAME
) -> List[KeyValuePair]:
    """
    Load a Wikipedia article and convert it to a list of KeyValuePair objects.
    
    Args:
        title: Title of the Wikipedia article
        max_pairs: Maximum number of key-value pairs
        tokenizer_name: Name of the tokenizer model to use
        
    Returns:
        List of KeyValuePair objects
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Try to fetch from Wikipedia API first as it's more direct for a known title
    logger.info(f"Fetching article with title: {title}")
    article_text = fetch_from_wikipedia_api(title)
    
    if article_text:
        # Process the article using the batch processing helper
        kv_pairs = process_article_for_batch(
            title=title,
            article_text=article_text,
            tokenizer=tokenizer,
            chunk_size=KEY_TOKEN_COUNT,
            min_chunks_per_article=max_pairs // 2  # Need half as many chunks as max_pairs for non-overlapping pairs
        )
        if kv_pairs:
            return kv_pairs
    
    # If API fetch failed, fall back to the dataset search
    logger.info(f"API fetch failed, searching for '{title}' in dataset")
    
    # Create wiki iterator
    wiki_iterator = create_wiki_dataset_iterator()
    
    # Search through articles to find one with matching title
    title_lower = title.lower()
    
    while True:
        try:
            current_title, current_text = next_article(wiki_iterator)
            
            # Check if title matches (case insensitive)
            if current_title.lower() == title_lower:
                kv_pairs = process_article_for_batch(
                    title=current_title,
                    article_text=current_text,
                    tokenizer=tokenizer,
                    chunk_size=KEY_TOKEN_COUNT,
                    min_chunks_per_article=max_pairs // 2
                )
                # Always return the result, even if empty
                return kv_pairs
                
        except (StopIteration, Exception) as e:
            # End of dataset or error
            break
    
    # If article not found, return an empty list
    logger.error(f"Article '{title}' not found")
    return []

def load_random_wikipedia_article(
    max_pairs: int = MAX_PAIRS,
    tokenizer_name: str = MODEL_NAME
) -> List[KeyValuePair]:
    """
    Load a random Wikipedia article and convert it to a list of KeyValuePair objects.
    
    This function will automatically skip articles that are too short to process.
    
    Args:
        max_pairs: Maximum number of key-value pairs
        tokenizer_name: Name of the tokenizer model to use
        
    Returns:
        List of KeyValuePair objects
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Create iterator
    wiki_iterator = create_wiki_dataset_iterator()
    
    # Try to find a suitable article
    while True:
        try:
            # Get next article
            title, article_text = next_article(wiki_iterator)
            if not title or not article_text:
                break
                
            # Try to process it
            kv_pairs = process_article_for_batch(
                title=title,
                article_text=article_text,
                tokenizer=tokenizer,
                chunk_size=KEY_TOKEN_COUNT,
                min_chunks_per_article=max_pairs // 2
            )
            
            # Return the first suitable article
            if kv_pairs:
                logger.info(f"Loaded random article: {title} with {len(kv_pairs)} pairs")
                return kv_pairs
                
        except (StopIteration, Exception) as e:
            # End of articles or error
            if not isinstance(e, StopIteration):
                logger.error(f"Error loading random article: {e}")
            break
    
    # If no suitable article found
    logger.error("Failed to find a suitable random article")
    return []

def process_articles_stream(
    wiki_iterator, 
    tokenizer,
    chunk_size: int,
    min_chunks_per_article: int = 10,
    max_pairs: int = MAX_PAIRS
) -> Iterator[List[KeyValuePair]]:
    """
    Process a stream of Wikipedia articles, yielding KeyValuePair lists.
    
    This function processes articles one by one and yields a list of KeyValuePair
    objects for each successfully processed article.
    
    Args:
        wiki_iterator: Iterator yielding Wikipedia articles
        tokenizer: Tokenizer to use for processing text
        chunk_size: Size of each chunk in tokens
        min_chunks_per_article: Minimum number of chunks required per article
        max_pairs: Maximum number of key-value pairs to create per article
        
    Yields:
        List of KeyValuePair objects for each processed article
    """
    logger.info(f"Processing Wikipedia articles stream with chunk_size={chunk_size}")
    
    while True:
        try:
            # Get next article
            title, article_text = next_article(wiki_iterator)
            if not title or not article_text:
                break
                
            logger.info(f"Processing article: {title}")
            
            # Process the article
            kv_pairs = process_article_for_batch(
                title=title,
                article_text=article_text,
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                min_chunks_per_article=min_chunks_per_article
            )
            
            # Skip if processing failed
            if not kv_pairs:
                continue
                
            # Yield the processed article
            logger.info(f"Yielding processed article: {title} with {len(kv_pairs)} pairs")
            yield kv_pairs
                
        except (StopIteration, Exception) as e:
            # End of articles or other error
            if not isinstance(e, StopIteration):
                logger.error(f"Error processing articles: {e}")
            break

def process_article_for_batch(
    title: str,
    article_text: str,
    tokenizer,
    chunk_size: int,
    min_chunks_per_article: int = 10
) -> List[KeyValuePair]:
    """
    Process a single article for batch processing, creating non-overlapping key-value pairs.
    
    Args:
        title: Article title
        article_text: Text content of the article
        tokenizer: Tokenizer to use for processing
        chunk_size: Size of each chunk in tokens
        min_chunks_per_article: Minimum number of chunks required for the article
        
    Returns:
        List of KeyValuePair objects or None if article is too short
    """
    try:
        # Tokenize article into chunks
        chunks = tokenize_article_into_chunks(
            article_text,
            tokenizer,
            chunk_size=chunk_size,
            max_chunks=min_chunks_per_article * 2,  # Need at least twice the min for non-overlapping pairs
            skip_if_too_short=True
        )
        
        # Create WikiArticle object
        wiki_article = WikiArticle(title=title, chunks=chunks, text=article_text)
        
        # Create non-overlapping key-value pairs
        pairs = []
        key_texts = []
        
        for i in range(0, len(wiki_article.chunks) - 1, 2):
            if i+1 < len(wiki_article.chunks):
                # Store token chunks
                pairs.append((i, wiki_article.chunks[i], wiki_article.chunks[i+1]))
                
                # Add key text for batch embedding - convert tensor to list for tokenizer
                key_text = tokenizer.decode(wiki_article.chunks[i].tolist())
                key_texts.append(key_text)
        
        # Check if we have enough pairs
        if len(pairs) < min_chunks_per_article // 2:
            logger.warning(f"Article '{title}' has too few pairs ({len(pairs)}), skipping")
            return []
            
        # Get embeddings for all keys at once
        key_embeddings = batch_compute_embeddings(key_texts, are_queries=False)
        
        # Create KeyValuePair objects
        result_pairs = []
        for idx, (pair_idx, key_tokens, value_tokens) in enumerate(pairs):
            key_id = f"key_{idx}"
            
            # Create KeyValuePair
            pair = KeyValuePair(
                key_id=key_id,
                key_tokens=key_tokens,
                value_tokens=value_tokens,
                key_embedding=key_embeddings[idx]
            )
            result_pairs.append(pair)
        
        logger.info(f"Created {len(result_pairs)} non-overlapping pairs from article '{title}'")
        return result_pairs
        
    except ValueError as e:
        # Article was too short
        logger.warning(f"Article '{title}' is too short: {str(e)}")
        return []
    except Exception as e:
        # Other errors
        logger.error(f"Error processing article '{title}': {e}")
        return []

def batch_process_articles(
    wiki_iterator: Iterator[Dict[str, Any]],
    tokenizer,
    batch_size: int = 4,
    chunk_size: int = KEY_TOKEN_COUNT,
    min_chunks_per_article: int = 10
) -> Iterator[List[KeyValuePair]]:
    """
    Process Wikipedia articles and yield batches of KeyValuePair objects.
    
    This function:
    1. Collects multiple articles and their tokenized chunks
    2. Creates a BatchedWikiArticle object to hold the data
    3. Processes all articles at once to create batches of KeyValuePair objects
    
    Args:
        wiki_iterator: Iterator over Wikipedia articles
        tokenizer: Tokenizer to use for processing text
        batch_size: Number of key-value pairs in each batch
        chunk_size: Size of each chunk in tokens
        min_chunks_per_article: Minimum number of chunks required per article
    
    Yields:
        List of KeyValuePair objects in each batch
    """
    logger.info(f"Starting batch processing of Wikipedia articles with batch_size={batch_size}")
    
    current_batch = []
    titles = []
    chunks_list = []
    texts = []
    
    # Process articles until we have enough for a batch
    while len(current_batch) < batch_size:
        try:
            # Get next article
            title, article_text = next_article(wiki_iterator)
            if not title or not article_text:
                break
                
            logger.info(f"Processing article: {title}")
            
            try:
                # Tokenize article into chunks
                chunks = tokenize_article_into_chunks(
                    article_text,
                    tokenizer,
                    chunk_size=chunk_size,
                    max_chunks=min_chunks_per_article * 2,  # Need at least twice the min for non-overlapping pairs
                    skip_if_too_short=True
                )
                
                if len(chunks) < min_chunks_per_article:
                    logger.warning(f"Article '{title}' has too few chunks ({len(chunks)}), skipping")
                    continue
                
                # Add to our collected articles
                titles.append(title)
                chunks_list.append(chunks)
                texts.append(article_text)
                
                # Process each article into key-value pairs
                kv_pairs = process_article_for_batch(
                    title=title,
                    article_text=article_text,
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    min_chunks_per_article=min_chunks_per_article
                )
                
                # Add to current batch
                current_batch.extend(kv_pairs)
                
                # Yield batch if we have enough pairs
                if len(current_batch) >= batch_size:
                    # Create BatchedWikiArticle object for the collected articles
                    batched_article = BatchedWikiArticle(
                        titles=titles,
                        chunks_list=chunks_list,
                        texts=texts
                    )
                    
                    logger.info(f"Yielding batch of {len(current_batch)} key-value pairs from {len(titles)} articles")
                    yield current_batch[:batch_size]  # Only yield requested batch size
                    
                    # Keep any overflow for next batch
                    current_batch = current_batch[batch_size:]
                    
                    # Reset article collections if we've used all pairs
                    if not current_batch:
                        titles = []
                        chunks_list = []
                        texts = []
                
            except ValueError as e:
                # Article too short or other validation error
                logger.warning(f"Skipping article '{title}': {str(e)}")
                continue
                
        except StopIteration:
            # End of articles
            break
        except Exception as e:
            # Other error
            logger.error(f"Error in batch processing: {e}")
            continue
    
    # Yield any remaining pairs
    if current_batch:
        # Create final BatchedWikiArticle object
        batched_article = BatchedWikiArticle(
            titles=titles,
            chunks_list=chunks_list,
            texts=texts
        )
        
        logger.info(f"Yielding final batch of {len(current_batch)} key-value pairs from {len(titles)} articles")
        yield current_batch

def create_embedding(text: str) -> np.ndarray:
    """
    Create an embedding vector for the given text.
    
    Args:
        text: The text to create an embedding for
        
    Returns:
        Numpy array containing the embedding vector
    """
    # Use the batch_ada_embeddings function which handles rate limiting
    embeddings = batch_ada_embeddings([text])
    return embeddings[0] 