"""
Functions for loading and processing Wikipedia articles and key-value pairs.
"""
import requests
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Iterator
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from dataclasses import dataclass, field
import random
import itertools

from src.config import (
    WIKI_ARTICLE_TITLE, 
    MAX_PAIRS, 
    KEY_TOKEN_COUNT, 
    MODEL_NAME
)
from src.embeddings import compute_embeddings

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
        key: The key string (optional)
        value: The value string (optional)
    """
    key_tokens: torch.Tensor
    value_tokens: torch.Tensor
    key_embedding: np.ndarray
    key_id: str = ""
    key: str = ""
    value: str = ""

@dataclass(frozen=True)
class WikiArticle:
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

def process_article(
    title: str,
    article_text: str,
    tokenizer,
    model,
    chunk_size: int = KEY_TOKEN_COUNT,
    max_pairs: int = MAX_PAIRS,
    non_overlapping: bool = True,
    random_selection: bool = True
) -> List[KeyValuePair]:
    """
    Process a single article into key-value pairs with embeddings.
    
    Args:
        title: Article title
        article_text: Text content of the article
        tokenizer: Tokenizer to use for processing
        model: Language model for computing embeddings
        chunk_size: Size of each chunk in tokens
        max_pairs: Maximum number of pairs to create
        non_overlapping: If True, create non-overlapping pairs
        random_selection: If True, randomly select which pairs to use
        
    Returns:
        List of KeyValuePair objects
    """
    # Get chunks
    try:
        chunks = tokenize_article_into_chunks(
            article_text,
            tokenizer,
            chunk_size=chunk_size,
            max_chunks=max_pairs * 2 if non_overlapping else max_pairs + 1
        )
        
        # If we don't have enough chunks, return empty list
        if (non_overlapping and len(chunks) < 2) or len(chunks) < 1:
            logger.warning(f"Article '{title}' has insufficient chunks ({len(chunks)})")
            return []
        
        # Create pairs
        chunk_pairs = create_key_value_pairs_from_chunks(
            chunks=chunks,
            max_pairs=max_pairs,
            non_overlapping=non_overlapping,
            random_selection=random_selection
        )
        
        if not chunk_pairs:
            logger.warning(f"No pairs could be created from article '{title}'")
            return []
        
        # Extract key texts for batch embedding
        key_texts = []
        for key_tokens, _ in chunk_pairs:
            key_text = tokenizer.decode(key_tokens)
            key_texts.append(key_text)
        
        # Get embeddings for all keys at once
        key_embeddings = compute_embeddings(key_texts, model=model, are_queries=False)
        
        # Create KeyValuePair objects
        result_pairs = []
        for idx, ((key_tokens, value_tokens), embedding) in enumerate(zip(chunk_pairs, key_embeddings)):
            key_text = key_texts[idx]
            value_text = tokenizer.decode(value_tokens)
            
            pair = KeyValuePair(
                key_id=f"key_{idx}",
                key_tokens=key_tokens,
                value_tokens=value_tokens,
                key_embedding=embedding,
                key=key_text,
                value=value_text
            )
            result_pairs.append(pair)
        
        logger.info(f"Created {len(result_pairs)} pairs from article '{title}'")
        return result_pairs
        
    except ValueError as e:
        logger.warning(f"Article '{title}' processing error: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error processing article '{title}': {e}")
        return []

def process_articles_stream(
    wiki_iterator, 
    tokenizer,
    model,
    chunk_size: int = KEY_TOKEN_COUNT,
    max_pairs: int = MAX_PAIRS,
    non_overlapping: bool = True,
    random_selection: bool = True,
    batch_size: int = 1  # New parameter for batching
) -> Iterator[List[KeyValuePair]]:
    """
    Process a stream of Wikipedia articles, yielding KeyValuePair lists.
    
    Args:
        wiki_iterator: Iterator yielding Wikipedia articles
        tokenizer: Tokenizer to use for processing text
        model: Language model to use for computing embeddings
        chunk_size: Size of each chunk in tokens
        max_pairs: Maximum number of key-value pairs to create per article
        non_overlapping: Whether to create non-overlapping pairs
        random_selection: Whether to randomly select pairs
        batch_size: Number of articles to process in a batch (1 = no batching)
        
    Yields:
        List of KeyValuePair objects for each processed article
    """
    logger.info(f"Processing Wikipedia articles stream with chunk_size={chunk_size}")
    
    while True:
        try:
            if batch_size <= 1:
                # Single article processing (original behavior)
                title, article_text = next_article(wiki_iterator)
                if not title or not article_text:
                    break
                    
                pairs = process_article(
                    title=title,
                    article_text=article_text,
                    tokenizer=tokenizer,
                    model=model,
                    chunk_size=chunk_size,
                    max_pairs=max_pairs,
                    non_overlapping=non_overlapping,
                    random_selection=random_selection
                )
                
                if pairs:
                    yield pairs
            else:
                # Batch processing
                batch_articles = []
                for _ in range(batch_size):
                    try:
                        title, article_text = next_article(wiki_iterator)
                        if title and article_text:
                            batch_articles.append((title, article_text))
                    except StopIteration:
                        break
                
                if not batch_articles:
                    break
                
                # Process each article in the batch and yield non-empty results
                for title, article_text in batch_articles:
                    pairs = process_article(
                        title=title,
                        article_text=article_text,
                        tokenizer=tokenizer,
                        model=model,
                        chunk_size=chunk_size,
                        max_pairs=max_pairs,
                        non_overlapping=non_overlapping,
                        random_selection=random_selection
                    )
                    
                    if pairs:
                        yield pairs
                
        except StopIteration:
            break
        except Exception as e:
            logger.error(f"Error processing articles stream: {e}")
            continue

def load_wikipedia_article(
    title: str = None,  # None = random article
    tokenizer_name: str = MODEL_NAME,
    device: str = None,
    max_pairs: int = MAX_PAIRS,
    non_overlapping: bool = True,
    random_selection: bool = True
) -> List[KeyValuePair]:
    """
    Load a Wikipedia article and convert it to KeyValuePair objects.
    If title is None, loads a random article.
    
    Args:
        title: Title of the Wikipedia article (None for random)
        tokenizer_name: Name of the tokenizer model to use
        device: Device to load the model on (cpu or cuda)
        max_pairs: Maximum number of key-value pairs
        non_overlapping: Whether to create non-overlapping pairs
        random_selection: Whether to randomly select pairs
        
    Returns:
        List of KeyValuePair objects
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    from src.model import load_language_model
    model = load_language_model(device=device)
    
    if title:
        # Try to fetch from Wikipedia API first
        article_text = fetch_from_wikipedia_api(title)
        
        if article_text:
            return process_article(
                title=title,
                article_text=article_text,
                tokenizer=tokenizer,
                model=model,
                max_pairs=max_pairs,
                non_overlapping=non_overlapping,
                random_selection=random_selection
            )
    
    # If title is None or API fetch failed, use the dataset
    wiki_iterator = create_wiki_dataset_iterator()
    
    while True:
        try:
            curr_title, curr_text = next_article(wiki_iterator)
            if not curr_title or not curr_text:
                break
                
            # If searching for specific title, check match
            if title and curr_title.lower() != title.lower():
                continue
                
            pairs = process_article(
                title=curr_title,
                article_text=curr_text,
                tokenizer=tokenizer,
                model=model,
                max_pairs=max_pairs,
                non_overlapping=non_overlapping,
                random_selection=random_selection
            )
            
            if pairs:
                return pairs
                
        except (StopIteration, Exception):
            break
    
    # If no article found/processed
    logger.error(f"Failed to load article{f' with title {title}' if title else ''}")
    return [] 