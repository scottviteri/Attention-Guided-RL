"""
Tests for the data module.
"""

import os
import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock, mock_open, ANY
import unittest
from pathlib import Path
import torch
import random
from typing import List, Dict, Any, Iterator

# Import the module to test
from src.data import (
    KeyValuePair,
    load_wikipedia_article,
    create_wiki_dataset_iterator,
    next_article,
    fetch_from_wikipedia_api,
    tokenize_article_into_chunks,
    create_key_value_pairs_from_chunks,
    process_article_text,
    process_article,
    process_articles,
    process_next_article,
    load_random_wikipedia_article,
    process_articles_stream
)


class TestWikiArticleProcessor(unittest.TestCase):
    def setUp(self):
        # Create mock tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens
        self.mock_tokenizer.decode.side_effect = lambda x: f"Text for tokens {x[:5]}..."
        self.mock_tokenizer.pad_token_id = 0
        
        # Mock article content
        self.article_content = "This is a test article content for unit testing."
    
    @patch('src.data.requests.get')
    def test_fetch_from_wikipedia_api(self, mock_get):
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "12345": {
                        "extract": "Test content from Wikipedia API"
                    }
                }
            }
        }
        mock_get.return_value = mock_response
        
        # Call the function
        result = fetch_from_wikipedia_api("Test Article")
        
        # Verify result
        self.assertEqual(result, "Test content from Wikipedia API")
        mock_get.assert_called_once()
    
    def test_tokenize_article_into_chunks(self):
        # Call the function
        chunks = tokenize_article_into_chunks(
            self.article_content,
            self.mock_tokenizer,
            chunk_size=20,
            max_chunks=3
        )
        
        # Verify results
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), 20)
        self.assertTrue(isinstance(chunks[0], torch.Tensor))
        self.mock_tokenizer.encode.assert_called_once_with(self.article_content, add_special_tokens=False)
    
    def test_create_key_value_pairs_from_chunks(self):
        # Create test chunks as torch tensors
        chunks = [
            torch.tensor(list(range(10)), dtype=torch.long),
            torch.tensor(list(range(10, 20)), dtype=torch.long),
            torch.tensor(list(range(20, 30)), dtype=torch.long),
            torch.tensor(list(range(30, 40)), dtype=torch.long),
            torch.tensor(list(range(40, 50)), dtype=torch.long),
            torch.tensor(list(range(50, 60)), dtype=torch.long)
        ]
        
        # Test with consecutive pairs (non-random) with overlapping mode
        pairs = create_key_value_pairs_from_chunks(
            chunks=chunks, 
            max_pairs=2, 
            non_overlapping=False,
            random_selection=False  # Use consecutive pairs from the beginning
        )
        
        # Verify results for consecutive pairs
        self.assertEqual(len(pairs), 2)
        self.assertTrue(torch.equal(pairs[0][0], chunks[0]))
        self.assertTrue(torch.equal(pairs[0][1], chunks[1]))
        self.assertTrue(torch.equal(pairs[1][0], chunks[1]))
        self.assertTrue(torch.equal(pairs[1][1], chunks[2]))
        
        # Test with consecutive pairs (non-random) with non-overlapping mode
        pairs_non_overlap = create_key_value_pairs_from_chunks(
            chunks=chunks, 
            max_pairs=2,
            non_overlapping=True,
            random_selection=False  # Use consecutive pairs from the beginning
        )
        
        # In non-overlapping mode, we should get the first two pairs: (0,1) and (2,3)
        self.assertEqual(len(pairs_non_overlap), 2)
        self.assertTrue(torch.equal(pairs_non_overlap[0][0], chunks[0]))
        self.assertTrue(torch.equal(pairs_non_overlap[0][1], chunks[1]))
        self.assertTrue(torch.equal(pairs_non_overlap[1][0], chunks[2]))
        self.assertTrue(torch.equal(pairs_non_overlap[1][1], chunks[3]))
        
        # Test with random selection of consecutive pairs
        with unittest.mock.patch('random.shuffle') as mock_shuffle:
            # Mock random.shuffle to reverse the list (to ensure we get a different order)
            def reverse_list(lst):
                lst.reverse()
            mock_shuffle.side_effect = reverse_list
            
            # Test with non-overlapping mode
            random_pairs = create_key_value_pairs_from_chunks(
                chunks=chunks,
                max_pairs=2,
                non_overlapping=True,
                random_selection=True
            )
            
            # With our mock shuffle, we should get the last two pairs: (2,3) and (0,1) in reverse order
            self.assertEqual(len(random_pairs), 2)
            # Since we reversed the list, we should get chunks (4,5) and (2,3)
            self.assertTrue(torch.equal(random_pairs[0][0], chunks[4]))
            self.assertTrue(torch.equal(random_pairs[0][1], chunks[5]))
            self.assertTrue(torch.equal(random_pairs[1][0], chunks[2]))
            self.assertTrue(torch.equal(random_pairs[1][1], chunks[3]))
    
    @patch('src.data.compute_embeddings')
    @patch('src.data.tokenize_article_into_chunks')
    def test_process_article_text(self, mock_tokenize_chunks, mock_compute_embeddings):
        mock_chunks = [
            torch.tensor(list(range(10)), dtype=torch.long),
            torch.tensor(list(range(10, 20)), dtype=torch.long),
            torch.tensor(list(range(20, 30)), dtype=torch.long),
            torch.tensor(list(range(30, 40)), dtype=torch.long)
        ]
        mock_tokenize_chunks.return_value = mock_chunks

        mock_compute_embeddings.return_value = [np.array([0.1, 0.2, 0.3]) for _ in range(2)]

        # Mock tokenizer decode method
        self.mock_tokenizer.decode.side_effect = lambda tokens: " ".join(map(str, tokens.tolist()))

        result = process_article_text(
            self.article_content,
            self.mock_tokenizer,
            max_pairs=2,
            should_compute_embeddings=True
        )

        self.assertEqual(result["title"], "Wikipedia Article")
        self.assertEqual(len(result["pairs"]), 2)
        self.assertEqual(len(result["key_embeddings"]), 2)

        for pair in result["pairs"]:
            self.assertTrue(isinstance(pair[0], str))
            self.assertTrue(isinstance(pair[1], str))

        mock_compute_embeddings.assert_called_once()
    
    @patch('src.data.next_article')
    @patch('src.data.AutoTokenizer')
    @patch('src.data.tokenize_article_into_chunks')
    @patch('src.data.compute_embeddings')
    def test_process_next_article(self, mock_compute_embeddings, mock_tokenize_chunks, mock_tokenizer_class, mock_next_article):
        # Mock chunks as PyTorch tensors - need 4 chunks to ensure 2 pairs with non_overlapping=True
        mock_chunks = [
            torch.tensor(list(range(10)), dtype=torch.long),
            torch.tensor(list(range(10, 20)), dtype=torch.long),
            torch.tensor(list(range(20, 30)), dtype=torch.long),
            torch.tensor(list(range(30, 40)), dtype=torch.long)
        ]
        mock_tokenize_chunks.return_value = mock_chunks
        
        # Mock tokenizer
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        
        # Mock article
        mock_next_article.return_value = ("Test Title", self.article_content)
        
        # Mock compute_embeddings to return fake embeddings
        mock_compute_embeddings.return_value = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6])
        ]
        
        # Call the function
        result = process_next_article(MagicMock(), max_pairs=2)
        
        # Verify results
        self.assertEqual(result["title"], "Test Title")
        self.assertTrue(isinstance(result["pairs"], list))
        self.assertTrue(len(result["pairs"]) > 0)
        self.assertTrue(isinstance(result["key_embeddings"], dict))
        
        # Verify compute_embeddings was called
        mock_compute_embeddings.assert_called_once()
    
    @patch('src.data.process_article')
    @patch('src.data.fetch_from_wikipedia_api')
    @patch('src.data.next_article')
    @patch('src.data.create_wiki_dataset_iterator')
    @patch('src.data.AutoTokenizer')
    def test_load_wikipedia_article(self, mock_tokenizer_class, mock_create_iterator, mock_next_article, mock_fetch_api, mock_process_article):
        """Test loading a Wikipedia article by title."""
        # Set up mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock fetch_from_wikipedia_api to return an article
        mock_fetch_api.return_value = "This is the article content."
        
        # Mock process_article_for_batch to return mock pairs
        mock_process_article.return_value = [
            KeyValuePair(
                key_tokens=torch.tensor([1, 2, 3]),
                value_tokens=torch.tensor([4, 5, 6]),
                key_embedding=np.array([0.1, 0.2, 0.3]),
                key_id="key1"
            )
        ]
        
        # Call the function
        result = load_wikipedia_article(title="Test Article")
        
        # Verify result
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], KeyValuePair)
        
        # Verify mocks were called correctly
        mock_fetch_api.assert_called_once_with("Test Article")
        mock_process_article.assert_called_once()
    
    @patch('src.data.load_dataset')
    def test_real_wikipedia_dataset(self, mock_load_dataset):
        """Test with a simulated Wikipedia dataset"""
        # Create a mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = iter([
            {"title": "Test Article", "text": "Test content"}
        ])
        mock_load_dataset.return_value = mock_dataset
        
        # Create an iterator from the mock dataset
        iterator = create_wiki_dataset_iterator(streaming=True)
        
        # Get the first article
        title, content = next_article(iterator)
        
        # Verify we got something
        self.assertEqual(title, "Test Article")
        self.assertEqual(content, "Test content")
        
        # Process just one tokenization step to verify it works
        tokenizer = self.mock_tokenizer
        tokenizer.encode.return_value = list(range(1000))  # simulate real tokens
        
        chunks = tokenize_article_into_chunks(
            content,
            tokenizer,
            chunk_size=20,
            max_chunks=3
        )
        
        # Verify chunks
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), 20)
        self.assertTrue(isinstance(chunks[0], torch.Tensor))

    def test_tokenize_article_too_short(self):
        # Mock a very short article
        short_article = "Short text"
        self.mock_tokenizer.encode.return_value = list(range(10))  # Only 10 tokens
        
        # Test with skip_if_too_short=True (default)
        with self.assertRaises(ValueError):
            # Try to get too many chunks from a short article
            list(tokenize_article_into_chunks(
                short_article,
                self.mock_tokenizer,
                chunk_size=20,
                max_chunks=3
            ))
        
        # Test with skip_if_too_short=False
        chunks = tokenize_article_into_chunks(
            short_article,
            self.mock_tokenizer,
            chunk_size=5,
            max_chunks=3,
            skip_if_too_short=False
        )
        
        # Should only get 2 chunks, not 3
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 5)
        self.assertTrue(isinstance(chunks[0], torch.Tensor))
    
    @patch('src.data.process_article')
    @patch('src.data.next_article')
    def test_process_articles_stream(self, mock_next_article, mock_process_article):
        """Test processing a stream of articles."""
        # Configure mock next_article to yield two articles then raise StopIteration
        mock_next_article.side_effect = [
            ("Article 1", "Content 1"),
            ("Article 2", "Content 2"),
            StopIteration()
        ]
        
        # Configure mock process_article to return different pairs for each article
        article1_pairs = [
            KeyValuePair(
                key_tokens=torch.tensor([1, 2, 3]),
                value_tokens=torch.tensor([4, 5, 6]),
                key_embedding=np.array([0.1, 0.2, 0.3]),
                key_id="key1"
            ),
            KeyValuePair(
                key_tokens=torch.tensor([7, 8, 9]),
                value_tokens=torch.tensor([10, 11, 12]),
                key_embedding=np.array([0.4, 0.5, 0.6]),
                key_id="key2"
            )
        ]
        
        article2_pairs = [
            KeyValuePair(
                key_tokens=torch.tensor([13, 14, 15]),
                value_tokens=torch.tensor([16, 17, 18]),
                key_embedding=np.array([0.7, 0.8, 0.9]),
                key_id="key3"
            )
        ]
        
        mock_process_article.side_effect = [article1_pairs, article2_pairs]
        
        # Set up mock tokenizer
        mock_tokenizer = MagicMock()
        
        # Call the function
        result = list(process_articles_stream(
            wiki_iterator=iter([]),  # Will be ignored due to mock
            tokenizer=mock_tokenizer,
            chunk_size=64,
            min_chunks_per_article=10,
            max_pairs=5
        ))
        
        # Verify the result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], article1_pairs)
        self.assertEqual(result[1], article2_pairs)
        
        # Verify that process_article was called twice with the correct arguments
        self.assertEqual(mock_process_article.call_count, 2)
        mock_process_article.assert_any_call(
            title="Article 1",
            article_text="Content 1",
            tokenizer=mock_tokenizer,
            chunk_size=64,
            min_chunks_per_article=10
        )

    @patch('src.data.process_articles_stream')
    def test_process_articles(self, mock_process_articles_stream):
        """Test batch processing of articles."""
        # Mock process_articles_stream to yield two batches
        article1_pairs = [
            KeyValuePair(
                key_tokens=torch.tensor([1, 2, 3]),
                value_tokens=torch.tensor([4, 5, 6]),
                key_embedding=np.array([0.1, 0.2, 0.3]),
                key_id="key1"
            ),
            KeyValuePair(
                key_tokens=torch.tensor([7, 8, 9]),
                value_tokens=torch.tensor([10, 11, 12]),
                key_embedding=np.array([0.4, 0.5, 0.6]),
                key_id="key2"
            )
        ]
        
        article2_pairs = [
            KeyValuePair(
                key_tokens=torch.tensor([13, 14, 15]),
                value_tokens=torch.tensor([16, 17, 18]),
                key_embedding=np.array([0.7, 0.8, 0.9]),
                key_id="key3"
            )
        ]
        
        # Set up the mock to yield two lists of pairs
        mock_process_articles_stream.side_effect = [[article1_pairs], [article2_pairs], []]
        
        # Call the function
        result = process_articles(
            wiki_iterator=iter([]),  # Will be ignored due to mock
            num_articles=5,
            tokenizer_name="gpt2",
            key_token_count=64,
            batch_size=2
        )
        
        # Verify the results
        assert len(result) == 2  # We get 2 batches
        assert result[0] == article1_pairs
        assert result[1] == article2_pairs
        
        # Verify the calls to process_articles_stream
        assert mock_process_articles_stream.call_count >= 2
        # The first call should have these parameters
        mock_process_articles_stream.assert_any_call(
            wiki_iterator=ANY,
            tokenizer=ANY,
            chunk_size=64,  # This matches key_token_count
            min_chunks_per_article=10,  # Default value
            max_pairs=10    # Actual value from config
        )
    
    @patch('src.data.process_article')
    @patch('src.data.next_article')
    @patch('src.data.create_wiki_dataset_iterator')
    @patch('src.data.AutoTokenizer')
    def test_load_random_wikipedia_article(self, mock_tokenizer_class, mock_create_iterator, mock_next_article, mock_process_article):
        """Test loading a random Wikipedia article."""
        # Set up mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock create_wiki_dataset_iterator to return a mock iterator
        mock_iter = MagicMock()
        mock_create_iterator.return_value = mock_iter
        
        # Mock next_article to return a title and content
        mock_next_article.return_value = ("Random Article", "This is a random article content.")
        
        # Mock process_article to return mock pairs
        mock_process_article.return_value = [
            KeyValuePair(
                key_tokens=torch.tensor([1, 2, 3]),
                value_tokens=torch.tensor([4, 5, 6]),
                key_embedding=np.array([0.1, 0.2, 0.3]),
                key_id="key1"
            )
        ]
        
        # Call the function
        result = load_random_wikipedia_article()
        
        # Verify result
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], KeyValuePair)
        
        # Verify mocks were called correctly
        mock_create_iterator.assert_called_once()
        mock_next_article.assert_called_once_with(mock_iter)
        mock_process_article.assert_called_once()

    @patch('src.data.compute_embeddings')
    @patch('src.data.tokenize_article_into_chunks')
    def test_process_article(self, mock_tokenize_chunks, mock_compute_embeddings):
        """Test the process_article function."""
        # Set up mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.side_effect = lambda x: f"Text for tokens {x[:5]}..."
        
        # Mock tokenize_article_into_chunks function - return 10 chunks
        mock_chunks = [torch.tensor(list(range(i, i+10)), dtype=torch.long) for i in range(10)]
        mock_tokenize_chunks.return_value = mock_chunks
        
        # Mock compute_embeddings function - return 5 embeddings
        # Each pair uses 2 chunks, so with 10 chunks we get 5 pairs
        mock_compute_embeddings.return_value = [np.array([0.1, 0.2, 0.3])] * 5
        
        # Call the function with actual parameters
        title = "Test Article"
        article_text = "This is a test article with enough content to create multiple chunks."
        result = process_article(
            title=title,
            article_text=article_text,
            tokenizer=mock_tokenizer,
            chunk_size=64,
            min_chunks_per_article=2
        )
        
        # Verify the result
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)  # Should create 5 pairs
        
        # Each pair should be a KeyValuePair object
        for pair in result:
            self.assertIsInstance(pair, KeyValuePair)
            self.assertTrue(isinstance(pair.key_tokens, torch.Tensor))
            self.assertTrue(isinstance(pair.value_tokens, torch.Tensor))
            self.assertTrue(isinstance(pair.key_embedding, np.ndarray))
        
        # Verify tokenizer and embedding calls
        mock_tokenize_chunks.assert_called_once()
        # Should call compute_embeddings once with all key texts
        mock_compute_embeddings.assert_called_once()
        
        # Test with a short article
        mock_tokenize_chunks.reset_mock()
        mock_compute_embeddings.reset_mock()
        
        # Configure mock to return only 1 chunk (too short)
        mock_tokenize_chunks.return_value = [torch.tensor([1, 2, 3], dtype=torch.long)]
        
        # Call with a short article
        short_result = process_article(
            title="Short Article",
            article_text="Too short.",
            tokenizer=mock_tokenizer,
            chunk_size=64,
            min_chunks_per_article=2
        )
        
        # Should return an empty list if there aren't enough chunks
        self.assertEqual(len(short_result), 0) 