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
import itertools

# Import the module to test
from src.data import (
    KeyValuePair,
    load_wikipedia_article,
    create_wiki_dataset_iterator,
    next_article,
    fetch_from_wikipedia_api,
    tokenize_article_into_chunks,
    create_key_value_pairs_from_chunks,
    process_article,
    process_articles_stream
)

# Note: We've removed process_article_text, process_next_article and load_random_wikipedia_article
# from imports as they have been removed from the codebase


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
    @patch('src.data.create_key_value_pairs_from_chunks')
    @patch('src.data.tokenize_article_into_chunks')
    def test_process_article(self, mock_tokenize_chunks, mock_create_pairs, mock_compute_embeddings):
        """Test the enhanced process_article function."""
        # Mock chunks
        mock_chunks = [
            torch.tensor(list(range(10)), dtype=torch.long),
            torch.tensor(list(range(10, 20)), dtype=torch.long),
            torch.tensor(list(range(20, 30)), dtype=torch.long),
            torch.tensor(list(range(30, 40)), dtype=torch.long)
        ]
        mock_tokenize_chunks.return_value = mock_chunks
        
        # Mock the key-value pairs
        mock_pairs = [
            (mock_chunks[0], mock_chunks[1]),
            (mock_chunks[2], mock_chunks[3])
        ]
        mock_create_pairs.return_value = mock_pairs
        
        # Mock embeddings
        mock_embeddings = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6])
        ]
        mock_compute_embeddings.return_value = mock_embeddings
        
        # Mock tokenizer decode
        self.mock_tokenizer.decode.side_effect = ["Key text 1", "Key text 2", "Value text 1", "Value text 2"]
        
        # Create a mock model
        mock_model = MagicMock()
        
        # Call the function with various parameters
        result = process_article(
            title="Test Article",
            article_text=self.article_content,
            tokenizer=self.mock_tokenizer,
            model=mock_model,
            max_pairs=2,
            non_overlapping=True,
            random_selection=False
        )
        
        # Verify results
        self.assertEqual(len(result), 2)
        for pair in result:
            self.assertIsInstance(pair, KeyValuePair)
            
        # Verify each pair has the correct structure and data
        self.assertEqual(result[0].key_id, "key_0")
        self.assertTrue(torch.equal(result[0].key_tokens, mock_chunks[0]))
        self.assertTrue(torch.equal(result[0].value_tokens, mock_chunks[1]))
        np.testing.assert_array_equal(result[0].key_embedding, mock_embeddings[0])
        
        # Verify calls to mocked functions
        mock_tokenize_chunks.assert_called_once()
        mock_create_pairs.assert_called_once()
        mock_compute_embeddings.assert_called_once()
    
    @patch('src.data.process_article')
    @patch('src.data.fetch_from_wikipedia_api')
    @patch('src.data.next_article')
    @patch('src.data.create_wiki_dataset_iterator')
    @patch('src.data.AutoTokenizer')
    def test_load_wikipedia_article(self, mock_tokenizer_class, mock_create_iterator, mock_next_article, mock_fetch_api, mock_process_article):
        """Test loading a Wikipedia article by title or randomly."""
        # Set up mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock fetch_from_wikipedia_api to return an article
        mock_fetch_api.return_value = "This is the article content."
        
        # Mock process_article to return mock pairs
        expected_pairs = [
            KeyValuePair(
                key_tokens=torch.tensor([1, 2, 3]),
                value_tokens=torch.tensor([4, 5, 6]),
                key_embedding=np.array([0.1, 0.2, 0.3]),
                key_id="key1"
            )
        ]
        mock_process_article.return_value = expected_pairs
        
        # Test with specific title
        result = load_wikipedia_article(title="Test Article")
        
        # Verify result
        self.assertEqual(result, expected_pairs)
        
        # Verify mocks were called correctly
        mock_fetch_api.assert_called_with("Test Article")
        mock_process_article.assert_called()
        
        # Reset mocks
        mock_fetch_api.reset_mock()
        mock_process_article.reset_mock()
        mock_next_article.reset_mock()
        
        # Test with random article (title=None)
        mock_fetch_api.return_value = ""  # Force it to use dataset
        mock_next_article.return_value = ("Random Title", "Random content")
        
        result = load_wikipedia_article(title=None)
        
        # Verify result
        self.assertEqual(result, expected_pairs)
        
        # API fetch should not be called for random article
        mock_fetch_api.assert_not_called()
        mock_next_article.assert_called()
        mock_process_article.assert_called()
    
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
        
        # Create a mock model
        mock_model = MagicMock()
        
        # Call the function
        result = list(process_articles_stream(
            wiki_iterator=iter([]),  # Will be ignored due to mock
            tokenizer=mock_tokenizer,
            model=mock_model,
            chunk_size=64,
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
            model=mock_model,
            chunk_size=64,
            max_pairs=5,
            non_overlapping=True,
            random_selection=True
        )
    
    @patch('src.data.process_article')
    @patch('src.data.next_article')
    def test_process_articles_stream_batching(self, mock_next_article, mock_process_article):
        """Test processing a stream of articles with batching."""
        # Configure mock to return 3 articles then StopIteration
        mock_next_article.side_effect = [
            ("Article 1", "Content 1"),
            ("Article 2", "Content 2"),
            ("Article 3", "Content 3"),
            StopIteration()
        ]
        
        # Configure mock_process_article to return different pairs for each article
        article_pairs = [
            [MagicMock()],  # Article 1
            [],  # Article 2 (empty to test filtering)
            [MagicMock(), MagicMock()]  # Article 3
        ]
        mock_process_article.side_effect = article_pairs
        
        # Call the function with batch_size=2
        result = list(process_articles_stream(
            wiki_iterator=iter([]),
            tokenizer=MagicMock(),
            model=MagicMock(),
            batch_size=2
        ))
        
        # We should get 2 results (Article 1 and Article 3) since Article 2 returned empty
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], article_pairs[0])
        self.assertEqual(result[1], article_pairs[2])
        
        # Verify process_article was called for all 3 articles
        self.assertEqual(mock_process_article.call_count, 3)

    def test_process_article_error_handling(self):
        """Test that process_article handles errors gracefully."""
        # Test with invalid article text (None)
        with patch('src.data.tokenize_article_into_chunks') as mock_tokenize:
            mock_tokenize.side_effect = ValueError("Test error")
            
            result = process_article(
                title="Test",
                article_text="Some text",
                tokenizer=self.mock_tokenizer,
                model=MagicMock()
            )
            self.assertEqual(result, [])
        
        # Test with article text that would cause tokenization error
        with patch('src.data.tokenize_article_into_chunks') as mock_tokenize:
            mock_tokenize.side_effect = Exception("Tokenization error")
            
            result = process_article(
                title="Test",
                article_text="Some text",
                tokenizer=self.mock_tokenizer,
                model=MagicMock()
            )
            self.assertEqual(result, []) 