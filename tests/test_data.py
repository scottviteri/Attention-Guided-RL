"""
Tests for the data module.
"""

import os
import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import unittest
from pathlib import Path
import torch
import random

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
    process_article_for_batch,
    process_next_article,
    load_random_wikipedia_article,
    process_articles_stream,
    batch_process_articles
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
    
    @patch('src.data.batch_compute_embeddings')
    @patch('src.data.tokenize_article_into_chunks')
    def test_process_article_text(self, mock_tokenize_chunks, mock_batch_embeddings):
        # Mock chunks as PyTorch tensors - need 4 chunks to ensure 2 pairs with non_overlapping=True
        mock_chunks = [
            torch.tensor(list(range(10)), dtype=torch.long),
            torch.tensor(list(range(10, 20)), dtype=torch.long),
            torch.tensor(list(range(20, 30)), dtype=torch.long),
            torch.tensor(list(range(30, 40)), dtype=torch.long)
        ]
        mock_tokenize_chunks.return_value = mock_chunks
        
        # Mock embeddings
        mock_batch_embeddings.return_value = [np.array([0.1, 0.2, 0.3]) for _ in range(2)]
        
        # Call the function
        result = process_article_text(
            self.article_content,
            self.mock_tokenizer,
            max_pairs=2,
            compute_embeddings=True
        )
        
        # Verify results
        self.assertEqual(result["title"], "Wikipedia Article")
        self.assertEqual(len(result["pairs"]), 2)
        self.assertEqual(len(result["key_embeddings"]), 2)
        # Verify each pair is a tuple of strings (decoded tokens)
        for pair in result["pairs"]:
            self.assertTrue(isinstance(pair[0], str))
            self.assertTrue(isinstance(pair[1], str))
        mock_batch_embeddings.assert_called_once()
    
    @patch('src.data.next_article')
    @patch('src.data.AutoTokenizer')
    @patch('src.data.tokenize_article_into_chunks')
    def test_process_next_article(self, mock_tokenize_chunks, mock_tokenizer_class, mock_next_article):
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
        
        # Call the function
        result = process_next_article(MagicMock(), max_pairs=2)
        
        # Verify results
        self.assertEqual(result["title"], "Test Title")
        self.assertEqual(len(result["pairs"]), 2)
        # Verify each pair is a tuple of strings (decoded tokens)
        for pair in result["pairs"]:
            self.assertTrue(isinstance(pair[0], str))
            self.assertTrue(isinstance(pair[1], str))
    
    @patch('src.data.process_article_for_batch')
    @patch('src.data.fetch_from_wikipedia_api')
    @patch('src.data.next_article')
    @patch('src.data.create_wiki_dataset_iterator')
    @patch('src.data.AutoTokenizer')
    def test_load_wikipedia_article(self, mock_tokenizer_class, mock_create_iterator, mock_next_article, mock_fetch_api, mock_process_article_for_batch):
        """Test the load_wikipedia_article function."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock article fetch from API directly returns content instead of empty string
        mock_fetch_api.return_value = "Test article content from API"
        
        # Mock article processing with PyTorch tensor
        mock_kv_pair = KeyValuePair(
            key_id="key_0",
            key_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
            value_tokens=torch.tensor([4, 5, 6], dtype=torch.long),
            key_embedding=np.array([0.1, 0.2, 0.3])
        )
        mock_process_article_for_batch.return_value = [mock_kv_pair]
        
        # Load article
        result = load_wikipedia_article(title="test article", max_pairs=10)
        
        # Verify
        mock_fetch_api.assert_called_once_with("test article")
        mock_create_iterator.assert_not_called()  # Should not be called since API fetch succeeded
        mock_next_article.assert_not_called()  # Should not be called since API fetch succeeded
        
        # Should be called once for API path
        mock_process_article_for_batch.assert_called_once()
        
        # Check the result
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], KeyValuePair)
        self.assertTrue(isinstance(result[0].key_tokens, torch.Tensor))
        self.assertTrue(isinstance(result[0].value_tokens, torch.Tensor))
    
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
    
    @patch('src.data.process_article_for_batch')
    @patch('src.data.next_article')
    def test_process_articles_stream(self, mock_next_article, mock_process_article_for_batch):
        """Test the process_articles_stream function."""
        # Configure mock to return a sequence of articles
        mock_next_article.side_effect = [
            # First article
            ("Article 1", "This is a test article content"),
            # Second article
            ("Article 2", "Another article with content"),
            # End of iteration
            StopIteration()
        ]
        
        # Mock article processing with PyTorch tensors
        mock_kv_pair1 = KeyValuePair(
            key_id="key_0",
            key_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
            value_tokens=torch.tensor([4, 5, 6], dtype=torch.long),
            key_embedding=np.array([0.1, 0.2, 0.3])
        )
        mock_kv_pair2 = KeyValuePair(
            key_id="key_1",
            key_tokens=torch.tensor([7, 8, 9], dtype=torch.long),
            value_tokens=torch.tensor([10, 11, 12], dtype=torch.long),
            key_embedding=np.array([0.4, 0.5, 0.6])
        )
        
        # First article returns two pairs, second article returns one pair
        mock_process_article_for_batch.side_effect = [
            [mock_kv_pair1, mock_kv_pair2],  # First article
            [mock_kv_pair2]  # Second article
        ]
        
        # Create a mock wiki iterator and tokenizer
        mock_iterator = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Call the function
        stream = process_articles_stream(
            mock_iterator,
            mock_tokenizer,
            chunk_size=20,
            min_chunks_per_article=1
        )
        
        # Get the first article from the stream
        result1 = next(stream)
        
        # Verify the first article result
        self.assertIsInstance(result1, list)
        self.assertEqual(len(result1), 2)
        self.assertIsInstance(result1[0], KeyValuePair)
        self.assertEqual(result1[0].key_id, "key_0")
        self.assertTrue(isinstance(result1[0].key_tokens, torch.Tensor))
        self.assertTrue(isinstance(result1[0].value_tokens, torch.Tensor))
        
        # Get the second article from the stream
        result2 = next(stream)
        
        # Verify the second article result
        self.assertIsInstance(result2, list)
        self.assertEqual(len(result2), 1)
        self.assertIsInstance(result2[0], KeyValuePair)
        self.assertEqual(result2[0].key_id, "key_1")
        self.assertTrue(isinstance(result2[0].key_tokens, torch.Tensor))
        self.assertTrue(isinstance(result2[0].value_tokens, torch.Tensor))
        
        # Verify StopIteration at end of stream
        with self.assertRaises(StopIteration):
            next(stream)
            
        # Verify the call sequence - called for first article, second article, and StopIteration attempt
        self.assertEqual(mock_next_article.call_count, 3)

    @patch('src.data.process_articles_stream')
    def test_batch_process_articles(self, mock_process_articles_stream):
        """Test the batch_process_articles function"""
        # Create mock KeyValuePair objects with PyTorch tensors
        kv_pair1 = KeyValuePair(
            key_id="key_0",
            key_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
            value_tokens=torch.tensor([4, 5, 6], dtype=torch.long),
            key_embedding=np.array([0.1, 0.2, 0.3])
        )
        kv_pair2 = KeyValuePair(
            key_id="key_1",
            key_tokens=torch.tensor([7, 8, 9], dtype=torch.long),
            value_tokens=torch.tensor([10, 11, 12], dtype=torch.long),
            key_embedding=np.array([0.4, 0.5, 0.6])
        )
        kv_pair3 = KeyValuePair(
            key_id="key_2",
            key_tokens=torch.tensor([13, 14, 15], dtype=torch.long),
            value_tokens=torch.tensor([16, 17, 18], dtype=torch.long),
            key_embedding=np.array([0.7, 0.8, 0.9])
        )
        
        # Configure mock to return a stream of articles with KeyValuePair objects
        mock_process_articles_stream.return_value = iter([
            # First article with two pairs
            [kv_pair1, kv_pair2],
            # Second article with one pair
            [kv_pair3],
            # No more articles - this is crucial to avoid infinite loops
            []
        ])
        
        # Create mock iterator and tokenizer
        mock_iterator = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Mock the next_article function to return appropriate articles and then stop iteration
        with patch('src.data.next_article') as mock_next_article:
            # Configure mock to return two articles and then stop
            mock_next_article.side_effect = [
                ("Article 1", "Content 1"),
                ("Article 2", "Content 2"),
                StopIteration()
            ]
            
            # Also mock tokenize_article_into_chunks to return valid chunks
            with patch('src.data.tokenize_article_into_chunks') as mock_tokenize:
                mock_tokenize.return_value = [torch.tensor([1, 2, 3]) for _ in range(20)]  # 20 chunks
                
                # Mock process_article_for_batch to return our predefined KeyValuePair objects
                with patch('src.data.process_article_for_batch') as mock_process:
                    mock_process.side_effect = [
                        [kv_pair1, kv_pair2],  # First article
                        [kv_pair3]  # Second article
                    ]
                    
                    # Call the function
                    batch_gen = batch_process_articles(
                        wiki_iterator=mock_iterator,
                        tokenizer=mock_tokenizer,
                        batch_size=2,
                        chunk_size=10,
                        min_chunks_per_article=1
                    )
                    
                    # Get the first batch (should contain the first two pairs)
                    batch1 = next(batch_gen)
                    
                    # Verify first batch
                    self.assertIsInstance(batch1, list)
                    self.assertEqual(len(batch1), 2)
                    self.assertIsInstance(batch1[0], KeyValuePair)
                    self.assertEqual(batch1[0].key_id, "key_0")
                    self.assertEqual(batch1[1].key_id, "key_1")
                    self.assertTrue(isinstance(batch1[0].key_tokens, torch.Tensor))
                    self.assertTrue(isinstance(batch1[0].value_tokens, torch.Tensor))
                    
                    # Get the second batch (should contain the third pair)
                    batch2 = next(batch_gen)
                    
                    # Verify second batch
                    self.assertIsInstance(batch2, list)
                    self.assertEqual(len(batch2), 1)
                    self.assertIsInstance(batch2[0], KeyValuePair)
                    self.assertEqual(batch2[0].key_id, "key_2")
                    self.assertTrue(isinstance(batch2[0].key_tokens, torch.Tensor))
                    self.assertTrue(isinstance(batch2[0].value_tokens, torch.Tensor))
                    
                    # Verify there are no more batches
                    with self.assertRaises(StopIteration):
                        next(batch_gen)
                    
                    # Verify the calls were made with the expected arguments
                    mock_next_article.assert_called()
                    mock_tokenize.assert_called()
                    mock_process.assert_called()

    @patch('src.data.process_article_for_batch')
    @patch('src.data.next_article')
    @patch('src.data.create_wiki_dataset_iterator')
    @patch('src.data.AutoTokenizer')
    def test_load_random_wikipedia_article(self, mock_tokenizer_class, mock_create_iterator, mock_next_article, mock_process_article_for_batch):
        """Test the load_random_wikipedia_article function."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock dataset search
        mock_iterator = MagicMock()
        mock_create_iterator.return_value = mock_iterator
        
        # Mock finding articles
        mock_next_article.side_effect = [
            ("Short Article", "Short text"),  # First article is too short
            ("Good Article", "This is a good article with enough content"),  # Second article is good
        ]
        
        # Mock article processing with PyTorch tensors
        # First call returns empty list (too short), second call returns a list of KeyValuePair objects
        mock_kv_pair = KeyValuePair(
            key_id="key_0",
            key_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
            value_tokens=torch.tensor([4, 5, 6], dtype=torch.long),
            key_embedding=np.array([0.1, 0.2, 0.3])
        )
        mock_process_article_for_batch.side_effect = [
            [],  # First article is too short
            [mock_kv_pair]  # Second article is good
        ]
        
        # Load random article
        result = load_random_wikipedia_article(max_pairs=10)
        
        # Verify
        mock_create_iterator.assert_called_once()
        self.assertEqual(mock_next_article.call_count, 2)
        self.assertEqual(mock_process_article_for_batch.call_count, 2)
        
        # Check the result
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], KeyValuePair)
        self.assertEqual(result[0].key_id, "key_0")
        self.assertTrue(isinstance(result[0].key_tokens, torch.Tensor))
        self.assertTrue(isinstance(result[0].value_tokens, torch.Tensor))

    @patch('src.data.batch_compute_embeddings')
    @patch('src.data.tokenize_article_into_chunks')
    def test_process_article_for_batch(self, mock_tokenize_chunks, mock_batch_embeddings):
        """Test the process_article_for_batch function."""
        # Set up mock tokenizer
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(100))  # 100 tokens
        tokenizer.decode.return_value = "Test text"
        
        # Create mock chunks as PyTorch tensors
        mock_chunks = [torch.tensor(list(range(i, i+10)), dtype=torch.long) for i in range(0, 100, 10)]
        mock_tokenize_chunks.return_value = mock_chunks
        
        # Mock batch_compute_embeddings function - return 5 embeddings
        # Each pair uses 2 chunks, so with 10 chunks we get 5 pairs
        mock_batch_embeddings.return_value = [np.array([0.1, 0.2, 0.3])] * 5
        
        # Call the function with min_chunks_per_article=2
        result = process_article_for_batch(
            title="Test Article",
            article_text="This is a test article with sufficient content",
            tokenizer=tokenizer,
            chunk_size=10,
            min_chunks_per_article=2
        )
        
        # Verify results - we get 5 pairs (non-overlapping) from 10 chunks
        self.assertIsInstance(result, list)
        # Each pair needs 2 chunks, so with 10 chunks we get 5 pairs
        self.assertEqual(len(result), 5)
        
        # Verify the structure of KeyValuePair objects
        for i, pair in enumerate(result):
            self.assertIsInstance(pair, KeyValuePair)
            self.assertEqual(pair.key_id, f"key_{i}")
            self.assertEqual(len(pair.key_tokens), 10)
            self.assertEqual(len(pair.value_tokens), 10)
            self.assertIsInstance(pair.key_embedding, np.ndarray)
            self.assertTrue(isinstance(pair.key_tokens, torch.Tensor))
            self.assertTrue(isinstance(pair.value_tokens, torch.Tensor))
        
        # Verify tokenizer and embedding calls
        mock_tokenize_chunks.assert_called_once()
        # Should call batch_compute_embeddings once with all key texts
        mock_batch_embeddings.assert_called_once()
        
        # Test with a short article
        # Set up mock to return only 1 chunk for the short article test
        mock_tokenize_chunks.side_effect = [ValueError("Article is too short")]
        
        result_short = process_article_for_batch(
            title="Short Article",
            article_text="Too short",
            tokenizer=tokenizer,
            chunk_size=10,
            min_chunks_per_article=2
        )
        
        # Should return an empty list (need at least 2 pairs)
        self.assertEqual(result_short, []) 