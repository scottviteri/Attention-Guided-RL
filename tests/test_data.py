"""
Tests for the data module.
"""

import os
import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

# Import the module to test
from src.data import (
    WikiArticleProcessor,
    KeyValueDatabase,
    load_wikipedia_article
)


class TestWikiArticleProcessor:
    """Test cases for the WikiArticleProcessor class."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        mock = MagicMock()
        mock.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mock.decode.side_effect = lambda ids, **kwargs: "Text for token IDs: " + str(ids)
        mock.pad_token_id = 0
        return mock
    
    @patch("src.data.AutoTokenizer")
    def test_initialization(self, mock_auto_tokenizer, mock_tokenizer, tmp_path):
        """Test WikiArticleProcessor initialization."""
        # Setup mocks
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # Create processor
        cache_dir = str(tmp_path / "test_cache")
        processor = WikiArticleProcessor(model_name="test-model", cache_dir=cache_dir)
        
        # Verify
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("test-model")
        assert processor.tokenizer == mock_tokenizer
        assert processor.cache_dir == cache_dir
        assert os.path.exists(cache_dir)
    
    @patch("src.data.AutoTokenizer")
    @patch("src.data.requests.get")
    @patch("src.data.load_dataset")
    def test_fetch_wikipedia_article(self, mock_load_dataset, mock_get, mock_auto_tokenizer, mock_tokenizer, tmp_path):
        """Test fetching a Wikipedia article."""
        # Setup mocks
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # Mock Wikipedia API as fallback
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "12345": {
                        "extract": "This is a test article from Wikipedia API."
                    }
                }
            }
        }
        mock_get.return_value = mock_response
        
        # Mock the dataset iterator to return a few test articles
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = iter([
            {"title": "Some other article", "text": "Content of some other article"},
            {"title": "Artificial intelligence", "text": "AI is the simulation of human intelligence in machines"},
            {"title": "Yet another article", "text": "Content of yet another article"}
        ])
        mock_load_dataset.return_value = mock_dataset
        
        # Create processor
        processor = WikiArticleProcessor(model_name="test-model")
        
        # Fetch article using mocked dataset
        article = processor.fetch_wikipedia_article("Artificial intelligence")
        
        # Verify article contains content
        assert len(article) > 0
        assert "ai is the simulation" in article.lower()
        
        # Test fallback to Wikipedia API when article not found
        # Reset the mock dataset iterator for the second test
        mock_dataset.__iter__.return_value = iter([
            {"title": "Some other article", "text": "Content of some other article"},
            {"title": "Yet another article", "text": "Content of yet another article"}
        ])
        
        article = processor.fetch_wikipedia_article("ThisIsAVeryUnlikelyWikipediaArticleTitleThatShouldNotExist12345")
        
        # Verify Wikipedia API was used as fallback
        mock_get.assert_called_once()
        assert "test article from wikipedia api" in article.lower()
    
    @patch("src.data.AutoTokenizer")
    def test_tokenize_article_into_chunks(self, mock_auto_tokenizer, mock_tokenizer):
        """Test tokenizing article into chunks."""
        # Setup mocks
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.encode.return_value = list(range(1, 20))  # 19 tokens
        
        # Create processor
        processor = WikiArticleProcessor(model_name="test-model")
        
        # Tokenize into chunks
        chunks = processor.tokenize_article_into_chunks("Test article text", chunk_size=5, max_chunks=5)
        
        # Verify
        mock_tokenizer.encode.assert_called_once_with("Test article text", add_special_tokens=False)
        assert len(chunks) == 4  # Should create 4 chunks of size 5 from 19 tokens
        assert chunks[0] == [1, 2, 3, 4, 5]
        assert chunks[1] == [6, 7, 8, 9, 10]
        assert chunks[2] == [11, 12, 13, 14, 15]
        assert chunks[3] == [16, 17, 18, 19, 0]  # Last chunk padded
    
    @patch("src.data.AutoTokenizer")
    def test_create_key_value_pairs_from_chunks(self, mock_auto_tokenizer, mock_tokenizer):
        """Test creating key-value pairs from chunks."""
        # Setup mocks
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # Create processor
        processor = WikiArticleProcessor(model_name="test-model")
        
        # Create pairs
        chunks = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20]
        ]
        pairs = processor.create_key_value_pairs_from_chunks(chunks, max_pairs=2)
        
        # Verify
        assert len(pairs) == 2
        assert pairs[0] == ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        assert pairs[1] == ([6, 7, 8, 9, 10], [11, 12, 13, 14, 15])
    
    @patch("src.data.AutoTokenizer")
    @patch("src.data.batch_ada_embeddings")
    def test_process_article(self, mock_batch_embeddings, mock_auto_tokenizer, mock_tokenizer):
        """Test processing a complete article."""
        # Setup mocks
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_batch_embeddings.return_value = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        
        # Mock processor methods
        processor = WikiArticleProcessor(model_name="test-model")
        processor.fetch_wikipedia_article = MagicMock(return_value="Test article content")
        processor.tokenize_article_into_chunks = MagicMock(return_value=[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])
        processor.create_key_value_pairs_from_chunks = MagicMock(return_value=[
            ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
            ([6, 7, 8, 9, 10], [11, 12, 13, 14, 15])
        ])
        
        # Process article
        data = processor.process_article(title="Test Article", max_pairs=2)
        
        # Verify
        assert processor.fetch_wikipedia_article.called
        assert processor.tokenize_article_into_chunks.called
        assert processor.create_key_value_pairs_from_chunks.called
        assert mock_batch_embeddings.called
        
        assert data["title"] == "Test Article"
        assert len(data["pairs"]) == 2
        assert len(data["key_embeddings"]) == 2


class TestKeyValueDatabase:
    """Test cases for the KeyValueDatabase class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {
            "title": "Test Article",
            "pairs": [
                ("Key 1", "Value 1"),
                ("Key 2", "Value 2"),
                ("Key 3", "Value 3")
            ],
            "key_embeddings": {
                "key_0": [0.1, 0.2, 0.3],
                "key_1": [0.4, 0.5, 0.6],
                "key_2": [0.7, 0.8, 0.9]
            }
        }
    
    def test_initialization(self, sample_data):
        """Test KeyValueDatabase initialization."""
        # Create database
        db = KeyValueDatabase(sample_data)
        
        # Verify
        assert db.title == "Test Article"
        assert len(db.pairs) == 3
        assert len(db.key_embeddings) == 3
        assert len(db.available_keys) == 3
        assert isinstance(db.key_embeddings["key_0"], np.ndarray)
    
    def test_get_available_key_embeddings(self, sample_data):
        """Test getting available key embeddings."""
        # Create database
        db = KeyValueDatabase(sample_data)
        
        # Get embeddings
        embeddings = db.get_available_key_embeddings()
        
        # Verify
        assert len(embeddings) == 3
        assert "key_0" in embeddings
        assert isinstance(embeddings["key_0"], np.ndarray)
    
    def test_select_key(self, sample_data):
        """Test selecting a key-value pair."""
        # Create database
        db = KeyValueDatabase(sample_data)
        
        # Select a key
        key, value = db.select_key("key_1")
        
        # Verify
        assert key == "Key 2"
        assert value == "Value 2"
        assert "key_1" not in db.available_keys
        assert len(db.available_keys) == 2
    
    def test_reset(self, sample_data):
        """Test resetting the database."""
        # Create database
        db = KeyValueDatabase(sample_data)
        
        # Select a key and then reset
        db.select_key("key_1")
        db.reset()
        
        # Verify
        assert "key_1" in db.available_keys
        assert len(db.available_keys) == 3
    
    def test_is_empty(self, sample_data):
        """Test checking if database is empty."""
        # Create database
        db = KeyValueDatabase(sample_data)
        
        # Initially not empty
        assert not db.is_empty()
        
        # Select all keys
        db.select_key("key_0")
        db.select_key("key_1")
        db.select_key("key_2")
        
        # Should be empty now
        assert db.is_empty()


@patch("src.data.WikiArticleProcessor")
def test_load_wikipedia_article(mock_processor_class):
    """Test the load_wikipedia_article helper function."""
    # Setup mocks
    mock_processor = MagicMock()
    mock_processor_class.return_value = mock_processor
    
    # Mock processor methods
    mock_processor.process_article.return_value = {
        "title": "Test Article",
        "pairs": [("Key", "Value")],
        "key_embeddings": {"key_0": [0.1, 0.2, 0.3]}
    }
    
    # Load article
    database = load_wikipedia_article(title="Test Article", max_pairs=10)
    
    # Verify
    assert mock_processor.process_article.called
    assert isinstance(database, KeyValueDatabase) 