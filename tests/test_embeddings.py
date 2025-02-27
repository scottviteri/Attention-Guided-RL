"""
Tests for the embeddings module.
"""

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module to test
from src.embeddings import (
    EmbeddingService,
    ada_embedding,
    batch_ada_embeddings,
    initialize_embedding_service,
    get_embedding_service
)


class TestEmbeddingService:
    """Test cases for the EmbeddingService class."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock = MagicMock()
        
        # Mock embeddings.create response for single embedding request
        embedding_data_single = MagicMock()
        embedding_data_single.embedding = [0.1, 0.2, 0.3]
        
        response_single = MagicMock()
        response_single.data = [embedding_data_single]
        
        # Mock embeddings.create response for batch embedding request
        embedding_data_1 = MagicMock()
        embedding_data_1.embedding = [0.1, 0.2, 0.3]
        
        embedding_data_2 = MagicMock()
        embedding_data_2.embedding = [0.4, 0.5, 0.6]
        
        response_batch = MagicMock()
        response_batch.data = [embedding_data_1, embedding_data_2]
        
        # Configure the mock to return different responses based on input
        def side_effect(model, input):
            if isinstance(input, list) and len(input) > 1:
                return response_batch
            else:
                return response_single
                
        mock.embeddings.create.side_effect = side_effect
        
        return mock
    
    @patch("src.embeddings.OpenAI")
    def test_initialization(self, mock_openai_class, tmp_path):
        """Test EmbeddingService initialization."""
        # Setup mocks
        mock_openai = MagicMock()
        mock_openai_class.return_value = mock_openai
        
        # Create embedding service
        cache_dir = str(tmp_path / "test_cache")
        service = EmbeddingService(model="test-model", cache_dir=cache_dir)
        
        # Verify OpenAI client was created
        mock_openai_class.assert_called_once()
        assert service.model == "test-model"
        assert service.cache_dir == cache_dir
        assert os.path.exists(cache_dir)
    
    @patch("src.embeddings.OpenAI")
    def test_get_embedding(self, mock_openai_class, mock_openai_client):
        """Test getting an embedding."""
        # Setup mocks
        mock_openai_class.return_value = mock_openai_client
        
        # Create embedding service
        service = EmbeddingService(model="test-model", use_cache=False)
        
        # Get embedding
        embedding = service.get_embedding("test text")
        
        # Verify API was called correctly
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="test-model",
            input="test text"
        )
        
        # Verify result
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (3,)  # Our mock returns 3 values
        assert embedding.tolist() == [0.1, 0.2, 0.3]
    
    @patch("src.embeddings.OpenAI")
    def test_get_batch_embeddings(self, mock_openai_class, mock_openai_client):
        """Test getting batch embeddings."""
        # Setup mocks
        mock_openai_class.return_value = mock_openai_client
        
        # Create embedding service
        service = EmbeddingService(model="test-model", use_cache=False)
        
        # Get embeddings
        embeddings = service.get_batch_embeddings(["text1", "text2"])
        
        # Verify API was called correctly
        mock_openai_client.embeddings.create.assert_called_once()
        
        # Verify result
        assert len(embeddings) == 2
        assert isinstance(embeddings[0], np.ndarray)
        assert embeddings[0].shape == (3,)  # Our mock returns 3 values
    
    @patch("src.embeddings.OpenAI")
    def test_compute_similarity(self, mock_openai_class):
        """Test computing similarity between embeddings."""
        # Create embedding service
        service = EmbeddingService(model="test-model", use_cache=False)
        
        # Setup test data
        query_embedding = np.array([1.0, 0.0, 0.0])
        key_embeddings = {
            "key1": np.array([1.0, 0.0, 0.0]),  # Perfect match
            "key2": np.array([0.0, 1.0, 0.0]),  # Orthogonal (no match)
            "key3": np.array([0.5, 0.5, 0.0])   # Partial match
        }
        
        # Compute similarities
        similarities = service.compute_similarity(query_embedding, key_embeddings)
        
        # Verify results
        assert similarities["key1"] == 1.0
        assert similarities["key2"] == 0.0
        assert similarities["key3"] == 0.5


def test_initialize_and_get_embedding_service():
    """Test the global embedding service functions."""
    # Reset the global variable
    import src.embeddings
    src.embeddings.embedding_service = None
    
    # Initialize
    service1 = initialize_embedding_service()
    assert service1 is not None
    
    # Get the service again
    service2 = get_embedding_service()
    assert service2 is service1  # Should be the same instance


@patch("src.embeddings.get_embedding_service")
def test_ada_embedding(mock_get_service):
    """Test the ada_embedding convenience function."""
    # Setup mock
    mock_service = MagicMock()
    mock_service.get_embedding.return_value = np.array([0.1, 0.2, 0.3])
    mock_get_service.return_value = mock_service
    
    # Call function
    embedding = ada_embedding("test text")
    
    # Verify
    mock_service.get_embedding.assert_called_once_with("test text")
    assert embedding.tolist() == [0.1, 0.2, 0.3]


@patch("src.embeddings.get_embedding_service")
def test_batch_ada_embeddings(mock_get_service):
    """Test the batch_ada_embeddings convenience function."""
    # Setup mock
    mock_service = MagicMock()
    mock_service.get_batch_embeddings.return_value = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
    mock_get_service.return_value = mock_service
    
    # Call function
    embeddings = batch_ada_embeddings(["text1", "text2"])
    
    # Verify
    mock_service.get_batch_embeddings.assert_called_once_with(["text1", "text2"])
    assert len(embeddings) == 2
    assert embeddings[0].tolist() == [0.1, 0.2, 0.3]
    assert embeddings[1].tolist() == [0.4, 0.5, 0.6] 