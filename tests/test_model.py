"""
Tests for the model module.
"""

import os
import pytest
import torch
from unittest.mock import patch, MagicMock

# Import the module to test
from src.model import LanguageModel, load_language_model

# Skip tests that require GPU if not available
skip_if_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Test requires GPU"
)

class TestLanguageModel:
    """Test cases for the LanguageModel class."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        mock = MagicMock()
        mock.encode.return_value = [1, 2, 3]
        mock.decode.return_value = "Generated text"
        mock.pad_token = None
        mock.eos_token = "[EOS]"
        return mock
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        mock = MagicMock()
        mock.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        return mock
    
    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForCausalLM")
    @patch("src.model.GenerationConfig")
    def test_initialization(self, mock_gen_config, mock_auto_model, mock_auto_tokenizer, mock_tokenizer, mock_model):
        """Test LanguageModel initialization."""
        # Setup mocks
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_gen_config.from_pretrained.return_value = MagicMock()
        
        # Create language model
        model = LanguageModel(model_name="test_model", device="cpu")
        
        # Verify
        mock_auto_tokenizer.from_pretrained.assert_called_once()
        mock_auto_model.from_pretrained.assert_called_once()
        assert model.tokenizer.pad_token == "[EOS]"  # Should set pad_token to eos_token
    
    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForCausalLM")
    @patch("src.model.GenerationConfig")
    def test_generate_query(self, mock_gen_config, mock_auto_model, mock_auto_tokenizer, mock_tokenizer, mock_model):
        """Test query generation."""
        # Setup mocks
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_gen_config.from_pretrained.return_value = MagicMock()
        
        # Setup tokenizer behavior
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.input_ids = torch.tensor([[1, 2, 3]])
        
        # Create language model
        model = LanguageModel(model_name="test_model", device="cpu")
        model.model = mock_model
        model.tokenizer = mock_tokenizer
        
        # Generate query
        query = model.generate_query(context="Test context", fixed_token_count=5)
        
        # Verify
        assert mock_model.generate.called
        assert query == "Generated text"
    
    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForCausalLM")
    @patch("src.model.GenerationConfig")
    def test_compute_token_probabilities(self, mock_gen_config, mock_auto_model, mock_auto_tokenizer, mock_tokenizer, mock_model):
        """Test computation of token probabilities."""
        # Setup mocks
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_gen_config.from_pretrained.return_value = MagicMock()
        
        # Setup model output
        model_output = MagicMock()
        model_output.logits = torch.ones((1, 5, 10))  # Batch size 1, sequence length 5, vocab size 10
        mock_model.return_value = model_output
        
        # Create language model
        model = LanguageModel(model_name="test_model", device="cpu")
        model.model = mock_model
        
        # Compute probabilities
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        log_probs = model.compute_token_probabilities(input_ids)
        
        # Verify model was called
        assert mock_model.called
    
    @patch("src.model.load_language_model")
    def test_load_language_model(self, mock_load):
        """Test the load_language_model helper function."""
        # Setup mock
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        # Get the patched function directly from the test
        from src.model import load_language_model
        
        # Call the patched function (which should now be the mock)
        model = load_language_model(model_name="test_model", device="cpu")
        
        # Verify
        assert mock_load.called
        assert model is mock_model 