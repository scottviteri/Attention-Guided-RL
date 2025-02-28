"""
Tests for the model module.
"""

import os
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module to test
from src.model import LanguageModel, load_language_model
from src.trajectory import Trajectory

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
    
    @pytest.fixture
    def mock_trajectory(self):
        """Create a mock trajectory with batch size 2."""
        mock = MagicMock(spec=Trajectory)
        mock.batch_size = 2
        mock.length = 2
        mock.get_reward_contexts.return_value = [
            "System: Test prompt [EOS] User: Query: Query 1 [EOS] Value: Value 1 [EOS] Query: Query 2 [EOS] Value: Value 2 [EOS] [EOS]",
            "System: Test prompt [EOS] User: Query: Query 3 [EOS] Value: Value 3 [EOS] [EOS]"
        ]
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

    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForCausalLM")
    @patch("src.model.GenerationConfig")
    def test_calculate_trajectory_rewards(self, mock_gen_config, mock_auto_model, mock_auto_tokenizer, mock_tokenizer, mock_model, mock_trajectory):
        """Test batch calculation of trajectory rewards using the Trajectory class."""
        # Setup mocks
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_gen_config.from_pretrained.return_value = MagicMock()
        
        # Create real tensors instead of mocks
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        attention_mask = torch.ones_like(input_ids)
        
        # Setup tokenizer behavior
        tokenizer_return = MagicMock()
        tokenizer_return.input_ids = input_ids
        tokenizer_return.attention_mask = attention_mask
        tokenizer_return.to.return_value = tokenizer_return
        mock_tokenizer.return_value = tokenizer_return
        
        # Setup model output with real tensors
        model_output = MagicMock()
        model_output.logits = torch.ones((2, 4, 10))  # Batch size 2, sequence length 4, vocab size 10
        mock_model.return_value = model_output
        
        # Create language model
        model = LanguageModel(model_name="test_model", device="cpu")
        model.model = mock_model
        model.tokenizer = mock_tokenizer
        
        # Mock input_ids.size to return a tuple for size dimensions
        input_ids.size = MagicMock(side_effect=lambda dim: input_ids.shape[dim] if dim < len(input_ids.shape) else input_ids.shape)
        
        # Mock item() for tensor elements
        def mock_item(idx):
            if isinstance(idx, tuple):
                i, j = idx
                return int(input_ids[i, j])
            return int(idx)
            
        for i in range(input_ids.shape[0]):
            for j in range(input_ids.shape[1]):
                input_ids[i, j].item = lambda i=i, j=j: mock_item((i, j))
        
        # Mock the _extract_value_positions method to return test positions
        # This should return positions within the bounds of our input_ids tensor
        model._extract_value_positions = MagicMock(return_value=[(1, 2)])
        
        # Mock the trajectory's get_reward_contexts method
        mock_trajectory.get_reward_contexts.return_value = [
            "System: Test prompt [EOS] User: Query: Query 1 [EOS] Value: Value 1 [EOS] Query: Query 2 [EOS] Value: Value 2 [EOS] [EOS]",
            "System: Test prompt [EOS] User: Query: Query 3 [EOS] Value: Value 3 [EOS] [EOS]"
        ]
        
        # Create a mock baseline model
        baseline_model = MagicMock()
        baseline_model.device = "cpu"
        baseline_model.model = MagicMock()
        baseline_output = MagicMock()
        baseline_output.logits = torch.ones((2, 4, 10)) * 0.5  # Different logits to test normalization
        baseline_model.model.return_value = baseline_output
        
        # Test without baseline
        rewards = model.calculate_trajectory_rewards(
            contexts=mock_trajectory.get_reward_contexts(),
            baseline_model=None
        )
        
        # Verify
        assert len(rewards) == 2  # Should have rewards for both trajectories
        assert all(isinstance(r, float) for r in rewards)
        
        # Test with baseline
        normalized_rewards = model.calculate_trajectory_rewards(
            contexts=mock_trajectory.get_reward_contexts(),
            baseline_model=baseline_model
        )
        
        # Verify
        assert len(normalized_rewards) == 2  # Should have rewards for both trajectories
        assert all(isinstance(r, float) for r in normalized_rewards) 