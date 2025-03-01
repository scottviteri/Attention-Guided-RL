"""
Tests for the trajectory module.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from src.trajectory import Trajectory
from src.config import QUERY_TOKEN_COUNT, VALUE_TOKEN_COUNT, EOT_TOKEN, SYSTEM_START, USER_START

class TestTrajectory:
    """Test cases for the batched Trajectory class."""
    
    @pytest.fixture
    def simple_trajectory(self):
        """Create a simple trajectory with one batch element and two steps."""
        trajectory = Trajectory(batch_size=1)
        
        # Add first step
        trajectory.add_step(
            query_tokens=torch.tensor([[1, 2, 3, 0, 0]]),
            value_tokens=torch.tensor([[4, 5, 6, 0, 0]]),
            query_embeddings=np.array([[0.1, 0.2, 0.3]]),
            key_ids=["key_1"],
            key_probs=[{"key_1": 0.8, "key_2": 0.2}],
            raw_queries=["First query"],
            raw_values=["First value"]
        )
        
        # Add second step
        trajectory.add_step(
            query_tokens=torch.tensor([[7, 8, 9, 0, 0]]),
            value_tokens=torch.tensor([[10, 11, 12, 0, 0]]),
            query_embeddings=np.array([[0.4, 0.5, 0.6]]),
            key_ids=["key_2"],
            key_probs=[{"key_1": 0.3, "key_2": 0.7}],
            raw_queries=["Second query"],
            raw_values=["Second value"]
        )
        
        return trajectory
    
    @pytest.fixture
    def batch_trajectory(self):
        """Create a batched trajectory with two batch elements and two steps."""
        trajectory = Trajectory(batch_size=2)
        
        # Add first step for both batch elements
        trajectory.add_step(
            query_tokens=torch.tensor([
                [1, 2, 3, 0, 0],
                [4, 5, 6, 0, 0]
            ]),
            value_tokens=torch.tensor([
                [7, 8, 9, 0, 0],
                [10, 11, 12, 0, 0]
            ]),
            query_embeddings=np.array([
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ]),
            key_ids=["key_1", "key_2"],
            key_probs=[
                {"key_1": 0.8, "key_2": 0.2},
                {"key_1": 0.3, "key_2": 0.7}
            ],
            raw_queries=["First batch query", "Second batch query"],
            raw_values=["First batch value", "Second batch value"]
        )
        
        # Add second step for both batch elements
        trajectory.add_step(
            query_tokens=torch.tensor([
                [13, 14, 15, 0, 0],
                [16, 17, 18, 0, 0]
            ]),
            value_tokens=torch.tensor([
                [19, 20, 21, 0, 0],
                [22, 23, 24, 0, 0]
            ]),
            query_embeddings=np.array([
                [0.7, 0.8, 0.9],
                [0.1, 0.2, 0.3]
            ]),
            key_ids=["key_3", "key_4"],
            key_probs=[
                {"key_3": 0.6, "key_4": 0.4},
                {"key_3": 0.2, "key_4": 0.8}
            ],
            raw_queries=["Third batch query", "Fourth batch query"],
            raw_values=["Third batch value", "Fourth batch value"]
        )
        
        return trajectory
    
    def test_initialization(self):
        """Test initializing a trajectory."""
        # Empty trajectory
        trajectory = Trajectory()
        assert trajectory.get_batch_size() == 1
        assert trajectory.length == 0
        
        # With specified batch size
        trajectory = Trajectory(batch_size=3)
        assert trajectory.get_batch_size() == 3
        assert trajectory.length == 0
    
    def test_add_step(self, simple_trajectory):
        """Test adding steps to a trajectory."""
        assert simple_trajectory.length == 2
        assert len(simple_trajectory.query_tokens) == 2
        assert len(simple_trajectory.value_tokens) == 2
        assert len(simple_trajectory.query_embeddings) == 2
        assert len(simple_trajectory.key_ids) == 2
        assert len(simple_trajectory.key_probs) == 2
        assert len(simple_trajectory.raw_queries) == 2
        assert len(simple_trajectory.raw_values) == 2
        
        # Check first step values
        assert torch.equal(simple_trajectory.query_tokens[0], torch.tensor([[1, 2, 3, 0, 0]]))
        assert torch.equal(simple_trajectory.value_tokens[0], torch.tensor([[4, 5, 6, 0, 0]]))
        assert np.array_equal(simple_trajectory.query_embeddings[0], np.array([[0.1, 0.2, 0.3]]))
        assert simple_trajectory.key_ids[0] == ["key_1"]
        assert simple_trajectory.key_probs[0] == [{"key_1": 0.8, "key_2": 0.2}]
        assert simple_trajectory.raw_queries[0] == ["First query"]
        assert simple_trajectory.raw_values[0] == ["First value"]
    
    def test_get_token_tensor(self, simple_trajectory):
        """Test converting a trajectory to a token tensor."""
        token_tensor = simple_trajectory.get_token_tensor()
        
        # Expected shape: [batch_size, steps * (query_tokens + value_tokens)]
        # In this case: [1, 10] because 2 steps * (5 query tokens + 5 value tokens)
        assert token_tensor.shape == (1, 20)
        
        # Check concatenated tokens for the first batch element
        expected_tokens = torch.tensor([
            [1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9, 0, 0, 10, 11, 12, 0, 0]
        ])
        assert torch.equal(token_tensor, expected_tokens)
    
    def test_batch_get_token_tensor(self, batch_trajectory):
        """Test converting a batched trajectory to a token tensor."""
        token_tensor = batch_trajectory.get_token_tensor()
        
        # Expected shape: [batch_size, steps * (query_tokens + value_tokens)]
        # In this case: [2, 20] because 2 steps * (5 query tokens + 5 value tokens)
        assert token_tensor.shape == (2, 20)
        
        # Check content of the tensor
        # First batch element, first step
        assert token_tensor[0, 0:5].tolist() == [1, 2, 3, 0, 0]  # First query
        assert token_tensor[0, 5:10].tolist() == [7, 8, 9, 0, 0]  # First value
        
        # First batch element, second step
        assert token_tensor[0, 10:15].tolist() == [13, 14, 15, 0, 0]  # Second query
        assert token_tensor[0, 15:20].tolist() == [19, 20, 21, 0, 0]  # Second value
        
        # Second batch element, first step
        assert token_tensor[1, 0:5].tolist() == [4, 5, 6, 0, 0]  # First query
        assert token_tensor[1, 5:10].tolist() == [10, 11, 12, 0, 0]  # First value
        
        # Second batch element, second step
        assert token_tensor[1, 10:15].tolist() == [16, 17, 18, 0, 0]  # Second query
        assert token_tensor[1, 15:20].tolist() == [22, 23, 24, 0, 0]  # Second value
    
    def test_get_query_contexts(self, simple_trajectory):
        """Test formatting a trajectory for query generation."""
        contexts = simple_trajectory.get_query_contexts()
        
        # Should return one context string per batch element
        assert len(contexts) == 1
        
        # Check content of the context string
        expected = f" <query> First query </query> {EOT_TOKEN} <value> First value </value> {EOT_TOKEN} " \
                   f" <query> Second query </query> {EOT_TOKEN} <value> Second value </value> {EOT_TOKEN} "
        assert contexts[0] == expected
    
    def test_get_reward_contexts(self, simple_trajectory):
        """Test formatting a trajectory for reward calculation."""
        system_prompt = "Test system prompt"
        contexts = simple_trajectory.get_reward_contexts(system_prompt)
        
        # Should return one context string per batch element
        assert len(contexts) == 1
        
        # Check content of the reward context
        query_value_part = f" <query> First query </query> {EOT_TOKEN} <value> First value </value> {EOT_TOKEN} " \
                           f" <query> Second query </query> {EOT_TOKEN} <value> Second value </value> {EOT_TOKEN} "
        expected = f"{SYSTEM_START} {system_prompt} {EOT_TOKEN} {USER_START} {query_value_part} {EOT_TOKEN}"
        assert contexts[0] == expected
    
    def test_batch_get_reward_contexts(self, batch_trajectory):
        """Test formatting a batched trajectory for reward calculation."""
        system_prompt = "Test system prompt"
        contexts = batch_trajectory.get_reward_contexts(system_prompt)
        
        # Should return one context string per batch element
        assert len(contexts) == 2
        
        # Check content of both reward contexts
        query_value_1 = f" <query> First batch query </query> {EOT_TOKEN} <value> First batch value </value> {EOT_TOKEN} "
        query_value_1 += f" <query> Third batch query </query> {EOT_TOKEN} <value> Third batch value </value> {EOT_TOKEN} "
        expected_1 = f"{SYSTEM_START} {system_prompt} {EOT_TOKEN} {USER_START} {query_value_1} {EOT_TOKEN}"
        
        query_value_2 = f" <query> Second batch query </query> {EOT_TOKEN} <value> Second batch value </value> {EOT_TOKEN} "
        query_value_2 += f" <query> Fourth batch query </query> {EOT_TOKEN} <value> Fourth batch value </value> {EOT_TOKEN} "
        expected_2 = f"{SYSTEM_START} {system_prompt} {EOT_TOKEN} {USER_START} {query_value_2} {EOT_TOKEN}"
        
        assert contexts[0] == expected_1
        assert contexts[1] == expected_2
    
    def test_to_dict_list(self, simple_trajectory):
        """Test converting a trajectory to a list of dictionaries."""
        dict_lists = simple_trajectory.to_dict_list()
        
        # Should return a list of dictionary lists, one per batch element
        assert len(dict_lists) == 1
        assert len(dict_lists[0]) == 2  # Two steps
        
        # Check first step of first batch element
        assert dict_lists[0][0]["query"] == "First query"
        assert dict_lists[0][0]["value"] == "First value"
        assert dict_lists[0][0]["key_id"] == "key_1"
        assert np.array_equal(dict_lists[0][0]["query_embedding"], np.array([0.1, 0.2, 0.3]))
        assert dict_lists[0][0]["probs"] == {"key_1": 0.8, "key_2": 0.2}
        
        # Check second step of first batch element
        assert dict_lists[0][1]["query"] == "Second query"
        assert dict_lists[0][1]["value"] == "Second value"
    
    def test_batch_to_dict_list(self, batch_trajectory):
        """Test converting a batched trajectory to a list of dictionaries."""
        dict_lists = batch_trajectory.to_dict_list()
        
        # Should return a list of dictionary lists, one per batch element
        assert len(dict_lists) == 2
        assert len(dict_lists[0]) == 2  # Two steps per batch element
        assert len(dict_lists[1]) == 2
        
        # Check content of the first batch element's first step
        assert dict_lists[0][0]["query"] == "First batch query"
        assert dict_lists[0][0]["key_id"] == "key_1"
        assert dict_lists[0][0]["value"] == "First batch value"
        assert dict_lists[0][0]["probs"] == {"key_1": 0.8, "key_2": 0.2}
        assert np.array_equal(dict_lists[0][0]["query_embedding"], np.array([0.1, 0.2, 0.3]))
        
        # Check content of the second batch element's first step
        assert dict_lists[1][0]["query"] == "Second batch query"
        assert dict_lists[1][0]["key_id"] == "key_2"
        assert dict_lists[1][0]["value"] == "Second batch value"
        assert dict_lists[1][0]["probs"] == {"key_1": 0.3, "key_2": 0.7}
        assert np.array_equal(dict_lists[1][0]["query_embedding"], np.array([0.4, 0.5, 0.6]))
        
        # Check content of the first batch element's second step
        assert dict_lists[0][1]["query"] == "Third batch query"
        assert dict_lists[0][1]["key_id"] == "key_3"
        assert dict_lists[0][1]["value"] == "Third batch value"
        assert dict_lists[0][1]["probs"] == {"key_3": 0.6, "key_4": 0.4}
        assert np.array_equal(dict_lists[0][1]["query_embedding"], np.array([0.7, 0.8, 0.9]))
        
        # Check content of the second batch element's second step
        assert dict_lists[1][1]["query"] == "Fourth batch query"
        assert dict_lists[1][1]["key_id"] == "key_4"
        assert dict_lists[1][1]["value"] == "Fourth batch value"
        assert dict_lists[1][1]["probs"] == {"key_3": 0.2, "key_4": 0.8}
        assert np.array_equal(dict_lists[1][1]["query_embedding"], np.array([0.1, 0.2, 0.3]))
    
    def test_from_dict_lists(self):
        """Test creating a Trajectory from lists of dictionaries."""
        # Set up mock tokenizer
        tokenizer = MagicMock()
        tokenizer.return_value = MagicMock()
        tokenizer.return_value.input_ids = torch.tensor([[1, 2, 3, 0, 0]])
        
        # Create dict lists
        dict_lists = [
            [
                {
                    "query": "Test query 1",
                    "query_embedding": np.array([0.1, 0.2, 0.3]),
                    "key_id": "test_key_1",
                    "value": "Test value 1",
                    "probs": {"test_key_1": 0.8, "test_key_2": 0.2}
                }
            ],
            [
                {
                    "query": "Test query 2",
                    "query_embedding": np.array([0.4, 0.5, 0.6]),
                    "key_id": "test_key_2",
                    "value": "Test value 2",
                    "probs": {"test_key_1": 0.3, "test_key_2": 0.7}
                }
            ]
        ]
        
        # Create trajectory from dict lists
        with patch('src.trajectory.Trajectory.add_step') as mock_add_step:
            trajectory = Trajectory.from_dict_lists(dict_lists, tokenizer)
            
            # Verify add_step was called once with the right parameters
            assert mock_add_step.call_count == 1
            
            # Extract the arguments that were passed to add_step
            args, kwargs = mock_add_step.call_args
            
            # Verify the passed arguments
            assert 'raw_queries' in kwargs
            assert kwargs['raw_queries'] == ["Test query 1", "Test query 2"]
            assert 'raw_values' in kwargs
            assert kwargs['raw_values'] == ["Test value 1", "Test value 2"]
            assert 'key_ids' in kwargs
            assert kwargs['key_ids'] == ["test_key_1", "test_key_2"]
    
    def test_length_and_batch_size(self, simple_trajectory, batch_trajectory):
        """Test the length and batch_size methods."""
        # Test length
        assert len(simple_trajectory) == 2
        assert simple_trajectory.get_batch_size() == 1
        
        assert len(batch_trajectory) == 2
        assert batch_trajectory.get_batch_size() == 2 