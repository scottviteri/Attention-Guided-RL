"""
Tests for the RL agent module.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, ANY
import dataclasses
import bitsandbytes as bnb
import os

# Import the module to test
from src.rl_agent import (
    TrajectoryCollector,
    ReinforcementLearner,
    train_episode,
    run_training,
    select_key
)
from src.trajectory import Trajectory
from src.config import QUERY_TOKEN_COUNT, VALUE_TOKEN_COUNT, Config
from src.data import KeyValuePair

# Try to import bitsandbytes, but don't fail if it's not available
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


@pytest.fixture
def mock_model():
    """Fixture for a mock model."""
    mock = MagicMock()
    mock.generate_query.return_value = "What is artificial intelligence?"
    mock.device = "cpu"
    mock.tokenizer.return_value.input_ids = torch.zeros((1, QUERY_TOKEN_COUNT), dtype=torch.long)
    return mock

@pytest.fixture
def mock_database():
    """Fixture for a mock database of key-value pairs."""
    return [
        KeyValuePair(
            key_tokens=torch.tensor([1, 2, 3]),
            value_tokens=torch.tensor([4, 5, 6]),
            key_embedding=np.array([0.1, 0.2, 0.3]),
            key_id="key_0",
            key="Key 0",
            value="Value 0"
        ),
        KeyValuePair(
            key_tokens=torch.tensor([7, 8, 9]),
            value_tokens=torch.tensor([10, 11, 12]),
            key_embedding=np.array([0.4, 0.5, 0.6]),
            key_id="key_1",
            key="Key 1", 
            value="Value 1"
        )
    ]

def test_select_key(mock_database):
    """Test the select_key pure function."""
    # Test selecting a key
    key, value, updated_db = select_key(mock_database, "key_0")
    
    # Check that the key was removed from the database
    assert len(updated_db) == 1
    assert updated_db[0].key_id == "key_1"
    
    # Check that the original database is unchanged
    assert len(mock_database) == 2
    
    # Check the returned key and value
    assert key == "Key 0"
    assert value == "Value 0"
    
    # Test with tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode = lambda x: "Decoded " + str(x.tolist())
    
    # Create a database with empty key/value strings to test tokenizer decoding
    test_db = [
        KeyValuePair(
            key_tokens=torch.tensor([7, 8, 9]),
            value_tokens=torch.tensor([10, 11, 12]),
            key_embedding=np.array([0.4, 0.5, 0.6]),
            key_id="key_1",
            key="",  # Empty key to force tokenizer decoding
            value=""  # Empty value to force tokenizer decoding
        )
    ]
    
    key, value, updated_db = select_key(test_db, "key_1", mock_tokenizer)
    
    # Check the decoded values
    assert key == "Decoded [7, 8, 9]"
    assert value == "Decoded [10, 11, 12]"
    
    # Test with invalid key_id
    with pytest.raises(ValueError):
        select_key(mock_database, "nonexistent_key")

def test_config():
    """Test that the Config dataclass is immutable."""
    config = Config()
    
    # Test that we can read values
    assert config.model_name == "meta-llama/Llama-3.2-3B-Instruct"
    
    # Test that the class is frozen
    with pytest.raises(Exception):
        config.model_name = "new-model-name"

class TestTrajectoryCollector:
    """Test cases for the TrajectoryCollector class."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock language model and tokenizer."""
        model = MagicMock()
        model.device = "cpu"
        model.generate_query = MagicMock(return_value="Test query")
        
        # Mock tokenizer
        tokenizer = MagicMock()
        model.tokenizer = tokenizer
        model.tokenizer.return_value.input_ids = torch.zeros((1, QUERY_TOKEN_COUNT), dtype=torch.long)
        
        return model
    
    @pytest.fixture
    def mock_database(self):
        """Mock database with key-value pairs."""
        kv1 = MagicMock()
        kv1.key_id = "key_0"
        kv1.key_embedding = np.array([0.1, 0.2, 0.3])
        
        kv2 = MagicMock()
        kv2.key_id = "key_1"
        kv2.key_embedding = np.array([0.2, 0.3, 0.4])
        
        return [kv1, kv2]
    
    @patch("src.rl_agent.select_value_with_attention")
    def test_collect_trajectory(self, mock_select_value, mock_model, mock_database):
        """Test collecting a trajectory with the new attention-based implementation."""
        # Mock the select_value_with_attention function
        mock_select_value.return_value = ("Mock Key", "Mock Value", [])
    
        # Create collector
        collector = TrajectoryCollector(
            model=mock_model,
            database=mock_database,
            temperature=0.8
        )
    
        # Collect trajectory
        trajectory = collector.collect_trajectory(query="Test query")
        
        # Verify
        assert isinstance(trajectory, Trajectory)
        assert trajectory.length > 0
        assert len(trajectory.steps) > 0
        assert trajectory.steps[0]['query'] is not None
        assert trajectory.steps[0]['key'] == "Mock Key"
        assert trajectory.steps[0]['value'] == "Mock Value"
        assert trajectory.steps[0]['context'] is not None
        mock_select_value.assert_called()


class TestReinforcementLearner:
    """Test cases for the ReinforcementLearner class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock language model."""
        mock = MagicMock()
        mock.device = "cpu"
        
        # Create a parameter with requires_grad=True for the optimizer
        mock_model_param = MagicMock()
        mock_model_param.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3], requires_grad=True))])
        mock.model = mock_model_param
        
        return mock
    
    @pytest.fixture
    def mock_baseline_model(self):
        """Create a mock baseline language model."""
        mock = MagicMock()
        mock.device = "cpu"
        mock.calculate_batch_trajectory_rewards = MagicMock(return_value=[0.5, 0.4])
        
        return mock
    
    @pytest.fixture
    def mock_trajectory(self):
        """Create a mock trajectory."""
        mock = MagicMock(spec=Trajectory)
        mock.batch_size = 2
        mock.length = 2
        mock.to_dict_list.return_value = [
            [{"query": "q1", "value": "v1"}, {"query": "q2", "value": "v2"}],
            [{"query": "q3", "value": "v3"}, {"query": "q4", "value": "v4"}]
        ]
        mock.get_reward_contexts.return_value = ["context1", "context2"]
        mock.get_batch_size.return_value = 2
        return mock
    
    def test_compute_trajectory_rewards(self, mock_model):
        """Test computing reward for a trajectory."""
        # Create learner
        learner = ReinforcementLearner(model=mock_model)
        
        # Mock the model's calculate_trajectory_rewards method
        mock_model.calculate_trajectory_rewards = MagicMock(return_value=[1.5])
        
        # Create a mock trajectory
        mock_trajectory = MagicMock(spec=Trajectory)
        mock_trajectory.to_dict_list.return_value = [[
            {"query": "q1", "value": "v1"},
            {"query": "q2", "value": "v2"}
        ]]
        mock_trajectory.get_reward_contexts.return_value = ["context1"]
        
        # Compute reward
        reward = learner.compute_trajectory_rewards(mock_trajectory)
        
        # Verify
        assert mock_model.calculate_trajectory_rewards.called
        assert reward == [1.5]
    
    @patch("src.rl_agent.compute_running_stats")
    @patch("torch.nn.utils.clip_grad_norm_")
    @patch("bitsandbytes.optim.Adam8bit")
    def test_update_policy(self, mock_adam8bit, mock_clip_grad, mock_compute_stats, mock_model, mock_baseline_model, mock_trajectory):
        """Test updating policy with a trajectory and rewards."""
        # Mock compute_running_stats
        mock_compute_stats.return_value = (0.5, 0.2)
        mock_clip_grad.return_value = torch.tensor(0.1)
    
        # Create mock optimizer
        mock_optimizer = MagicMock()
        mock_adam8bit.return_value = mock_optimizer
    
        # Create learner with mocked optimizer and always include a baseline model
        learner = ReinforcementLearner(
            model=mock_model,
            baseline_model=mock_baseline_model,  # Always provide a baseline model
            optimizer=None,  # Let it create the optimizer
            learning_rate=0.0001,
            kl_weight=0.01
        )
    
        # Replace the optimizer with our mock
        learner.optimizer = mock_optimizer
    
        # Set up enough baseline rewards to be past the warm-up phase
        learner.baseline_rewards = [0.8, 0.9, 1.0] * 10  # More than WARMUP_EPISODES
    
        # Prepare mocks for policy loss and KL loss
        policy_loss = torch.tensor(0.5, requires_grad=True)
        kl_loss = torch.tensor(0.2, requires_grad=True)
        learner.compute_policy_loss = MagicMock(return_value=policy_loss)
        learner.compute_kl_loss = MagicMock(return_value=kl_loss)
    
        # Setup mock for extract value positions
        mock_model._extract_value_positions = MagicMock(return_value=[(1, 5)])
        
        # Mock model outputs for KL computation
        mock_logits = torch.randn(2, 10)
        mock_model.model.return_value = MagicMock(logits=mock_logits)
        mock_baseline_model.model.return_value = MagicMock(logits=mock_logits)
        
        # Mock the tokenizer
        mock_inputs = MagicMock()
        mock_inputs.input_ids = torch.ones((2, 10), dtype=torch.long)
        mock_inputs.attention_mask = torch.ones((2, 10), dtype=torch.long)
        mock_inputs.to.return_value = mock_inputs
        mock_model.tokenizer.return_value = mock_inputs
        
        # Test 1: No trajectories pass the filter - should return early without updating
        mock_compute_stats.return_value = (1.5, 0.2)  # High mean, rewards below threshold
        no_filter_stats = learner.update_policy(trajectory=mock_trajectory, rewards=[1.0, 1.2])
    
        # Verify no optimizer calls when no trajectories pass filter
        assert not mock_optimizer.zero_grad.called, "optimizer.zero_grad() should not be called when no trajectories pass filter"
        assert "loss" in no_filter_stats and no_filter_stats["loss"] == 0.0
        assert "filtered_ratio" in no_filter_stats and no_filter_stats["filtered_ratio"] == 0.0
    
        # Test 2: Some trajectories pass the filter - should update policy
        mock_compute_stats.return_value = (0.5, 0.2)  # Low mean, rewards above threshold
        
        # Mock get_reward_contexts to return example contexts
        mock_trajectory.get_reward_contexts.return_value = ["context1", "context2"]
        
        stats = learner.update_policy(trajectory=mock_trajectory, rewards=[1.0, 1.2])
    
        # Verify optimizer was called
        assert mock_optimizer.zero_grad.called, "optimizer.zero_grad() should be called"
        assert mock_optimizer.step.called, "optimizer.step() should be called"
        
        # Verify stats
        assert "loss" in stats and stats["loss"] > 0
        assert "policy_loss" in stats
        assert "kl_loss" in stats
        assert "grad_norm" in stats
        assert "filtered_ratio" in stats and 0 < stats["filtered_ratio"] <= 1.0
    
        # Verify compute_policy_loss was called with filtered rewards
        learner.compute_policy_loss.assert_called_once()
        
        # Verify compute_kl_loss was called with appropriate arguments
        learner.compute_kl_loss.assert_called_once()


@patch("src.rl_agent.TrajectoryCollector")
@patch("src.rl_agent.REWARD_SYSTEM_PROMPT", "Test Prompt")
def test_train_episode(mock_trajectory_collector_class, mock_reward_prompt):
    """Test training a single episode with the new Trajectory class."""
    # Setup mocks
    mock_collector = MagicMock()
    mock_trajectory = MagicMock(spec=Trajectory)
    mock_collector.collect_trajectory.return_value = mock_trajectory
    
    # Create learner with both model and baseline model
    learner = MagicMock()
    rewards = [1.0, 1.5]
    learner.compute_trajectory_rewards.return_value = rewards
    
    # Use the exact same values in the return dict as in the rewards variable
    update_stats = {
        "loss": 0.5,
        "reward_mean": np.mean(rewards),  # Use numpy calculation to match test assertion
        "reward_std": np.std(rewards),    # Use numpy calculation to match test assertion
        "filtered_count": 1,
        "grad_norm": 0.1
    }
    learner.update_policy.return_value = update_stats
    
    # Call the function with baseline_shift parameter
    result = train_episode(
        collector=mock_collector,
        learner=learner,
        batch_size=2,
        baseline_shift=0.5,
        verbose=True
    )
    
    # Verify
    mock_collector.collect_trajectory.assert_called_once_with(
        query="Test Prompt",
        num_steps=5,
        verbose=True
    )
    learner.compute_trajectory_rewards.assert_called_once_with(mock_trajectory)
    
    # Verify policy update with rewards
    learner.update_policy.assert_called_once_with(mock_trajectory, rewards)
    
    # Check result stats - using the exact values from the mock return
    assert result["reward_mean"] == np.mean(rewards)
    assert result["reward_std"] == np.std(rewards)
    assert result["reward_min"] == np.min(rewards)
    assert result["reward_max"] == np.max(rewards)
    assert result["loss"] == 0.5


@pytest.fixture
def global_mock_model():
    """Create a mock language model for global tests."""
    mock = MagicMock()
    mock.device = "cpu"
    
    # Create a parameter with requires_grad=True for the optimizer
    mock_model_param = MagicMock()
    mock_model_param.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3], requires_grad=True))])
    mock.model = mock_model_param
    
    return mock


@patch("src.rl_agent.LanguageModel")
@patch("src.rl_agent.ReinforcementLearner")
@patch("src.rl_agent.TrajectoryCollector")
@patch("src.rl_agent.compute_running_stats")
@patch("src.rl_agent.logger")
@patch("src.rl_agent.train_episode")
def test_run_training(mock_train_episode, mock_logger, mock_compute_stats, mock_trajectory_collector_class, mock_learner_class, mock_language_model, global_mock_model):
    """Test the full training loop with baseline model and warm-up period."""
    # Setup mocks
    mock_collector = MagicMock()
    mock_trajectory_collector_class.return_value = mock_collector
    
    mock_learner = MagicMock()
    mock_learner_class.return_value = mock_learner
    
    mock_baseline = MagicMock()
    mock_language_model.return_value = mock_baseline
    
    # Mock train_episode output
    mock_train_episode.return_value = {
        "loss": 0.1,
        "policy_loss": 0.05,
        "kl_loss": 0.05,
        "grad_norm": 0.5,
        "reward_mean": 0.6,
        "reward_std": 0.1,
        "filtered_ratio": 0.5
    }
    
    # Mock KeyValueDatabase.from_file
    with patch("src.rl_agent.KeyValueDatabase") as mock_db:
        mock_db.from_file.return_value = [MagicMock(), MagicMock()]
        
        # Mock os.makedirs
        with patch("os.makedirs") as mock_makedirs:
            # Test run_training
            from src.rl_agent import run_training
            
            # Run with minimal episodes
            rewards = run_training(
                model_name="test-model",
                data_path="test-data",
                output_dir="test-output",
                num_episodes=3,  # Minimal for testing
                batch_size=2,
                learning_rate=0.001,
                kl_weight=0.1,
                verbose=1
            )
            
            # Verify
            mock_language_model.assert_called_once()
            mock_trajectory_collector_class.assert_called_once()
            mock_learner_class.assert_called_once()
            
            # Verify train_episode was called the correct number of times
            assert mock_train_episode.call_count == 3
            
            # Verify the rewards list is returned
            assert len(rewards) == 3
            assert all(reward == 0.6 for reward in rewards) 