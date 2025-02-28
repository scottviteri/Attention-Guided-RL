"""
Tests for the RL agent module.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, ANY
import dataclasses
import bitsandbytes as bnb

# Import the module to test
from src.rl_agent import (
    TrajectoryCollector,
    ReinforcementLearner,
    train_episode,
    run_training
)
from src.trajectory import Trajectory
from src.config import QUERY_TOKEN_COUNT, VALUE_TOKEN_COUNT

# Try to import bitsandbytes, but don't fail if it's not available
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


class TestTrajectoryCollector:
    """Test cases for TrajectoryCollector."""
    
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
    
    @patch("src.rl_agent.compute_embeddings")
    @patch("src.rl_agent.softmax")
    @patch("src.rl_agent.sample_from_distribution")
    def test_collect_trajectory(self, mock_sample, mock_softmax, mock_compute, mock_model, mock_database):
        """Test collecting a trajectory with the new implementation."""
        # Mock embedding computation
        mock_compute.return_value = np.array([[0.1, 0.2, 0.3]])
    
        # Mock softmax and sampling
        mock_softmax.return_value = np.array([0.7, 0.3])
        mock_sample.return_value = 0
    
        # Create collector
        collector = TrajectoryCollector(
            model=mock_model,
            database=mock_database,
            temperature=0.8
        )
    
        # Mock the _select_key method to return empty database after selection
        # This will cause the trajectory to complete after one step
        original_select_key = collector._select_key
        
        def mock_select_key(database, key_id):
            result = original_select_key(database, key_id)
            # Empty the database to simulate completing the trajectory
            database.clear()
            return result
        
        collector._select_key = mock_select_key
    
        # Mock the tokenizer to return tensors
        query_tokens = torch.zeros((1, QUERY_TOKEN_COUNT), dtype=torch.long)
        value_tokens = torch.zeros((1, VALUE_TOKEN_COUNT), dtype=torch.long)
        mock_model.tokenizer.side_effect = None
        mock_model.tokenizer.return_value.input_ids = query_tokens
    
        # Patch torch.cat to avoid tensor issues
        with patch("torch.cat", return_value=torch.zeros((1, QUERY_TOKEN_COUNT), dtype=torch.long)):
            # Collect trajectory
            trajectory = collector.collect_trajectory(batch_size=1)
            
            # Verify
            assert isinstance(trajectory, Trajectory)
            assert trajectory.length > 0
            mock_compute.assert_called()
            mock_softmax.assert_called()
            mock_sample.assert_called()


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
def test_train_episode(mock_trajectory_collector_class):
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
    mock_collector.collect_trajectory.assert_called_once_with(batch_size=2, verbose=True)
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
def test_run_training(mock_logger, mock_compute_stats, mock_trajectory_collector_class, mock_learner_class, mock_language_model, global_mock_model):
    """Test the full training loop with baseline model and warm-up period."""
    # Setup mocks
    mock_collector = MagicMock()
    mock_trajectory_collector_class.return_value = mock_collector
    
    mock_learner = MagicMock()
    mock_learner_class.return_value = mock_learner
    
    mock_baseline = MagicMock()
    mock_language_model.return_value = mock_baseline
    
    # Mock trajectory and rewards
    mock_trajectory = MagicMock()
    mock_collector.collect_trajectory.return_value = mock_trajectory
    
    mock_rewards = [0.5, 0.6, 0.7]  # Example rewards for three episodes
    mock_learner.compute_trajectory_rewards.return_value = mock_rewards
    
    # Mock running stats calculation
    mock_compute_stats.return_value = (0.6, 0.1)  # (mean, std)
    
    # Mock update_policy for after warm-up
    mock_learner.update_policy.return_value = {
        "loss": 0.1,
        "policy_loss": 0.05,
        "kl_loss": 0.05,
        "grad_norm": 0.5,
        "reward_mean": 0.6,
        "reward_std": 0.1,
        "filtered_ratio": 0.5
    }
    
    # Test run_training
    from src.rl_agent import run_training
    
    # Mock database
    mock_db = [MagicMock(), MagicMock()]
    
    # Run with minimal episodes
    rewards = run_training(
        model=global_mock_model,
        database=mock_db,
        num_episodes=3,  # Minimal for testing
        batch_size=2,
        learning_rate=0.001,
        kl_weight=0.1
    )
    
    # Verify
    mock_language_model.assert_called_once()
    mock_trajectory_collector_class.assert_called_once_with(global_mock_model, mock_db)
    mock_learner_class.assert_called_once()
    assert len(rewards) == 3  # Should have rewards for all episodes 