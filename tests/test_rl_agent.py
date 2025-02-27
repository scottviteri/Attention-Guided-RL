"""
Tests for the RL agent module.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, call

# Import the module to test
from src.rl_agent import (
    TrajectoryCollector,
    ReinforcementLearner,
    train_episode,
    run_training
)


class TestTrajectoryCollector:
    """Test cases for the TrajectoryCollector class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock language model."""
        mock = MagicMock()
        mock.device = "cpu"
        mock.generate_query.return_value = "Test query"
        return mock
    
    @pytest.fixture
    def mock_database(self):
        """Create a mock key-value database."""
        mock = MagicMock()
        mock.get_available_key_embeddings.return_value = {
            "key_1": np.array([0.1, 0.2, 0.3]),
            "key_2": np.array([0.4, 0.5, 0.6])
        }
        mock.select_key.return_value = ("Test key", "Test value")
        
        # Mock is_empty to return True after two calls
        mock.is_empty.side_effect = [False, False, True]
        
        return mock
    
    @patch("src.rl_agent.ada_embedding")
    def test_collect_trajectory(self, mock_ada_embedding, mock_model, mock_database):
        """Test collecting a trajectory."""
        # Setup mocks
        mock_ada_embedding.return_value = np.array([0.7, 0.8, 0.9])
        
        # Create collector
        collector = TrajectoryCollector(model=mock_model, database=mock_database)
        
        # Collect trajectory
        trajectory = collector.collect_trajectory()
        
        # Verify
        assert mock_database.reset.called
        assert mock_model.generate_query.called
        assert mock_ada_embedding.called
        assert mock_database.select_key.called
        
        # Should have collected 2 steps (based on our mock is_empty setup)
        assert len(trajectory) == 2
        
        # Check structure of trajectory steps
        for step in trajectory:
            assert "query" in step
            assert "query_embedding" in step
            assert "key_id" in step
            assert "key" in step
            assert "value" in step
            assert "similarities" in step
            assert "probs" in step
    
    def test_collect_batch_trajectories(self, mock_model, mock_database):
        """Test collecting multiple trajectories."""
        # Create collector with a mock collect_trajectory method
        collector = TrajectoryCollector(model=mock_model, database=mock_database)
        collector.collect_trajectory = MagicMock(return_value=[{"dummy": "trajectory"}])
        
        # Collect batch
        batch_size = 3
        trajectories = collector.collect_batch_trajectories(batch_size)
        
        # Verify
        assert collector.collect_trajectory.call_count == batch_size
        assert len(trajectories) == batch_size


class TestReinforcementLearner:
    """Test cases for the ReinforcementLearner class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock language model."""
        mock = MagicMock()
        mock.device = "cpu"
        mock.calculate_trajectory_reward.return_value = 0.5
        return mock
    
    @pytest.fixture
    def mock_baseline_model(self):
        """Create a mock baseline model."""
        mock = MagicMock()
        mock.device = "cpu"
        return mock
    
    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer."""
        return MagicMock()
    
    def test_initialization(self, mock_model, mock_baseline_model, mock_optimizer):
        """Test ReinforcementLearner initialization."""
        # Create learner
        learner = ReinforcementLearner(
            model=mock_model,
            baseline_model=mock_baseline_model,
            optimizer=mock_optimizer,
            learning_rate=0.001,
            kl_weight=0.1
        )
        
        # Verify
        assert learner.model == mock_model
        assert learner.baseline_model == mock_baseline_model
        assert learner.optimizer == mock_optimizer
        assert learner.kl_weight == 0.1
        assert isinstance(learner.baseline_rewards, list)
        assert isinstance(learner.filtered_indices, list)
    
    def test_compute_trajectory_reward(self, mock_model, mock_baseline_model):
        """Test computing trajectory reward."""
        # Create learner with a mock optimizer to avoid parameter issues
        mock_optimizer = MagicMock()
        
        learner = ReinforcementLearner(
            model=mock_model,
            baseline_model=mock_baseline_model,
            optimizer=mock_optimizer  # Use the mock optimizer instead of creating a real one
        )
        
        # Compute reward
        trajectory = [{"query": "q1", "value": "v1"}, {"query": "q2", "value": "v2"}]
        reward = learner.compute_trajectory_reward(trajectory)
        
        # Verify
        assert mock_model.calculate_trajectory_reward.called
        assert reward == 0.5  # From our mock setup
    
    def test_compute_policy_loss(self, mock_model):
        """Test computing policy loss."""
        # Create a mock optimizer to avoid parameter issues
        mock_optimizer = MagicMock()
        
        # Create learner
        learner = ReinforcementLearner(model=mock_model, optimizer=mock_optimizer)
        
        # Setup test data
        trajectory = [
            {
                "key_id": "key_1",
                "probs": {"key_1": 0.7, "key_2": 0.3}
            },
            {
                "key_id": "key_2",
                "probs": {"key_1": 0.2, "key_2": 0.8}
            }
        ]
        reward = 1.0
        baseline = 0.5
        
        # Compute loss
        loss = learner.compute_policy_loss(trajectory, reward, baseline)
        
        # Verify
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    @patch("src.rl_agent.compute_running_stats")
    def test_update_policy_warmup(self, mock_compute_stats, mock_model, mock_optimizer):
        """Test policy update during warmup phase."""
        # Create learner
        learner = ReinforcementLearner(model=mock_model, optimizer=mock_optimizer)
        
        # Setup test data
        trajectories = [[{"dummy": "data"}], [{"dummy": "data"}]]
        rewards = [0.5, 0.7]
        
        # Update policy during warmup
        result = learner.update_policy(trajectories, rewards)
        
        # Verify
        assert len(learner.baseline_rewards) == 2
        assert result["loss"] == 0.0  # No update during warmup
        assert not mock_optimizer.step.called
    
    @patch("src.rl_agent.compute_running_stats")
    def test_update_policy_after_warmup(self, mock_compute_stats, mock_model, mock_optimizer):
        """Test policy update after warmup phase."""
        # Create learner
        learner = ReinforcementLearner(model=mock_model, optimizer=mock_optimizer)
        
        # Setup test data
        learner.baseline_rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                    1.1, 1.2, 1.3, 1.4, 1.5]  # 15 warmup episodes
        mock_compute_stats.return_value = (0.5, 0.1)  # mean, std
        
        trajectories = [[{"dummy": "data"}], [{"dummy": "data"}]]
        rewards = [0.7, 0.5]  # First reward > threshold (0.5 + 0.1), second <= threshold
        
        # Mock policy loss computation
        learner.compute_policy_loss = MagicMock(return_value=torch.tensor(0.3, requires_grad=True))
        
        # Update policy after warmup
        result = learner.update_policy(trajectories, rewards)
        
        # Verify
        assert mock_optimizer.step.called
        assert result["filtered_count"] == 1  # Only first trajectory should pass filter
        assert result["loss"] > 0  # Should have a loss value


@patch("src.rl_agent.ReinforcementLearner")
@patch("src.rl_agent.TrajectoryCollector")
def test_run_training(mock_collector_class, mock_learner_class):
    """Test the run_training function."""
    # Create mocks directly instead of using fixtures
    mock_model = MagicMock()
    mock_database = MagicMock()
    
    # Setup mocks
    mock_collector = MagicMock()
    mock_collector_class.return_value = mock_collector
    
    mock_learner = MagicMock()
    mock_learner_class.return_value = mock_learner
    
    # Mock train_episode result
    result = {
        "loss": 0.1,
        "reward_mean": 0.5,
        "reward_std": 0.1,
        "filtered_count": 2,
        "rewards": [0.4, 0.5, 0.6],
        "trajectories": [{"dummy": "trajectory"}]
    }
    
    # Set up our train_episode to be mocked within the module
    with patch("src.rl_agent.train_episode", return_value=result):
        # Run training
        num_episodes = 5
        history = run_training(
            model=mock_model,
            database=mock_database,
            num_episodes=num_episodes,
            batch_size=4,
            learning_rate=0.001,
            kl_weight=0.1,
            checkpoint_dir="test_checkpoints",
            save_interval=2
        )
        
        # Verify
        assert mock_collector_class.called
        assert mock_learner_class.called
        assert len(history["losses"]) == num_episodes
        assert len(history["rewards"]) == num_episodes * 3  # 3 rewards per episode in our mock 