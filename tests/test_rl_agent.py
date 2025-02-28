"""
Tests for the RL agent module.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, call
import dataclasses

# Import the module to test
from src.rl_agent import (
    TrajectoryCollector,
    ReinforcementLearner,
    train_episode,
    run_training
)
from src.data import KeyValuePair


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
        """Create a mock list of KeyValuePair objects."""
        # Create two KeyValuePair objects
        kv_pair1 = MagicMock(spec=KeyValuePair)
        kv_pair1.key_id = "key_1"
        kv_pair1.key_embedding = np.array([0.1, 0.2, 0.3])
        kv_pair1.key_tokens = [101, 102, 103]
        kv_pair1.value_tokens = [201, 202, 203]
        
        kv_pair2 = MagicMock(spec=KeyValuePair)
        kv_pair2.key_id = "key_2"
        kv_pair2.key_embedding = np.array([0.4, 0.5, 0.6])
        kv_pair2.key_tokens = [301, 302, 303]
        kv_pair2.value_tokens = [401, 402, 403]
        
        # Return a list with two KeyValuePair objects
        return [kv_pair1, kv_pair2]
    
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
        assert len(collector.original_database) == 2  # Should have stored original database
        assert mock_model.generate_query.called
        assert mock_ada_embedding.called
        assert len(trajectory) >= 1  # At least one step in trajectory
    
    def test_collect_batch_trajectories(self, mock_model, mock_database):
        """Test collecting batch trajectories."""
        # Create collector
        collector = TrajectoryCollector(model=mock_model, database=mock_database)
        
        # Mock collect_trajectory method
        collector.collect_trajectory = MagicMock(return_value=[{"step": "test"}])
        
        # Collect batch trajectories
        batch_size = 3
        trajectories = collector.collect_batch_trajectories(batch_size=batch_size)
        
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
        mock.parameters.return_value = [torch.tensor([0.1, 0.2, 0.3])]
        mock.to.return_value = mock  # Return self when to() is called
        return mock
    
    @pytest.fixture
    def mock_trajectory_collector(self):
        """Create a mock trajectory collector."""
        mock = MagicMock()
        mock.collect_batch_trajectories.return_value = [
            [{"step": "test1"}],
            [{"step": "test2"}]
        ]
        return mock
    
    def test_train_batch(self, mock_model, mock_trajectory_collector):
        """Test training on a batch of trajectories."""
        # Create learner
        learner = ReinforcementLearner(model=mock_model)
        
        # Mock methods
        learner.compute_rewards = MagicMock(return_value=[1.0, 2.0])
        learner.compute_policy_loss = MagicMock(return_value=(torch.tensor(0.5), {}))
        
        # Mock optimizer
        mock_optimizer = MagicMock()
        
        # Train on batch
        stats = learner.train_batch(
            trajectory_collector=mock_trajectory_collector,
            optimizer=mock_optimizer,
            batch_size=2
        )
        
        # Verify
        assert mock_trajectory_collector.collect_batch_trajectories.called
        assert learner.compute_rewards.called
        assert learner.compute_policy_loss.called
        assert mock_optimizer.zero_grad.called
        assert mock_optimizer.step.called
        
        # Check stats
        assert "policy_loss" in stats
        assert "reward_mean" in stats
        assert "reward_std" in stats


@patch("src.rl_agent.TrajectoryCollector")
def test_train_episode(mock_trajectory_collector_class, mock_model):
    """Test training for one episode."""
    # Setup mocks
    mock_collector = MagicMock()
    mock_trajectory_collector_class.return_value = mock_collector
    mock_learner = MagicMock()
    mock_optimizer = MagicMock()
    mock_database = [MagicMock(spec=KeyValuePair) for _ in range(3)]
    
    # Call function
    stats = train_episode(
        model=mock_model,
        database=mock_database,
        learner=mock_learner,
        optimizer=mock_optimizer,
        batch_size=2,
        episode=1
    )
    
    # Verify
    mock_trajectory_collector_class.assert_called_once_with(
        model=mock_model,
        database=mock_database,
        temperature=1.0  # Default temperature
    )
    assert mock_learner.train_batch.called
    assert isinstance(stats, dict)


@patch("src.rl_agent.tqdm")
@patch("src.rl_agent.train_episode")
def test_run_training(mock_train_episode, mock_tqdm, mock_model):
    """Test running full training."""
    # Setup mocks
    mock_train_episode.return_value = {"policy_loss": 0.5, "reward_mean": 1.0}
    mock_tqdm.return_value = range(3)  # Mock 3 episodes
    mock_database = [MagicMock(spec=KeyValuePair) for _ in range(3)]
    
    # Call function
    stats = run_training(
        model=mock_model,
        database=mock_database,
        num_episodes=3,
        batch_size=2
    )
    
    # Verify
    assert mock_train_episode.call_count == 3
    assert isinstance(stats, list)
    assert len(stats) == 3 