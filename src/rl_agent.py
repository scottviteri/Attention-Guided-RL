"""
Functions for reinforcement learning agent and policy gradient updates.
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque

from src.config import (
    KL_WEIGHT,
    BATCH_SIZE,
    QUERY_TOKEN_COUNT,
    WARMUP_EPISODES,
    REWARD_SYSTEM_PROMPT
)
from src.utils import (
    softmax,
    sample_from_distribution,
    compute_running_stats,
    compute_kl_divergence
)
from src.model import LanguageModel
from src.data import KeyValueDatabase
from src.embeddings import ada_embedding

# Set up logging
logger = logging.getLogger(__name__)

class TrajectoryCollector:
    """
    Class for collecting trajectories by sampling actions from the policy.
    """
    
    def __init__(
        self, 
        model: LanguageModel,
        database: KeyValueDatabase,
        temperature: float = 1.0
    ):
        """
        Initialize the trajectory collector.
        
        Args:
            model: Language model for query generation
            database: Key-value database for retrieving content
            temperature: Temperature for controlling exploration
        """
        self.model = model
        self.database = database
        self.temperature = temperature
        
        logger.info(f"Initialized TrajectoryCollector with temperature: {temperature}")
    
    def collect_trajectory(self) -> List[Dict[str, Any]]:
        """
        Collect a single trajectory by rolling out the policy.
        
        Returns:
            List of steps in the trajectory
        """
        # Reset the database
        self.database.reset()
        
        # Get the article title
        article_title = self.database.title
        
        # Initialize empty context and trajectory
        context = ""
        trajectory = []
        
        # Continue until all key-value pairs are selected
        while not self.database.is_empty():
            # Generate query based on current context
            query = self.model.generate_query(
                context=context,
                fixed_token_count=QUERY_TOKEN_COUNT,
                temperature=self.temperature,
                article_title=article_title
            )
            
            # Compute query embedding
            query_embedding = ada_embedding(query)
            
            # Compute similarities with available keys
            available_key_embeddings = self.database.get_available_key_embeddings()
            similarities = {}
            
            for key_id, key_embedding in available_key_embeddings.items():
                similarity = np.dot(query_embedding, key_embedding)
                similarities[key_id] = float(similarity)
            
            # Convert similarities to probabilities
            similarity_values = np.array(list(similarities.values()))
            probs = softmax(similarity_values, temperature=self.temperature)
            
            # Sample a key based on probabilities
            selected_idx = sample_from_distribution(probs)
            selected_key_id = list(similarities.keys())[selected_idx]
            
            # Get corresponding key-value pair
            key, value = self.database.select_key(selected_key_id)
            
            # Append to context
            if context:
                context += f" Query: {query} Value: {value} "
            else:
                context = f"Query: {query} Value: {value} "
            
            # Record step in trajectory
            trajectory.append({
                "query": query,
                "query_embedding": query_embedding,
                "key_id": selected_key_id,
                "key": key,
                "value": value,
                "similarities": similarities,
                "probs": {k: float(v) for k, v in zip(similarities.keys(), probs)}
            })
        
        return trajectory
    
    def collect_batch_trajectories(self, batch_size: int = BATCH_SIZE) -> List[List[Dict[str, Any]]]:
        """
        Collect multiple trajectories in parallel.
        
        Args:
            batch_size: Number of trajectories to collect
            
        Returns:
            List of trajectories
        """
        # For simplicity, collect trajectories sequentially
        # In a more advanced implementation, this could be parallelized
        trajectories = []
        
        for _ in range(batch_size):
            trajectory = self.collect_trajectory()
            trajectories.append(trajectory)
        
        return trajectories

class ReinforcementLearner:
    """
    Class for reinforcement learning using policy gradients.
    """
    
    def __init__(
        self, 
        model: LanguageModel,
        baseline_model: Optional[LanguageModel] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        learning_rate: float = 1e-5,
        kl_weight: float = KL_WEIGHT
    ):
        """
        Initialize the reinforcement learner.
        
        Args:
            model: Language model to train
            baseline_model: Baseline model for reward normalization
            optimizer: PyTorch optimizer or None to create a new one
            learning_rate: Learning rate for optimizer
            kl_weight: Weight for KL divergence regularization
        """
        self.model = model
        self.baseline_model = baseline_model
        self.kl_weight = kl_weight
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.model.parameters(),
                lr=learning_rate
            )
        else:
            self.optimizer = optimizer
        
        # Initialize reward statistics
        self.baseline_rewards = []
        self.filtered_indices = []
        
        logger.info("Initialized ReinforcementLearner")
    
    def compute_trajectory_reward(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        Compute reward for a trajectory.
        
        Args:
            trajectory: List of steps in the trajectory
            
        Returns:
            Reward value
        """
        return self.model.calculate_trajectory_reward(
            trajectory=trajectory,
            system_prompt=REWARD_SYSTEM_PROMPT,
            baseline_model=self.baseline_model
        )
    
    def compute_policy_loss(
        self, 
        trajectory: List[Dict[str, Any]], 
        reward: float, 
        baseline: float = 0.0
    ) -> torch.Tensor:
        """
        Compute policy gradient loss for a trajectory.
        
        Args:
            trajectory: List of steps in the trajectory
            reward: Reward for the trajectory
            baseline: Baseline reward for advantage
            
        Returns:
            Loss tensor
        """
        # Compute advantage
        advantage = reward - baseline
        
        # Initialize loss
        loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
        
        # For each step in the trajectory
        for step in trajectory:
            # Get the probability of the selected action
            selected_key_id = step["key_id"]
            action_probs = step["probs"]
            action_prob = action_probs[selected_key_id]
            
            # Policy gradient loss: -log(π(a|s)) * advantage
            step_loss = -torch.log(torch.tensor(action_prob, device=self.model.device)) * advantage
            loss = loss + step_loss
        
        # Average over trajectory length
        loss = loss / len(trajectory)
        
        return loss
    
    def compute_kl_loss(
        self, 
        old_outputs: torch.Tensor, 
        new_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence loss between old and new model outputs.
        
        Args:
            old_outputs: Logits from the old model
            new_outputs: Logits from the new model
            
        Returns:
            KL divergence loss
        """
        # Compute softmax probabilities
        old_probs = torch.softmax(old_outputs, dim=-1)
        new_probs = torch.softmax(new_outputs, dim=-1)
        
        # Compute KL divergence
        kl_div = compute_kl_divergence(old_probs, new_probs)
        
        return kl_div
    
    def update_policy(
        self, 
        trajectories: List[List[Dict[str, Any]]], 
        rewards: List[float]
    ) -> Dict[str, float]:
        """
        Update policy using batch of trajectories.
        
        Args:
            trajectories: List of trajectories
            rewards: Rewards for each trajectory
            
        Returns:
            Dictionary with training metrics
        """
        # Update baseline rewards
        self.baseline_rewards.extend(rewards)
        
        # Compute mean and std of rewards
        if len(self.baseline_rewards) <= WARMUP_EPISODES:
            # During warm-up, don't update the policy
            logger.info(f"Warm-up phase: {len(self.baseline_rewards)}/{WARMUP_EPISODES}")
            return {
                "loss": 0.0,
                "reward_mean": np.mean(rewards),
                "reward_std": np.std(rewards),
                "filtered_count": 0
            }
        
        # Compute running statistics
        running_mean, running_std = compute_running_stats(self.baseline_rewards)
        
        # Filter trajectories based on reward threshold
        reward_threshold = running_mean + running_std
        filtered_trajectories = []
        filtered_rewards = []
        
        for i, (trajectory, reward) in enumerate(zip(trajectories, rewards)):
            if reward > reward_threshold:
                filtered_trajectories.append(trajectory)
                filtered_rewards.append(reward)
                self.filtered_indices.append(len(self.baseline_rewards) - len(rewards) + i)
        
        # If no trajectories pass the filter, skip the update
        if not filtered_trajectories:
            logger.info("No trajectories passed the filter, skipping update")
            return {
                "loss": 0.0,
                "reward_mean": running_mean,
                "reward_std": running_std,
                "filtered_count": 0
            }
        
        # Compute baseline for advantage
        baseline = running_mean
        
        # Accumulate gradient for each filtered trajectory
        total_loss = 0.0
        
        self.optimizer.zero_grad()
        
        for trajectory, reward in zip(filtered_trajectories, filtered_rewards):
            # Compute policy loss
            policy_loss = self.compute_policy_loss(trajectory, reward, baseline)
            
            # Add regularization (KL divergence could be computed here)
            # For simplicity, we'll skip actual KL computation in this example
            total_loss += policy_loss
        
        # Average loss over filtered trajectories
        avg_loss = total_loss / len(filtered_trajectories)
        
        # Backpropagate and update
        avg_loss.backward()
        
        # Get gradient norm for logging
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Return metrics
        return {
            "loss": avg_loss.item(),
            "grad_norm": grad_norm.item(),
            "reward_mean": running_mean,
            "reward_std": running_std,
            "filtered_count": len(filtered_trajectories)
        }

def train_episode(
    collector: TrajectoryCollector,
    learner: ReinforcementLearner,
    batch_size: int = BATCH_SIZE
) -> Dict[str, Any]:
    """
    Train for one episode (batch of trajectories).
    
    Args:
        collector: Trajectory collector
        learner: Reinforcement learner
        batch_size: Number of trajectories per batch
        
    Returns:
        Dictionary with training metrics
    """
    # Collect trajectories
    trajectories = collector.collect_batch_trajectories(batch_size)
    
    # Compute rewards
    rewards = [learner.compute_trajectory_reward(trajectory) for trajectory in trajectories]
    
    # Update policy
    metrics = learner.update_policy(trajectories, rewards)
    
    # Return metrics and trajectory info
    return {
        **metrics,
        "trajectories": trajectories,
        "rewards": rewards
    }

def run_training(
    model: LanguageModel,
    database: KeyValueDatabase,
    num_episodes: int,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = 1e-5,
    kl_weight: float = KL_WEIGHT,
    checkpoint_dir: str = "checkpoints",
    save_interval: int = 10
) -> Dict[str, Any]:
    """
    Run full training loop.
    
    Args:
        model: Language model to train
        database: Key-value database
        num_episodes: Number of episodes to train
        batch_size: Number of trajectories per batch
        learning_rate: Learning rate for optimizer
        kl_weight: Weight for KL divergence regularization
        checkpoint_dir: Directory to save checkpoints
        save_interval: Save checkpoint every N episodes
        
    Returns:
        Dictionary with training history
    """
    # Initialize baseline model
    baseline_model = None  # Or implement a baseline model
    
    # Initialize trajectory collector
    collector = TrajectoryCollector(model, database)
    
    # Initialize reinforcement learner
    learner = ReinforcementLearner(
        model=model,
        baseline_model=baseline_model,
        learning_rate=learning_rate,
        kl_weight=kl_weight
    )
    
    # Training history
    history = {
        "losses": [],
        "rewards": [],
        "filtered_counts": [],
        "grad_norms": []
    }
    
    # Training loop
    for episode in range(num_episodes):
        logger.info(f"Episode {episode+1}/{num_episodes}")
        
        # Train one episode
        result = train_episode(collector, learner, batch_size)
        
        # Log metrics
        logger.info(f"  Loss: {result['loss']:.4f}")
        logger.info(f"  Mean reward: {result['reward_mean']:.4f}")
        logger.info(f"  Filtered trajectories: {result['filtered_count']}/{batch_size}")
        
        # Update history
        history["losses"].append(result["loss"])
        history["rewards"].extend(result["rewards"])
        history["filtered_counts"].append(result["filtered_count"])
        if "grad_norm" in result:
            history["grad_norms"].append(result["grad_norm"])
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_episode_{episode+1}")
            model.save_checkpoint(checkpoint_path)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, "model_final")
    model.save_checkpoint(final_checkpoint_path)
    
    return history 