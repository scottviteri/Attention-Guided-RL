"""
Functions for reinforcement learning agent and policy gradient updates.
"""

import os
import torch
import numpy as np
import logging
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque

from src.config import (
    KL_WEIGHT,
    BATCH_SIZE,
    QUERY_TOKEN_COUNT,
    WARMUP_EPISODES,
    REWARD_SYSTEM_PROMPT
)
from src.embeddings import compute_embedding, batch_compute_embeddings
from src.utils import (
    softmax,
    sample_from_distribution,
    compute_running_stats,
    compute_kl_divergence
)
from src.model import LanguageModel
from src.data import KeyValuePair

# Set up logging
logger = logging.getLogger(__name__)

class TrajectoryCollector:
    """
    Class for collecting trajectories by sampling actions from the policy.
    """
    
    def __init__(
        self, 
        model: LanguageModel,
        database: List[KeyValuePair],
        temperature: float = 1.0
    ):
        """
        Initialize the trajectory collector.
        
        Args:
            model: Language model for query generation
            database: List of KeyValuePair objects
            temperature: Temperature for controlling exploration
        """
        self.model = model
        self.database = database
        self.original_database = database.copy()  # Store a copy for reset
        self.temperature = temperature
        self.title = f"Wikipedia Article"  # Default title
        
        # Get article title if available
        if len(database) > 0 and hasattr(database[0], 'title'):
            self.title = database[0].title
        
        logger.info(f"Initialized TrajectoryCollector with temperature: {temperature}")
    
    def collect_trajectory(self, verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Collect a single trajectory by rolling out the policy.
        
        Args:
            verbose: Whether to enable verbose logging
        
        Returns:
            List of steps in the trajectory
        """
        # Reset the database
        self.reset()
        
        # Get the article title
        article_title = self.title
        
        # Initialize empty context and trajectory
        context = ""
        trajectory = []
        
        # Continue until all key-value pairs are selected
        step_count = 0
        while not self.is_empty():
            step_count += 1
            if verbose:
                logger.debug(f"Trajectory step {step_count}: Generating query...")
            
            # Generate query based on current context
            query = self.model.generate_query(
                context=context,
                fixed_token_count=QUERY_TOKEN_COUNT,
                temperature=self.temperature,
                article_title=article_title
            )
            
            if verbose:
                logger.debug(f"Generated query: {query}")
            
            # Compute query embedding
            # Explicitly specify this is a query embedding (is_query=True)
            query_embedding = compute_embedding(query, is_query=True)
            
            if verbose:
                logger.debug(f"Query embedding computed - shape: {query_embedding.shape}")
            
            # Compute similarities with available keys
            available_key_embeddings = self.get_available_key_embeddings()
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
            
            if verbose:
                logger.debug(f"Selected key ID: {selected_key_id} (probability: {probs[selected_idx]:.4f})")
            
            # Get corresponding key-value pair
            key, value = self.select_key(selected_key_id)
            
            if verbose:
                logger.debug(f"Retrieved value: {value[:50]}...")
            
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

    def reset(self):
        """Reset the database to its original state."""
        self.database = self.original_database.copy()
    
    def is_empty(self) -> bool:
        """Check if the database is empty."""
        return len(self.database) == 0
    
    def get_available_key_embeddings(self) -> Dict[str, np.ndarray]:
        """Get embeddings for available keys."""
        embeddings = {}
        for i, kv_pair in enumerate(self.database):
            key_id = kv_pair.key_id if kv_pair.key_id else f"key_{i}"
            embeddings[key_id] = kv_pair.key_embedding
        return embeddings
    
    def select_key(self, key_id: str) -> Tuple[str, str]:
        """Select a key-value pair and remove it from the database."""
        # Find the key-value pair with the given key_id
        selected_idx = -1
        for i, kv_pair in enumerate(self.database):
            current_key_id = kv_pair.key_id if kv_pair.key_id else f"key_{i}"
            if current_key_id == key_id:
                selected_idx = i
                break
        
        if selected_idx == -1:
            raise ValueError(f"Key ID {key_id} not found in database")
        
        # Get the key-value pair
        kv_pair = self.database.pop(selected_idx)
        
        # Convert token IDs to text using the tokenizer if available
        if hasattr(self.model, 'tokenizer') and hasattr(kv_pair, 'key_tokens') and hasattr(kv_pair, 'value_tokens'):
            key = self.model.tokenizer.decode(kv_pair.key_tokens)
            value = self.model.tokenizer.decode(kv_pair.value_tokens)
        else:
            # Fallback if tokenizer is not available or tokens are not available
            key = f"Key {key_id}"
            value = f"Value for {key_id}"
        
        return key, value
    
    def collect_batch_trajectories(self, batch_size: int = BATCH_SIZE, verbose: bool = False) -> List[List[Dict[str, Any]]]:
        """
        Collect multiple trajectories in parallel.
        
        Args:
            batch_size: Number of trajectories to collect
            verbose: Whether to enable verbose logging
            
        Returns:
            List of trajectories
        """
        # For simplicity, collect trajectories sequentially
        # In a more advanced implementation, this could be parallelized
        trajectories = []
        
        for i in range(batch_size):
            if verbose:
                logger.debug(f"Collecting trajectory {i+1}/{batch_size}")
            trajectory = self.collect_trajectory(verbose=verbose)
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
        Compute reward for a single trajectory.
        
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

    def compute_batch_trajectory_rewards(self, trajectories: List[List[Dict[str, Any]]]) -> List[float]:
        """
        Compute rewards for multiple trajectories in a batched manner for efficiency.
        
        Args:
            trajectories: List of trajectories, where each trajectory is a list of steps
            
        Returns:
            List of reward values
        """
        return self.model.calculate_batch_trajectory_rewards(
            trajectories=trajectories,
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
    batch_size: int = BATCH_SIZE,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a single training episode.
    
    Args:
        collector: Trajectory collector
        learner: Reinforcement learner
        batch_size: Number of trajectories to collect per batch
        verbose: Whether to enable verbose logging
        
    Returns:
        Dictionary with episode metrics
    """
    # Collect trajectories
    trajectories = collector.collect_batch_trajectories(batch_size, verbose)
    
    # Compute rewards for all trajectories in batched manner
    rewards = learner.compute_batch_trajectory_rewards(trajectories)
    
    # Update policy
    metrics = learner.update_policy(trajectories, rewards)
    
    return {
        "trajectories": trajectories,
        "rewards": rewards,
        "metrics": metrics
    }

def run_training(
    model: 'LanguageModel',
    database: 'List[KeyValuePair]',
    num_episodes: int = 50,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = 0.0001,
    kl_weight: float = KL_WEIGHT,
    disable_checkpoints: bool = False,
    verbose: bool = False
) -> List[float]:
    """
    Run training for the reinforcement learning agent.
    
    Args:
        model: Language model to train
        database: List of KeyValuePair objects
        num_episodes: Number of episodes to train
        batch_size: Number of trajectories per batch
        learning_rate: Learning rate for optimizer
        kl_weight: Weight for KL divergence regularization
        disable_checkpoints: Whether to disable saving model checkpoints
        verbose: Whether to enable verbose logging
        
    Returns:
        List of rewards for each episode
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
    
    # Track rewards
    episode_rewards = []
    
    # Training loop
    for episode in range(num_episodes):
        if verbose:
            logger.debug(f"Starting episode {episode+1}/{num_episodes}")
        else:
            logger.info(f"Episode {episode+1}/{num_episodes}")
        
        # Train one episode
        result = train_episode(collector, learner, batch_size, verbose)
        
        # Log metrics
        logger.info(f"  Loss: {result['loss']:.4f}")
        logger.info(f"  Mean reward: {result['reward_mean']:.4f}")
        logger.info(f"  Filtered trajectories: {result['filtered_count']}/{batch_size}")
        
        if verbose:
            logger.debug(f"  Detailed metrics:")
            logger.debug(f"    Reward std: {result.get('reward_std', 0):.4f}")
            logger.debug(f"    Gradient norm: {result.get('grad_norm', 0):.4f}")
            logger.debug(f"    Individual rewards: {result['rewards']}")
        
        # Track rewards for this episode
        episode_rewards.append(result['reward_mean'])
        
        # Log to wandb if available
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "loss": result["loss"],
                    "reward_mean": result["reward_mean"],
                    "filtered_count": result["filtered_count"],
                    "grad_norm": result.get("grad_norm", 0)
                }, step=episode)
        except ImportError:
            pass
    
    return episode_rewards 