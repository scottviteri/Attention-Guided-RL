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
import bitsandbytes as bnb

from src.config import (
    KL_WEIGHT,
    BATCH_SIZE,
    QUERY_TOKEN_COUNT,
    VALUE_TOKEN_COUNT,
    WARMUP_EPISODES,
    REWARD_SYSTEM_PROMPT
)
from src.embeddings import compute_embeddings
from src.utils import (
    softmax,
    sample_from_distribution,
    compute_running_stats
)
from src.model import LanguageModel
from src.data import KeyValuePair
from src.trajectory import Trajectory

# Set up logging
logger = logging.getLogger(__name__)

class TrajectoryCollector:
    """
    Class for collecting trajectories by rolling out the policy.
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
    
    def collect_trajectory(self, batch_size: int = 1, verbose: bool = False) -> Trajectory:
        """
        Collect trajectories by rolling out the policy.
        
        Args:
            batch_size: Number of parallel trajectories to collect
            verbose: Whether to enable verbose logging
        
        Returns:
            Batched Trajectory object with steps
        """
        # Initialize batched trajectory
        trajectory = Trajectory(batch_size=batch_size, device=self.model.device)
        
        # Initialize contexts for each batch element
        contexts = [""] * batch_size
        article_titles = [self.title] * batch_size
        
        # Reset the database for each batch element
        databases = [self.original_database.copy() for _ in range(batch_size)]
        
        # Track if each trajectory is finished
        active_trajectories = [True] * batch_size
        
        # Continue until all trajectories are complete
        step_count = 0
        while any(active_trajectories):
            step_count += 1
            if verbose:
                logger.debug(f"Trajectory step {step_count}: Generating queries...")
            
            # Generate queries based on current contexts (only for active trajectories)
            active_indices = [i for i, active in enumerate(active_trajectories) if active]
            active_contexts = [contexts[i] for i in active_indices]
            active_titles = [article_titles[i] for i in active_indices]
            
            if not active_indices:
                break
            
            # Generate queries for active trajectories
            active_queries = []
            for i, context in enumerate(active_contexts):
                query = self.model.generate_query(
                    context=context,
                    fixed_token_count=QUERY_TOKEN_COUNT,
                    temperature=self.temperature,
                    article_title=active_titles[i]
                )
                active_queries.append(query)
                
                if verbose:
                    logger.debug(f"Batch {active_indices[i]}, Generated query: {query}")
            
            # Initialize step data with empty values
            query_tokens_list = []
            value_tokens_list = []
            query_embeddings = []
            key_ids = []
            key_probs = []
            raw_queries = [""] * batch_size
            raw_values = [""] * batch_size
            
            # Compute embeddings for active queries
            active_query_embeddings = compute_embeddings(active_queries, are_queries=True)
            
            # For each active trajectory, compute similarities and select value
            for batch_idx, global_idx in enumerate(active_indices):
                query = active_queries[batch_idx]
                query_embedding = active_query_embeddings[batch_idx]
                
                # Get available key embeddings for this batch element
                available_key_embeddings = {}
                for i, kv_pair in enumerate(databases[global_idx]):
                    key_id = kv_pair.key_id if kv_pair.key_id else f"key_{i}"
                    available_key_embeddings[key_id] = kv_pair.key_embedding
                
                # If no keys left, mark trajectory as complete
                if not available_key_embeddings:
                    active_trajectories[global_idx] = False
                    continue
                
                # Compute similarities
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
                    logger.debug(f"Batch {global_idx}, Selected key ID: {selected_key_id} (probability: {probs[selected_idx]:.4f})")
                
                # Get corresponding key-value pair
                key, value = self._select_key(databases[global_idx], selected_key_id)
                
                if verbose:
                    logger.debug(f"Batch {global_idx}, Retrieved value: {value[:50]}...")
                
                # Append to context for this batch element
                if contexts[global_idx]:
                    contexts[global_idx] += f" Query: {query} Value: {value} "
                else:
                    contexts[global_idx] = f"Query: {query} Value: {value} "
                
                # Tokenize query and value
                query_tokens = self.model.tokenizer(
                    query, 
                    padding="max_length", 
                    max_length=QUERY_TOKEN_COUNT,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.squeeze(0)
                
                value_tokens = self.model.tokenizer(
                    value, 
                    padding="max_length", 
                    max_length=VALUE_TOKEN_COUNT,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.squeeze(0)
                
                # Add to batch lists
                query_tokens_list.append(query_tokens.unsqueeze(0))
                value_tokens_list.append(value_tokens.unsqueeze(0))
                query_embeddings.append(query_embedding)
                key_ids.append(selected_key_id)
                
                # Create key probabilities dictionary
                key_probs_dict = {k: float(v) for k, v in zip(similarities.keys(), probs)}
                key_probs.append(key_probs_dict)
                
                # Store raw strings
                raw_queries[global_idx] = query
                raw_values[global_idx] = value
                
                # Check if this batch element has completed its trajectory
                if not databases[global_idx]:
                    active_trajectories[global_idx] = False
            
            # If we have any active trajectories that selected values in this step
            if query_tokens_list:
                # Stack tensors for batch processing
                query_tokens_batch = torch.cat(query_tokens_list, dim=0)
                value_tokens_batch = torch.cat(value_tokens_list, dim=0)
                query_embeddings_array = np.stack(query_embeddings)
                
                # For inactive trajectories, fill with zeros or empty values
                for i in range(batch_size):
                    if i not in active_indices or raw_queries[i] == "":
                        # This batch element didn't get updated this step
                        # Use zeros or empty values
                        raw_queries[i] = "PADDING"
                        raw_values[i] = "PADDING"
                        if i not in active_indices:
                            # This ensures we have the right shape for batching
                            query_tokens_batch = torch.cat([
                                query_tokens_batch, 
                                torch.zeros(1, QUERY_TOKEN_COUNT, dtype=torch.long, device=self.model.device)
                            ], dim=0)
                            value_tokens_batch = torch.cat([
                                value_tokens_batch, 
                                torch.zeros(1, VALUE_TOKEN_COUNT, dtype=torch.long, device=self.model.device)
                            ], dim=0)
                            query_embeddings_array = np.concatenate([
                                query_embeddings_array,
                                np.zeros((1, query_embeddings_array.shape[1]))
                            ], axis=0)
                            key_ids.append("padding")
                            key_probs.append({})
                
                # Add step to trajectory
                trajectory.add_step(
                    query_tokens=query_tokens_batch,
                    value_tokens=value_tokens_batch,
                    query_embeddings=query_embeddings_array,
                    key_ids=key_ids,
                    key_probs=key_probs,
                    raw_queries=raw_queries,
                    raw_values=raw_values
                )
        
        return trajectory

    def reset(self):
        """Reset the database to its original state."""
        self.database = self.original_database.copy()
    
    def _select_key(self, database: List[KeyValuePair], key_id: str) -> Tuple[str, str]:
        """
        Select a key-value pair from a database and remove it.
        
        Args:
            database: List of KeyValuePair objects
            key_id: ID of the key to select
            
        Returns:
            Tuple of (key, value) strings
        """
        # Find the key-value pair with the given key_id
        selected_idx = -1
        for i, kv_pair in enumerate(database):
            current_key_id = kv_pair.key_id if kv_pair.key_id else f"key_{i}"
            if current_key_id == key_id:
                selected_idx = i
                break
        
        if selected_idx == -1:
            raise ValueError(f"Key ID {key_id} not found in database")
        
        # Get the key-value pair
        kv_pair = database.pop(selected_idx)
        
        # Convert token IDs to text using the tokenizer if available
        if hasattr(self.model, 'tokenizer') and hasattr(kv_pair, 'key_tokens') and hasattr(kv_pair, 'value_tokens'):
            key = self.model.tokenizer.decode(kv_pair.key_tokens)
            value = self.model.tokenizer.decode(kv_pair.value_tokens)
        else:
            # Fallback if tokenizer is not available or tokens are not available
            key = f"Key {key_id}"
            value = f"Value for {key_id}"
        
        return key, value

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
        Initialize reinforcement learner.
        
        Args:
            model: Language model to train
            baseline_model: Optional frozen model for baseline computation
            optimizer: Optional optimizer (defaults to 8bit Adam)
            learning_rate: Learning rate for optimizer
            kl_weight: Weight for KL divergence penalty
        """
        self.model = model
        self.baseline_model = baseline_model
        self.kl_weight = kl_weight
        
        # Initialize optimizer with 8bit Adam from bitsandbytes
        if optimizer is None:
            self.optimizer = bnb.optim.Adam8bit(
                self.model.model.parameters(),
                lr=learning_rate
            )
            print("Using 8bit Adam optimizer")
        else:
            self.optimizer = optimizer
        
        # Store baseline rewards for warm-up phase
        self.baseline_rewards = []
        self.filtered_indices = []
        
        logger.info("Initialized ReinforcementLearner")
    
    def compute_trajectory_rewards(self, trajectory: Trajectory) -> List[float]:
        """
        Compute rewards for a batch of trajectories.
        If a baseline model is available, normalizes rewards by subtracting baseline rewards.
        
        Args:
            trajectory: Batched trajectory
            
        Returns:
            List of rewards, one per batch element
        """
        # Get reward contexts
        contexts = trajectory.get_reward_contexts(system_prompt="Evaluate the log probability")
        
        # Compute rewards using current model
        rewards = self.model.calculate_trajectory_rewards(contexts, baseline_model=None)
        
        # If baseline model exists, compute baseline rewards and normalize
        if self.baseline_model is not None:
            # Compute baseline rewards using frozen model
            baseline_rewards = self.baseline_model.calculate_trajectory_rewards(contexts, baseline_model=None)
            
            # Normalize by subtracting baseline
            normalized_rewards = [r - b for r, b in zip(rewards, baseline_rewards)]
            return normalized_rewards
        
        # Return unnormalized rewards if no baseline model
        return rewards
    
    def compute_policy_loss(
        self, 
        trajectory: Trajectory, 
        rewards: Union[float, List[float]], 
        baseline: float = 0.0
    ) -> torch.Tensor:
        """
        Compute policy gradient loss for a trajectory.
        
        Args:
            trajectory: Trajectory object with steps
            rewards: Reward for the trajectory or list of rewards for batch
            baseline: Baseline reward for advantage
            
        Returns:
            Loss tensor
        """
        # Handle batched vs. single reward
        if isinstance(rewards, (int, float)):
            batch_rewards = [rewards]
        else:
            batch_rewards = rewards
            
        # Make sure we have a reward for each batch element
        if len(batch_rewards) != trajectory.get_batch_size():
            raise ValueError(f"Mismatch between rewards length ({len(batch_rewards)}) and batch size ({trajectory.get_batch_size()})")
        
        # Compute advantage for each batch element
        advantages = [reward - baseline for reward in batch_rewards]
        
        # Initialize loss
        loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
        
        # For each step in the trajectory
        for step in range(len(trajectory)):
            for b in range(trajectory.get_batch_size()):
                # Get the key probabilities for this batch element at this step
                key_probs_dict = trajectory.key_probs[step][b]
                
                # Skip if this is a padding step
                if not key_probs_dict:
                    continue
                
                # Get the selected key ID
                selected_key_id = trajectory.key_ids[step][b]
                
                # Skip if this is a padding step
                if selected_key_id == "padding":
                    continue
                
                # Get the probability of the selected action
                action_prob = key_probs_dict[selected_key_id]
                
                # Policy gradient loss: -log(π(a|s)) * advantage
                step_loss = -torch.log(torch.tensor(action_prob, device=self.model.device)) * advantages[b]
                loss = loss + step_loss
        
        # Average over trajectory length and batch size
        loss = loss / (len(trajectory) * trajectory.get_batch_size())
        
        return loss
    
    def compute_kl_loss(
        self, 
        old_outputs: torch.Tensor, 
        new_outputs: torch.Tensor,
        value_positions: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Compute KL divergence loss between old and new model outputs.
        
        Args:
            old_outputs: Logits from the old model
            new_outputs: Logits from the new model
            value_positions: List of (start, end) positions for values.
                             All batch elements are expected to have the same positions.
            
        Returns:
            KL divergence loss
        """
        # Create a mask for the positions we want to include
        mask = torch.zeros_like(old_outputs[:, :, 0], dtype=torch.bool)
        
        # Fill in the mask for value positions
        for start, end in value_positions:
            # Skip the tag tokens, only include content tokens
            for pos in range(start, end):
                if pos < old_outputs.size(1):
                    mask[:, pos] = True
        
        # Apply the mask to keep only value tokens for KL calculation
        # We need to expand mask to match the shape of old_outputs and new_outputs
        expanded_mask = mask.unsqueeze(-1).expand_as(old_outputs)
        
        # Use the mask to select only relevant positions
        masked_old_outputs = old_outputs[expanded_mask].view(-1, old_outputs.size(-1))
        masked_new_outputs = new_outputs[expanded_mask].view(-1, new_outputs.size(-1))
        
        # Compute softmax probabilities for the masked outputs
        old_probs = torch.softmax(masked_old_outputs, dim=-1)
        log_new_probs = torch.log_softmax(masked_new_outputs, dim=-1)
        
        # Compute KL divergence directly using F.kl_div for numerical stability
        # Note: F.kl_div expects log probabilities as first argument
        kl_div = torch.nn.functional.kl_div(
            log_new_probs, 
            old_probs, 
            reduction='batchmean', 
            log_target=False
        )
        
        return kl_div
    
    def update_policy(
        self, 
        trajectory: Trajectory, 
        rewards: List[float]
    ) -> Dict[str, float]:
        """
        Update policy using batch of trajectories.
        
        Args:
            trajectory: Batched Trajectory containing multiple trajectories
            rewards: Rewards for each trajectory
            
        Returns:
            Dictionary with training metrics
        """
        # Update baseline rewards
        self.baseline_rewards.extend(rewards)
        
        # Compute mean and std of rewards
        if len(self.baseline_rewards) <= WARMUP_EPISODES:
            # Still in warm-up phase, don't update policy
            logger.info(f"Warmup phase: {len(self.baseline_rewards)}/{WARMUP_EPISODES} episodes")
            return {
                "loss": 0.0, 
                "kl_loss": 0.0, 
                "policy_loss": 0.0,
                "grad_norm": 0.0,
                "reward_mean": np.mean(rewards), 
                "reward_std": np.std(rewards),
                "filtered_ratio": 0.0
            }
        
        # Compute filter threshold using running stats
        mean, std = compute_running_stats(self.baseline_rewards)
        threshold = mean + std
        
        # Filter trajectories that exceed the threshold
        filtered_indices = [i for i, r in enumerate(rewards) if r > threshold]
        if not filtered_indices:
            logger.info(f"No trajectories passed the filter (threshold: {threshold:.4f})")
            return {
                "loss": 0.0, 
                "kl_loss": 0.0, 
                "policy_loss": 0.0,
                "grad_norm": 0.0,
                "reward_mean": mean, 
                "reward_std": std,
                "filtered_ratio": 0.0
            }
        
        logger.info(f"{len(filtered_indices)}/{len(rewards)} trajectories passed the filter")
        
        # Get reward contexts for filtered trajectories
        filtered_contexts = [trajectory.get_reward_contexts()[i] for i in filtered_indices]
        filtered_rewards = [rewards[i] for i in filtered_indices]
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Tokenize filtered contexts
        inputs = self.model.tokenizer(
            filtered_contexts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.model.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # Extract value positions using only the first batch element
        # since all batch elements should have the same structure
        value_positions = self.model._extract_value_positions(input_ids[:1])
        
        # Forward pass with old model
        with torch.no_grad():
            old_outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            old_logits = old_outputs.logits.detach()
        
        # Forward pass with updated model
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        
        # Compute policy loss
        policy_loss = self.compute_policy_loss(
            trajectory=trajectory,
            rewards=filtered_rewards,
            baseline=mean
        )
        
        # Compute KL divergence loss
        kl_loss = self.compute_kl_loss(
            old_outputs=old_logits,
            new_outputs=logits,
            value_positions=value_positions
        )
        
        # Total loss
        total_loss = policy_loss + self.kl_weight * kl_loss
        
        # Backward pass and optimization
        total_loss.backward()
        
        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.model.parameters(),
            max_norm=1.0
        )
        
        # Update model parameters
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "grad_norm": grad_norm.item(),
            "reward_mean": mean,
            "reward_std": std,
            "filtered_ratio": len(filtered_indices) / len(rewards)
        }

def train_episode(
    collector: TrajectoryCollector,
    learner: ReinforcementLearner,
    batch_size: int = BATCH_SIZE,
    baseline_shift: float = 0.0,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Train for a single episode.
    
    Args:
        collector: TrajectoryCollector to collect trajectories
        learner: ReinforcementLearner to update policy
        batch_size: Number of trajectories per batch
        baseline_shift: Additional shift for the baseline (can help with exploration)
        verbose: Whether to enable verbose logging
        
    Returns:
        Dictionary of training metrics
    """
    # Collect batched trajectory
    trajectory = collector.collect_trajectory(batch_size=batch_size, verbose=verbose)
    
    # Compute rewards
    rewards = learner.compute_trajectory_rewards(trajectory)
    
    # Update policy
    update_stats = learner.update_policy(trajectory, rewards)
    
    # Combine stats
    stats = {
        'reward_mean': np.mean(rewards),
        'reward_std': np.std(rewards),
        'reward_min': np.min(rewards),
        'reward_max': np.max(rewards),
        **update_stats
    }
    
    return stats

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
    # Initialize baseline model - use a fresh instance of the same pretrained model
    baseline_model = LanguageModel(
        model_name=model.model_name,
        device=model.device
    )
    logger.info("Initialized baseline model for reward normalization")
    
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
    running_reward_stats = {"mean": 0.0, "std": 1.0}
    
    # Track best model
    best_reward = float("-inf")
    
    # Training loop
    for episode in range(1, num_episodes + 1):
        start_time = time.time()
        
        # Warm-up period: no policy updates
        if episode <= WARMUP_EPISODES:
            logger.info(f"Episode {episode}/{num_episodes} (warm-up)")
            # Collect trajectories but don't update policy
            trajectory = collector.collect_trajectory(batch_size=batch_size, verbose=verbose)
            rewards = learner.compute_trajectory_rewards(trajectory)
            episode_reward = np.mean(rewards)
            duration = time.time() - start_time
            
            logger.info(f"Episode {episode}: Reward = {episode_reward:.4f}, Duration = {duration:.2f}s")
            
            # Update reward stats
            if episode == 1:
                running_reward_stats["mean"] = episode_reward
                running_reward_stats["std"] = 1.0
            else:
                running_reward_stats = compute_running_stats(
                    rewards=episode_rewards + [episode_reward], 
                    window_size=max(1, min(len(episode_rewards), 5))
                )
            
            episode_rewards.append(episode_reward)
            continue
        
        # Regular episodes
        logger.info(f"Episode {episode}/{num_episodes}")
        
        # Train for one episode
        stats = train_episode(
            collector=collector,
            learner=learner,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Extract reward
        episode_reward = stats["reward_mean"]
        episode_rewards.append(episode_reward)
        
        # Update running stats for standardization
        running_reward_stats = compute_running_stats(
            rewards=episode_rewards, 
            window_size=max(1, min(len(episode_rewards), 5))
        )
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log metrics
        logger.info(
            f"Episode {episode}: Reward = {episode_reward:.4f}, "
            f"Loss = {stats.get('loss', 0):.4f}, "
            f"Duration = {duration:.2f}s"
        )
        
        # Track best model
        if episode_reward > best_reward and not disable_checkpoints:
            best_reward = episode_reward
            # Save checkpoint
            model_path = f"checkpoints/episode_{episode}_reward_{episode_reward:.4f}"
            model.save_checkpoint(model_path)
            logger.info(f"New best model saved to {model_path}")
    
    return episode_rewards 