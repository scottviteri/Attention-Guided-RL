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
from src.embeddings import (
    compute_embeddings, 
    tokenize_text, 
    tokenize_text_batch, 
    extract_attention_activations, 
    extract_attention_activations_batch,
    compute_attention_query_embedding,
    compute_similarity
)
from src.utils import (
    softmax,
    sample_from_distribution,
    compute_running_stats
)
from src.model import LanguageModel, load_language_model
from src.data import KeyValuePair
from src.trajectory import Trajectory

# Set up logging
logger = logging.getLogger(__name__)

def select_key(database: List[KeyValuePair], key_id: str, tokenizer=None) -> Tuple[str, str, List[KeyValuePair]]:
    """
    Select a key-value pair from a database by key_id without modifying the original database.
    
    Args:
        database: List of KeyValuePair objects
        key_id: ID of the key to select
        tokenizer: Optional tokenizer to decode tokens
        
    Returns:
        Tuple of (key, value, updated_database) where updated_database has the selected item removed
    """
    # Create a copy of the database to avoid modifying the original
    updated_database = database.copy()
    
    # Find the key-value pair with the matching key_id
    selected_pair = None
    for i, pair in enumerate(updated_database):
        if pair.key_id == key_id:
            selected_pair = pair
            updated_database.pop(i)
            break
    
    if selected_pair is None:
        raise ValueError(f"Key ID {key_id} not found in database")
    
    # Get key and value strings, decode tokens if available
    key = selected_pair.key
    value = selected_pair.value
    
    # If tokenizer is provided and tokens are available, decode them
    if tokenizer:
        if selected_pair.key_tokens is not None:
            key = tokenizer.decode(selected_pair.key_tokens)
        if selected_pair.value_tokens is not None:
            value = tokenizer.decode(selected_pair.value_tokens)
    
    # Return the key, value, and updated database
    return key, value, updated_database

def select_value_with_attention(
    query: str,
    database: List[KeyValuePair],
    model: LanguageModel,
    temperature: float = 1.0,
    verbose: bool = False
) -> Tuple[str, str, List[KeyValuePair]]:
    """
    Select a key-value pair from the database using attention-based similarity.
    Uses scaled dot product attention (scaled by sqrt(d)) instead of normalization.
    
    Args:
        query: Query string to match against keys
        database: List of KeyValuePair objects
        model: Language model for computing embeddings
        temperature: Temperature for softmax sampling (higher = more random)
        verbose: Whether to print debug information
        
    Returns:
        Tuple of (selected_key, selected_value, updated_database)
    """
    if not database:
        return "", "", []
    
    # Compute query embedding (without normalization)
    query_embedding = compute_attention_query_embedding(query, model)
    
    # Get all keys from the database
    keys = [pair.key for pair in database]
    key_ids = [pair.key_id for pair in database]
    
    # Compute key embeddings (without normalization)
    key_embeddings = {}
    for i, key in enumerate(keys):
        key_embedding = compute_attention_query_embedding(key, model)
        key_embeddings[key_ids[i]] = key_embedding
    
    # Compute similarities (now using scaled dot product)
    similarities = compute_similarity(query_embedding, key_embeddings)
    
    # Convert to list for sampling
    similarity_values = [similarities[key_id] for key_id in key_ids]
    
    # Apply temperature and softmax
    probabilities = softmax(np.array(similarity_values) / temperature)
    
    # Sample based on probabilities
    selected_idx = sample_from_distribution(probabilities)
    selected_key_id = key_ids[selected_idx]
    
    if verbose:
        # Print top 3 most similar keys
        top_indices = np.argsort(similarity_values)[::-1][:3]
        logger.info(f"Query: {query}")
        logger.info(f"Top 3 similar keys:")
        for idx in top_indices:
            logger.info(f"  {keys[idx]}: {similarity_values[idx]:.4f} (prob: {probabilities[idx]:.4f})")
        logger.info(f"Selected: {keys[selected_idx]}")
    
    # Select the key-value pair
    return select_key(database, selected_key_id, model.tokenizer)

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
        self.device = model.device  # Store the device from the model
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
        Collect a trajectory by repeatedly generating a query, selecting a key-value pair,
        and updating the context.
        
        Args:
            batch_size: Number of trajectories to collect in parallel
            verbose: Whether to print debug information
            
        Returns:
            Trajectory object containing the collected steps
        """
        if batch_size != 1:
            raise NotImplementedError("Batch size > 1 not yet implemented for trajectory collection")
        
        trajectory = Trajectory()
        context = ""
        current_db = self.database.copy()
        
        # Continue until we've used all key-value pairs or reached max turns
        max_turns = len(self.database)
        for turn in range(max_turns):
            if verbose:
                logger.info(f"\nTurn {turn + 1}/{max_turns}")
                logger.info(f"Context length: {len(context)}")
                logger.info(f"Database size: {len(current_db)}")
            
            # Generate query based on current context
            query = self.model.generate_query(
                context=context, 
                temperature=self.temperature
            )
            
            if verbose:
                logger.info(f"Generated query: {query}")
            
            # If database is empty, break
            if not current_db:
                break
                
            # Use attention-based value selection
            selected_key, selected_value, current_db = select_value_with_attention(
                query=query,
                database=current_db,
                model=self.model,
                temperature=self.temperature,
                verbose=verbose
            )
            
            if verbose:
                logger.info(f"Selected key: {selected_key}")
                logger.info(f"Selected value: {selected_value}")
            
            # Update context with the new query-value pair
            if context:
                # Add separator if needed
                context += " "
            context += f"Query: {query} Value: {selected_value}"
            
            # Add step to trajectory
            trajectory.add_step(
                query=query,
                key=selected_key,
                value=selected_value,
                context=context
            )
            
            # If database is now empty, we're done
            if not current_db:
                break
        
        return trajectory

    def reset(self):
        """Reset the database to its original state."""
        self.database = self.original_database.copy()

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
    verbose: int = 0
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
        verbose: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace)
        
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
            trajectory = collector.collect_trajectory(
                batch_size=batch_size, 
                verbose=(verbose >= 3)  # Only show detailed trajectory logs at trace level
            )
            rewards = learner.compute_trajectory_rewards(trajectory)
            episode_reward = np.mean(rewards)
            duration = time.time() - start_time
            
            logger.info(f"Episode {episode}: Reward = {episode_reward:.4f}, Duration = {duration:.2f}s")
            
            # Update reward stats
            if episode == 1:
                running_reward_stats["mean"] = episode_reward
                running_reward_stats["std"] = 1.0
            else:
                # Update moving average of rewards for filtering
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
            verbose=(verbose >= 2)  # Show detailed logs at debug level or higher
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
        
        # Detailed logging at debug level or higher
        if verbose >= 2:
            logger.debug(
                f"Episode {episode} details: Policy loss = {stats.get('policy_loss', 0):.4f}, "
                f"KL loss = {stats.get('kl_loss', 0):.4f}, "
                f"Gradient norm = {stats.get('grad_norm', 0):.4f}, "
                f"Filtered ratio = {stats.get('filtered_ratio', 0):.2f}"
            )
        
        # Track best model
        if episode_reward > best_reward and not disable_checkpoints:
            best_reward = episode_reward
            # Save checkpoint
            model_path = f"checkpoints/episode_{episode}_reward_{episode_reward:.4f}"
            model.save_checkpoint(model_path)
            logger.info(f"New best model saved to {model_path}")
    
    return episode_rewards 