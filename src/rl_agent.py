"""
Functions for reinforcement learning agent and policy gradient updates.
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque
import bitsandbytes as bnb

from src.config import (
    KL_WEIGHT,
    BATCH_SIZE,
    WARMUP_EPISODES,
)
from src.embeddings import (
    compute_similarity,
    extract_attention_activations,
    compute_multihead_similarity
)
from src.utils import (
    softmax,
    sample_from_distribution,
    compute_running_stats,
    get_device
)
from src.model import LanguageModel, load_language_model
from src.data import KeyValuePair
from src.trajectory import Trajectory

# Constants
REWARD_SYSTEM_PROMPT = "You are a helpful AI assistant that provides accurate and useful information."

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
    verbose: bool = False,
    use_multihead: bool = True
) -> Tuple[str, str, List[KeyValuePair]]:
    """
    Select a key-value pair from a database using attention-based similarity.
    
    Args:
        query: Query string
        database: List of KeyValuePair objects
        model: Language model for computing embeddings
        temperature: Temperature for softmax sampling (higher = more random)
        verbose: Whether to print debug information
        use_multihead: Whether to use multi-head attention for similarity calculation
        
    Returns:
        Tuple of (selected_key, selected_value, updated_database)
    """
    if not database:
        return "", "", []
    
    # Device to use for all tensor operations
    device = model.device
    
    # Process query - tokenize it first
    query_inputs = model.tokenizer(
        query,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Compute query embedding using attention mechanism
    query_activations = extract_attention_activations(
        query_inputs["input_ids"], 
        model, 
        activation_type="query"
    )
    query_embedding = query_activations[0]  # Take first (only) embedding
    
    # Add batch dimension to query_embedding
    query_embedding = query_embedding.unsqueeze(0)  # [embed_dim] -> [1, embed_dim]
    
    # Get all keys from the database
    keys = [pair.key for pair in database]
    key_ids = [pair.key_id for pair in database]
    
    # Prepare key embeddings as a tensor
    key_embeddings = torch.stack([torch.tensor(pair.key_embedding, device=device) for pair in database])
    
    # Add batch dimension to key_embeddings
    key_embeddings = key_embeddings.unsqueeze(0)  # [num_keys, embed_dim] -> [1, num_keys, embed_dim]
    
    # For multihead similarity, we need to process each key text separately
    # For standard similarity, we can directly use the pre-computed key_embeddings
    if use_multihead:
        # Compute multi-head similarities using tensor-based API
        avg_similarities = compute_multihead_similarity(
            query_embedding=query_embedding,
            key_embeddings=key_embeddings,
            model=model,
            verbose=verbose
        )
        # Remove batch dimension from avg_similarities [1, num_keys] -> [num_keys]
        similarity_values = avg_similarities.squeeze(0).cpu().numpy()
    else:
        # Use standard dot product similarity with tensor-based API
        similarities = compute_similarity(query_embedding, key_embeddings)
        # Remove batch dimension from similarities [1, num_keys] -> [num_keys]
        similarity_values = similarities.squeeze(0).cpu().numpy()
    
    # Apply temperature and softmax
    probabilities = softmax(similarity_values / temperature)
    
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
        
    def collect_trajectory(
        self, 
        query: str, 
        num_steps: int = 1, 
        temperature: float = 1.0,
        verbose: bool = False,
        use_multihead: bool = True
    ) -> Dict[str, Any]:
        """
        Collect a trajectory of key-value pairs from the database using the query as a prompt.
        
        Args:
            query: Initial query
            num_steps: Number of steps to take in the trajectory
            temperature: Temperature for sampling (higher = more random)
            verbose: Whether to print debug information
            use_multihead: Whether to use multi-head attention for similarity calculation
            
        Returns:
            Dictionary with the trajectory information
        """
        logger.info(f"Collecting trajectory for query: {query}")
        trajectory = []
        
        # Initial query
        current_query = query
        
        # Collect trajectory
        for step in range(num_steps):
            # Select a key-value pair (with attention)
            selected_key, selected_value, _ = select_value_with_attention(
                current_query, 
                self.database.data, 
                self.model,
                temperature=temperature,
                verbose=verbose,
                use_multihead=use_multihead
            )
            
            # Log the selected key-value pair
            if verbose:
                logger.info(f"Step {step} query: {current_query}")
                logger.info(f"Step {step} selected: {selected_key}")
            
            # Add to trajectory
            trajectory.append({
                "query": current_query,
                "key": selected_key,
                "value": selected_value
            })
            
            # Update the query
            current_query = selected_value
            
        # Return the trajectory
        return {
            "initial_query": query,
            "steps": trajectory
        }

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
        self, 
        initial_query: str, 
        target_answer: str, 
        num_steps: int = 5,
        temperature: float = 1.0,
        update_steps: int = 1,
        verbose: bool = False,
        use_multihead: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model for a single episode using the given query and target answer.
        
        Args:
            initial_query: Initial query to start the trajectory
            target_answer: Target answer to measure reward against
            num_steps: Number of steps in the trajectory
            temperature: Temperature for sampling (higher = more random)
            update_steps: Number of gradient update steps per episode
            verbose: Whether to print debug information
            use_multihead: Whether to use multi-head attention for similarity calculation
            
        Returns:
            Dictionary with the training information
        """
        # Collect trajectory
        trajectory = self.collect_trajectory(
            query=initial_query,
            num_steps=num_steps,
            temperature=temperature,
            verbose=verbose,
            use_multihead=use_multihead
        )
        
        # Compute rewards
        rewards = self.compute_trajectory_rewards(trajectory)
        
        # Update policy
        update_stats = self.update_policy(trajectory, rewards)
        
        # Combine stats
        stats = {
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_min': np.min(rewards),
            'reward_max': np.max(rewards),
            **update_stats
        }
        
        return stats

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
    # Collect trajectory
    trajectory = collector.collect_trajectory(
        query=REWARD_SYSTEM_PROMPT,
        num_steps=5,
        verbose=verbose
    )
    
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
    model_name: str,
    data_path: str,
    output_dir: str,
    num_episodes: int = 10,
    batch_size: int = 1,
    log_interval: int = 1,
    save_interval: int = 5,
    temperature: float = 1.0,
    learning_rate: float = 1e-5,
    kl_weight: float = 0.1,
    verbose: int = 1,
    use_multihead: bool = True,
    device: Optional[str] = None
) -> None:
    """
    Run the reinforcement learning training loop.
    
    Args:
        model_name: Name of the HuggingFace model to use
        data_path: Path to the dataset
        output_dir: Directory to save models and logs
        num_episodes: Number of episodes to train for
        batch_size: Batch size for training
        log_interval: Log interval (in episodes)
        save_interval: Save interval (in episodes)
        temperature: Temperature for sampling
        learning_rate: Learning rate for policy updates
        kl_weight: Weight for KL penalty in the loss function
        verbose: Verbosity level (0=quiet, 1=info, 2=debug)
        use_multihead: Whether to use multi-head attention
        device: Device to use for model and tensors (None for auto-detection)
    """
    # Set up logging
    setup_logging(verbose)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the appropriate device for training
    device_to_use = get_device(device)
    device_str = str(device_to_use)
    logger.info(f"Using device: {device_str}")
    
    # Initialize components
    logger.info(f"Loading model: {model_name}")
    model = LanguageModel(model_name, device=device_str)
    
    logger.info(f"Loading dataset: {data_path}")
    database = KeyValueDatabase.from_file(data_path)
    
    # Initialize collector and learner
    collector = TrajectoryCollector(model, database, temperature=temperature)
    learner = ReinforcementLearner(model, learning_rate=learning_rate, kl_weight=kl_weight)
    
    # Training loop
    logger.info(f"Starting training for {num_episodes} episodes")
    rewards_list = []
    for episode in range(num_episodes):
        logger.info(f"Episode {episode+1}/{num_episodes}")
        
        # Train for one episode
        stats = train_episode(
            collector=collector,
            learner=learner,
            batch_size=batch_size,
            verbose=(verbose >= 2),  # Show detailed logs at debug level or higher
        )
        
        # Store rewards
        rewards_list.append(stats.get('reward_mean', 0.0))
        
        # Log statistics
        if (episode + 1) % log_interval == 0:
            log_stats(stats, episode)
        
        # Save model
        if (episode + 1) % save_interval == 0:
            save_path = os.path.join(output_dir, f"model_episode_{episode+1}")
            model.save(save_path)
            logger.info(f"Saved model to {save_path}")
    
    # Save final model
    final_path = os.path.join(output_dir, "model_final")
    model.save(final_path)
    logger.info(f"Saved final model to {final_path}")
    logger.info("Training complete!")
    
    return rewards_list 