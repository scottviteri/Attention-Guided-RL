"""
Utility functions for the Attention-Guided RL project.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_dir(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Compute softmax values for array of scores.
    
    Args:
        x: Array of scores
        temperature: Temperature parameter for controlling exploration
        
    Returns:
        Softmax probabilities
    """
    x = x / temperature
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()

def sample_from_distribution(probs: np.ndarray) -> int:
    """
    Sample an index based on probability distribution.
    
    Args:
        probs: Probability distribution
        
    Returns:
        Sampled index
    """
    return np.random.choice(len(probs), p=probs)

def compute_running_stats(values: List[float], window_size: Optional[int] = None) -> Tuple[float, float]:
    """
    Compute running mean and standard deviation of values.
    
    Args:
        values: List of values
        window_size: Size of window for running stats (None for all values)
        
    Returns:
        Tuple of (mean, std)
    """
    if window_size is not None:
        recent_values = values[-window_size:]
    else:
        recent_values = values
    
    mean = np.mean(recent_values)
    std = np.std(recent_values) if len(recent_values) > 1 else 1.0
    
    return mean, std

def log_metrics(
    episode: int,
    trajectory: List[Dict[str, Any]],
    reward: float, 
    loss: Optional[float] = None,
    gradient_norm: Optional[float] = None,
    pass_filter: bool = False,
    log_interval: int = 10
) -> None:
    """
    Log metrics for training.
    
    Args:
        episode: Current episode number
        trajectory: List of trajectory steps
        reward: Reward for the trajectory
        loss: Loss value (if applicable)
        gradient_norm: Gradient norm (if applicable)
        pass_filter: Whether trajectory passed the filter
        log_interval: How often to log details
    """
    if episode % log_interval == 0:
        logger.info(f"Episode {episode}:")
        logger.info(f"  Reward: {reward:.4f}")
        if loss is not None:
            logger.info(f"  Loss: {loss:.4f}")
        if gradient_norm is not None:
            logger.info(f"  Gradient norm: {gradient_norm:.4f}")
        logger.info(f"  Passed filter: {pass_filter}")
        logger.info(f"  Trajectory length: {len(trajectory)}")
    
    # Could also log to wandb or tensorboard here
    
def plot_rewards(rewards: List[float], filtered_indices: List[int], save_path: str = "reward_plot.png") -> None:
    """
    Plot rewards over episodes.
    
    Args:
        rewards: List of rewards
        filtered_indices: Indices of episodes that passed the filter
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot all rewards
    plt.plot(range(len(rewards)), rewards, 'b-', alpha=0.5, label='All rewards')
    
    # Highlight filtered rewards
    filtered_rewards = [rewards[i] for i in filtered_indices]
    plt.plot(filtered_indices, filtered_rewards, 'ro', label='Filtered rewards')
    
    # Plot running mean
    window_size = min(50, len(rewards))
    running_mean = [np.mean(rewards[max(0, i-window_size):i+1]) for i in range(len(rewards))]
    plt.plot(range(len(rewards)), running_mean, 'g-', label='Running mean')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards over Episodes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path)
    logger.info(f"Reward plot saved to {save_path}")

def compute_kl_divergence(old_probs: torch.Tensor, new_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between two probability distributions.
    
    Args:
        old_probs: Old probability distribution (shape: batch_size x seq_len x vocab_size)
        new_probs: New probability distribution (shape: batch_size x seq_len x vocab_size)
        
    Returns:
        KL divergence value
    """
    # Add small epsilon for numerical stability
    eps = 1e-8
    old_probs = torch.clamp(old_probs, min=eps)
    
    # Apply log_softmax to new_probs directly for numerical stability
    # instead of computing softmax first and then taking the log
    log_new_probs = F.log_softmax(new_probs, dim=-1)
    
    # Use KL divergence formula with log_probs and probs directly
    kl_div = F.kl_div(log_new_probs, old_probs, reduction='batchmean')
    return kl_div

def format_query_prompt(context: str, instructions: str, article_title: str = None) -> str:
    """
    Format the prompt for query generation using INSTRUCT format.
    
    Args:
        context: Previous query-value pairs
        instructions: Instructions for generating a query
        article_title: Title of the Wikipedia article (optional)
        
    Returns:
        Formatted prompt string with correct INSTRUCT markers
    """
    from src.config import USER_START, ASSISTANT_START, EOT_TOKEN
    
    # Include article title in the prompt if provided
    article_prefix = ""
    if article_title:
        article_prefix = f"Article: {article_title}\n\n"
    
    return f"{USER_START} {article_prefix}{context} {instructions} {EOT_TOKEN} {ASSISTANT_START} "

def format_reward_context(system_prompt: str, query_key_pairs: str) -> str:
    """
    Format the context for reward calculation using INSTRUCT format.
    
    Args:
        system_prompt: Descriptive system prompt
        query_key_pairs: All query-key pairs in the trajectory
        
    Returns:
        Formatted context string with correct INSTRUCT markers
    """
    from src.config import SYSTEM_START, USER_START, EOT_TOKEN
    
    return f"{SYSTEM_START} {system_prompt} {EOT_TOKEN} {USER_START} {query_key_pairs} {EOT_TOKEN}"

def extract_value_positions(input_ids: torch.Tensor, tokenizer, value_marker: str = "Value:") -> List[Tuple[int, int]]:
    """
    Extract positions of values in the tokenized input.
    
    Args:
        input_ids: Tokenized input
        tokenizer: Tokenizer for the model
        value_marker: Marker for value content
        
    Returns:
        List of (start, end) positions for values
    """
    # Convert input_ids to string
    text = tokenizer.decode(input_ids[0])
    
    # Find all occurrences of the value marker
    value_positions = []
    marker_token_ids = tokenizer.encode(value_marker, add_special_tokens=False)
    marker_len = len(marker_token_ids)
    
    for i in range(len(input_ids[0]) - marker_len + 1):
        if input_ids[0][i:i+marker_len].tolist() == marker_token_ids:
            # Find the end of this value (next EOT_TOKEN or end of sequence)
            eot_token_ids = tokenizer.encode(EOT_TOKEN, add_special_tokens=False)
            eot_len = len(eot_token_ids)
            
            end_pos = None
            for j in range(i + marker_len, len(input_ids[0]) - eot_len + 1):
                if input_ids[0][j:j+eot_len].tolist() == eot_token_ids:
                    end_pos = j
                    break
            
            if end_pos is None:
                end_pos = len(input_ids[0])
                
            value_positions.append((i + marker_len, end_pos))
    
    return value_positions 