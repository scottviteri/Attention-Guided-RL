"""
Main script for running the reinforcement learning training.
"""

import os
import time
import logging
import argparse
# Configure matplotlib to be silent
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
# Suppress matplotlib debug logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import numpy as np
import wandb
from typing import Optional, Dict, Any, List
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import (
    WIKI_ARTICLE_TITLE,
    MAX_PAIRS,
    KL_WEIGHT,
    NUM_EPISODES,
    BATCH_SIZE,
    LEARNING_RATE
)
from src.data import load_wikipedia_article, load_random_wikipedia_article
from src.model import load_language_model
from src.embeddings import EmbeddingService, initialize_llama_embedding_service, initialize_embedding_service
from src.rl_agent import ReinforcementLearner, run_training
from src.utils import ensure_dir

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent on Wikipedia articles")
    
    # Data arguments
    parser.add_argument("--title", type=str, default=WIKI_ARTICLE_TITLE,
                        help="Title of the Wikipedia article to use")
    parser.add_argument("--random-article", action="store_true",
                        help="Use a random Wikipedia article instead of a specific title")
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS,
                        help="Maximum number of key-value pairs to extract")
    
    # Training arguments
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES,
                        help="Number of episodes to train for")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for the optimizer")
    parser.add_argument("--kl-weight", type=float, default=KL_WEIGHT,
                        help="Weight for the KL divergence term in the loss")
    
    # Misc arguments
    parser.add_argument("--enable-wandb", action="store_true",
                        help="Enable logging to wandb (disabled by default)")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="Disable saving model checkpoints")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable embedding cache")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save outputs")
    
    return parser.parse_args()


def setup_logging(args, disable_wandb: bool = False) -> Optional[Dict[str, Any]]:
    """Set up logging to wandb."""
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    # Only enable wandb if explicitly requested
    if not args.enable_wandb or disable_wandb:
        logger.info("Wandb logging disabled")
        return None
    
    # Set up wandb
    logger.info("Setting up wandb logging")
    config = {
        "title": args.title,
        "max_pairs": args.max_pairs,
        "num_episodes": args.num_episodes,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "random_article": args.random_article,
        "no_checkpoints": args.no_checkpoints,
        "verbose": args.verbose,
        "no_cache": args.no_cache
    }
    
    wandb.init(project="attention-guided-rl", config=config)
    return config


def plot_rewards(episode_rewards: List[float], output_dir: str):
    """Plot rewards over episodes."""
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title("Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"rewards_{timestamp}.png")
    plt.savefig(output_path)
    logger.info(f"Saved rewards plot to {output_path}")
    
    # Log to wandb if available
    if wandb.run is not None:
        wandb.log({"rewards_plot": plt})


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Configure logging level based on verbose flag
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True
        )
        logger.debug("Verbose logging enabled")
    
    # Set up logging
    config = setup_logging(args)
    
    # Load the language model
    logger.info("Loading language model")
    model = load_language_model()
    
    # Initialize the embedding services
    # First initialize the Llama-based embedding service
    logger.info("Initializing Llama embedding service")
    llama_embedding_service = initialize_llama_embedding_service(
        model=model,
        use_cache=not args.no_cache
    )
    logger.info(f"Llama embedding service initialized with cache {'disabled' if args.no_cache else 'enabled'}")
    
    # Also initialize OpenAI embedding service as fallback
    if args.verbose:
        logger.debug("Initializing OpenAI embedding service as fallback")
    openai_embedding_service = initialize_embedding_service(
        use_cache=not args.no_cache
    )
    
    # Load the Wikipedia article
    logger.info("Loading Wikipedia article")
    if args.random_article:
        database = load_random_wikipedia_article(max_pairs=args.max_pairs)
        logger.info(f"Loaded random article with {len(database)} key-value pairs")
    else:
        database = load_wikipedia_article(
            title=args.title,
            max_pairs=args.max_pairs
        )
        logger.info(f"Loaded article: {args.title} with {len(database)} key-value pairs")
    
    # Run training
    logger.info("Starting training")
    start_time = time.time()
    
    episode_rewards = run_training(
        model=model,
        database=database,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        disable_checkpoints=args.no_checkpoints,
        verbose=args.verbose
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Plot rewards
    plot_rewards(episode_rewards, args.output_dir)
    
    # Clean up
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main() 