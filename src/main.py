"""
Main script for running the reinforcement learning training.
"""
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

from src.config import (
    MODEL_NAME,
    KL_WEIGHT,
    NUM_EPISODES,
    BATCH_SIZE,
    LEARNING_RATE
)
from src.data import get_wikipedia_kv_stream
from src.model import load_language_model
from src.utils import ensure_dir, plot_rewards
from src.rl_agent import run_training

# Set up basic logging (will be overridden by verbosity settings)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent on Wikipedia articles")
    
    # Data arguments
    parser.add_argument("--random-article", action="store_true", 
                        help="Use a random Wikipedia article (default behavior)")
    
    # Model arguments
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None,
                        help="Device to use for model loading (default: auto-detect)")
    
    # Training arguments
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES,
                        help="Number of episodes to train for")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for the optimizer")
    parser.add_argument("--kl-weight", type=float, default=KL_WEIGHT,
                        help="Weight for the KL divergence term in the loss")
    
    # Logging and output control
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument("--quiet", action="store_true",
                        help="Minimize output, show only warnings and errors")
    verbosity_group.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging (INFO level)")
    verbosity_group.add_argument("--debug", action="store_true",
                        help="Enable debug logging (DEBUG level)")
    verbosity_group.add_argument("--trace", action="store_true",
                        help="Enable trace logging (most detailed output)")
    
    # Misc arguments
    parser.add_argument("--enable-wandb", action="store_true",
                        help="Enable logging to wandb (disabled by default)")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="Disable saving model checkpoints")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable embedding cache")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save outputs")
    
    return parser.parse_args()


def setup_logging(args):
    """
    Configure logging based on verbosity level.
    
    Args:
        args: Command line arguments
    
    Hierarchy of verbosity (from least to most verbose):
    - quiet: Only warnings and errors (WARNING)
    - default: Standard information (INFO)
    - verbose: More detailed information (still INFO but with additional logs)
    - debug: Detailed debugging information (DEBUG)
    - trace: Most detailed output possible (DEBUG with all loggers verbose)
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if args.quiet:
        # Quiet mode - show only warnings and errors
        logging.basicConfig(level=logging.WARNING, format=log_format, force=True)
        logger.warning("Quiet mode enabled - showing only warnings and errors")
    elif args.trace:
        # Trace mode - show all possible details (most verbose)
        logging.basicConfig(level=logging.DEBUG, format=log_format, force=True)
        # Set specific loggers to DEBUG level for maximum verbosity
        for module in ["src", "transformers", "bitsandbytes"]:
            logging.getLogger(module).setLevel(logging.DEBUG)
        logger.debug("Trace logging enabled (most detailed output)")
    elif args.debug:
        # Debug mode - show detailed debug information
        logging.basicConfig(level=logging.DEBUG, format=log_format, force=True)
        logger.debug("Debug logging enabled")
    elif args.verbose:
        # Verbose mode - more detailed information than default
        logging.basicConfig(level=logging.INFO, format=log_format, force=True)
        logger.info("Verbose logging enabled - showing detailed information")
    else:
        # Default mode - standard information output
        logging.basicConfig(level=logging.INFO, format=log_format, force=True)
        logger.info("Standard logging level enabled")


def setup_wandb(args, disable_wandb: bool = False) -> Optional[Dict[str, Any]]:
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
        "num_episodes": args.num_episodes,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "random_article": args.random_article,
        "no_checkpoints": args.no_checkpoints,
        "verbose": args.verbose,
        "debug": args.debug,
        "trace": args.trace,
        "no_cache": args.no_cache
    }
    
    wandb.init(project="attention-guided-rl", config=config)
    return config


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Configure logging based on verbosity level
    setup_logging(args)
    
    # Set up wandb if enabled
    config = setup_wandb(args)
    
    # Load the language model
    logger.info("Loading language model")
    model = load_language_model(device=args.device)
    if args.device:
        logger.info(f"Using device: {args.device}")
    else:
        logger.info(f"Using device: {model.device} (auto-detected)")
    
    # Load Wikipedia articles
    logger.info("Loading Wikipedia article")
    stream = get_wikipedia_kv_stream(device=args.device)
    
    # Get the first article from the stream
    database = next(stream)
    logger.info(f"Loaded article with {len(database)} key-value pairs")
    
    # Run training
    logger.info("Starting training")
    start_time = time.time()
    
    # Determine verbosity level to pass to run_training
    verbose_level = 0  # Default
    if args.trace:
        verbose_level = 3  # Most verbose
    elif args.debug:
        verbose_level = 2  # Debug level
    elif args.verbose:
        verbose_level = 1  # Standard verbose
    
    episode_rewards = run_training(
        model=model,
        database=database,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        disable_checkpoints=args.no_checkpoints,
        verbose=verbose_level
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