"""
Main script for running the Attention-Guided RL training.
"""

import os
import argparse
import logging
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any

import wandb

from config import (
    MODEL_NAME,
    WIKI_ARTICLE_TITLE,
    MAX_PAIRS,
    NUM_EPISODES,
    BATCH_SIZE,
    LEARNING_RATE,
    KL_WEIGHT,
    CHECKPOINT_DIR,
    LOGS_DIR,
    LOG_INTERVAL,
    SAVE_MODEL_INTERVAL,
    WANDB_PROJECT,
    WANDB_ENTITY
)
from model import load_language_model, create_baseline_model
from data import load_wikipedia_article
from utils import ensure_dir, plot_rewards
from rl_agent import run_training
from embeddings import initialize_embedding_service

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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Attention-Guided RL model")
    
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="Language model to use")
    parser.add_argument("--article", type=str, default=WIKI_ARTICLE_TITLE,
                        help="Wikipedia article to use")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES,
                        help="Number of episodes to train")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--kl-weight", type=float, default=KL_WEIGHT,
                        help="KL divergence weight")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                        help="Directory to save checkpoints")
    parser.add_argument("--logs-dir", type=str, default=LOGS_DIR,
                        help="Directory to save logs")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--save-interval", type=int, default=SAVE_MODEL_INTERVAL,
                        help="Save model checkpoint every N episodes")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Load model from checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit precision")
    
    return parser.parse_args()

def setup_experiment_directories(args):
    """Set up directories for experiment artifacts."""
    # Create a unique experiment ID based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{timestamp}_{args.article.replace(' ', '_')}"
    
    # Set up directories
    checkpoint_dir = os.path.join(args.checkpoint_dir, experiment_id)
    logs_dir = os.path.join(args.logs_dir, experiment_id)
    
    ensure_dir(checkpoint_dir)
    ensure_dir(logs_dir)
    
    return {
        "id": experiment_id,
        "checkpoint_dir": checkpoint_dir,
        "logs_dir": logs_dir
    }

def save_experiment_config(config, directory):
    """Save experiment configuration to a JSON file."""
    config_path = os.path.join(directory, "config.json")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved experiment config to: {config_path}")

def setup_wandb(experiment_id, config):
    """Initialize Weights & Biases for experiment tracking."""
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=experiment_id,
        config=config
    )
    logger.info(f"Initialized W&B with experiment ID: {experiment_id}")

def save_history(history, logs_dir):
    """Save training history to files."""
    # Save raw history data
    history_path = os.path.join(logs_dir, "history.json")
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, np.ndarray):
                serializable_history[key] = value.tolist()
            else:
                serializable_history[key] = value
        
        json.dump(serializable_history, f, indent=2)
    
    # Create and save plots
    if "rewards" in history and "filtered_counts" in history:
        # Plot rewards
        rewards_plot_path = os.path.join(logs_dir, "rewards.png")
        filtered_indices = [i for i, count in enumerate(history["filtered_counts"]) if count > 0]
        plot_rewards(history["rewards"], filtered_indices, rewards_plot_path)
    
    if "losses" in history:
        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(history["losses"])
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(logs_dir, "losses.png"))
    
    logger.info(f"Saved training history to: {logs_dir}")

def log_to_wandb(history):
    """Log metrics to Weights & Biases."""
    if not wandb.run:
        return
    
    # Log scalar metrics
    for episode in range(len(history.get("losses", []))):
        metrics = {
            "loss": history["losses"][episode] if episode < len(history.get("losses", [])) else None,
            "filtered_count": history["filtered_counts"][episode] if episode < len(history.get("filtered_counts", [])) else None,
        }
        
        if "grad_norms" in history and episode < len(history["grad_norms"]):
            metrics["grad_norm"] = history["grad_norms"][episode]
        
        wandb.log(metrics, step=episode)
    
    # Log reward plots
    if "rewards" in history and "filtered_counts" in history:
        filtered_indices = [i for i, count in enumerate(history["filtered_counts"]) if count > 0]
        reward_fig = plt.figure(figsize=(10, 6))
        
        # Plot all rewards
        plt.plot(range(len(history["rewards"])), history["rewards"], 'b-', alpha=0.5, label='All rewards')
        
        # Highlight filtered rewards
        filtered_rewards = [history["rewards"][i] for i in filtered_indices]
        plt.plot(filtered_indices, filtered_rewards, 'ro', label='Filtered rewards')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Rewards over Episodes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        wandb.log({"reward_plot": wandb.Image(reward_fig)})
        plt.close(reward_fig)
    
    logger.info("Logged metrics to W&B")

def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up experiment directories
    experiment = setup_experiment_directories(args)
    
    # Save experiment configuration
    config = vars(args)
    config["experiment_id"] = experiment["id"]
    save_experiment_config(config, experiment["logs_dir"])
    
    # Initialize W&B if requested
    if args.use_wandb:
        setup_wandb(experiment["id"], config)
    
    # Initialize embedding service
    initialize_embedding_service()
    
    # Load the language model
    logger.info(f"Loading language model: {args.model}")
    model = load_language_model(
        model_name=args.model,
        device=args.device,
        load_in_8bit=args.load_in_8bit
    )
    
    # Load from checkpoint if specified
    if args.load_checkpoint:
        logger.info(f"Loading model from checkpoint: {args.load_checkpoint}")
        model.load_checkpoint(args.load_checkpoint)
    
    # Create baseline model if needed
    # baseline_model = create_baseline_model(model)
    # For simplicity, we'll skip the baseline model for now
    
    # Load the Wikipedia article data
    logger.info(f"Loading Wikipedia article: {args.article}")
    database = load_wikipedia_article(
        title=args.article,
        max_pairs=MAX_PAIRS
    )
    
    # Run training
    logger.info(f"Starting training for {args.episodes} episodes")
    start_time = time.time()
    
    history = run_training(
        model=model,
        database=database,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        kl_weight=args.kl_weight,
        checkpoint_dir=experiment["checkpoint_dir"],
        save_interval=args.save_interval
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save training history
    save_history(history, experiment["logs_dir"])
    
    # Log to W&B
    if args.use_wandb:
        log_to_wandb(history)
        wandb.finish()
    
    logger.info("All done!")

if __name__ == "__main__":
    main() 