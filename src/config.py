"""
Configuration parameters for the Attention-Guided RL project.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

# API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Model parameters
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Training parameters
LEARNING_RATE = 1e-5
KL_WEIGHT = 0.01
NUM_EPISODES = 1000
WARMUP_EPISODES = 15
BATCH_SIZE = 8  # Number of trajectories to process at once

# Token parameters
QUERY_TOKEN_COUNT = 32  # Fixed token count for queries
KEY_TOKEN_COUNT = 24    # Fixed token count for keys
VALUE_TOKEN_COUNT = 40  # Fixed token count for values

# Wikipedia article parameters
WIKI_ARTICLE_TITLE = "Artificial intelligence"  # Default article to use
MAX_PAIRS = 10  # Maximum number of key-value pairs to extract
MIN_ARTICLE_TOKENS = 200  # Minimum tokens required in an article to process it
CHUNK_SIZE = 24  # Size of each chunk in tokens
STRIDE = 12  # Stride for overlapping chunks
MAX_CHUNKS = 50  # Maximum number of chunks to create from article

# Special tokens
USER_START = "<|start_header_id|>user<|end_header_id|>"
SYSTEM_START = "<|start_header_id|>system<|end_header_id|>"
ASSISTANT_START = "<|start_header_id|>assistant<|end_header_id|>"
EOT_TOKEN = "<|eot_id|>"

# Instructions for query generation
QUERY_INSTRUCTIONS = """
Now, generate a single specific question that would help you learn the most 
important information from this article that hasn't been covered yet. Create a 
focused, direct question without any preamble or explanation.
Generate your question immediately:
"""

# System prompt for reward calculation
REWARD_SYSTEM_PROMPT = """
Evaluate the log probability of the following query-value sequence.
"""

# Logging parameters
LOG_INTERVAL = 10  # Log metrics every N episodes
SAVE_MODEL_INTERVAL = 0  # Set to 0 to disable checkpointing by default
WANDB_PROJECT = "attention-guided-rl"  # W&B project name
WANDB_ENTITY = None  # W&B entity name

# Paths
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
LOGS_DIR = "logs"

@dataclass(frozen=True)
class Config:
    """
    A frozen dataclass representing all configuration settings for the project.
    
    This provides a single place to access all configuration and ensures immutability.
    """
    # Model parameters
    model_name: str = MODEL_NAME 
    # Training parameters
    learning_rate: float = LEARNING_RATE
    kl_weight: float = KL_WEIGHT
    num_episodes: int = NUM_EPISODES
    warmup_episodes: int = WARMUP_EPISODES
    batch_size: int = BATCH_SIZE
    
    # Token parameters
    query_token_count: int = QUERY_TOKEN_COUNT
    key_token_count: int = KEY_TOKEN_COUNT
    value_token_count: int = VALUE_TOKEN_COUNT
    
    # Wikipedia article parameters
    wiki_article_title: str = WIKI_ARTICLE_TITLE
    max_pairs: int = MAX_PAIRS
    min_article_tokens: int = MIN_ARTICLE_TOKENS
    chunk_size: int = CHUNK_SIZE
    stride: int = STRIDE
    max_chunks: int = MAX_CHUNKS
    
    # Special tokens
    user_start: str = USER_START
    system_start: str = SYSTEM_START
    assistant_start: str = ASSISTANT_START
    eot_token: str = EOT_TOKEN
    
    # Instructions
    query_instructions: str = QUERY_INSTRUCTIONS
    reward_system_prompt: str = REWARD_SYSTEM_PROMPT
    
    # Logging parameters
    log_interval: int = LOG_INTERVAL
    save_model_interval: int = SAVE_MODEL_INTERVAL
    wandb_project: str = WANDB_PROJECT
    wandb_entity: Optional[str] = WANDB_ENTITY
    
    # Paths
    data_dir: str = DATA_DIR
    checkpoint_dir: str = CHECKPOINT_DIR
    logs_dir: str = LOGS_DIR
    
    # API keys
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))

# Create a default config instance
DEFAULT_CONFIG = Config() 