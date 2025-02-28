"""
Configuration parameters for the Attention-Guided RL project.
"""

import os

# API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Model parameters
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
EMBEDDING_MODEL = "text-embedding-ada-002"

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