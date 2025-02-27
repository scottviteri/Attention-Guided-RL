"""
Configuration parameters for the Attention-Guided RL project.
"""

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
MAX_PAIRS = 50  # Maximum number of key-value pairs to extract

# Special tokens
USER_START = "<|start_header_id|>user<|end_header_id|>"
SYSTEM_START = "<|start_header_id|>system<|end_header_id|>"
ASSISTANT_START = "<|start_header_id|>assistant<|end_header_id|>"
EOT_TOKEN = "<|eot_id|>"

# Instructions for query generation
QUERY_INSTRUCTIONS = """
Based on the context above, generate a query that would help you learn the most 
from the available information. Your query should help select the most informative 
next piece of content from the article.
"""

# System prompt for reward calculation
REWARD_SYSTEM_PROMPT = """
Evaluate the log probability of the following query-value sequence.
"""

# Logging parameters
LOG_INTERVAL = 10  # Log metrics every N episodes
SAVE_MODEL_INTERVAL = 100  # Save model checkpoint every N episodes
WANDB_PROJECT = "attention-guided-rl"  # W&B project name
WANDB_ENTITY = None  # W&B entity name

# Paths
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
LOGS_DIR = "logs" 