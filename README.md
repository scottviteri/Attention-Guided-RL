# Attention-Guided Reinforcement Learning for Self-Directed Language Model Training

This project implements an RL-based active learning framework for language models that guides the ordering of training data to maximize learning efficiency.

## Overview

In this system:
- A base language model (meta-llama/Llama-3.2-3B-Instruct) generates queries
- Query embeddings (via OpenAI's ADA) compute attention scores against key-value pairs from a Wikipedia article
- An RL agent chooses the order in which these pairs are added to the context window
- The reward is defined as the average conditional log probability over the entire trajectory
- Policy gradients with KL divergence regularization update the query-generation policy

## Features

- Natural language active learning using Wikipedia articles
- Efficient use of context windows with key-value pairs
- Self-directed curriculum learning via reinforcement learning
- Stable training through KL divergence regularization
- Comprehensive logging and trajectory filtering

## Installation

```bash
git clone https://github.com/yourusername/Attention-Guided-RL.git
cd Attention-Guided-RL
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py
```

## Configuration

Adjust hyperparameters in `src/config.py`:
- Learning rate
- KL divergence weight
- Number of episodes
- Batch size
- Token counts for queries, keys, and values

## Testing

```bash
pytest tests/
```

## Project Structure

```
Attention-Guided-RL/
├── README.md                # Project overview and instructions
├── requirements.txt         # Dependencies
├── src/
│   ├── main.py              # Main training loop
│   ├── model.py             # LM loading and query generation
│   ├── rl_agent.py          # Policy gradient functions
│   ├── embeddings.py        # ADA embedding functions
│   ├── data.py              # Wikipedia article processing
│   ├── utils.py             # Utility functions
│   └── config.py            # Hyperparameters
└── tests/
    ├── test_model.py        # Tests for model functions
    ├── test_rl_agent.py     # Tests for policy updates
    ├── test_embeddings.py   # Tests for embedding functions
    └── test_data.py         # Tests for data processing
```

## License

[MIT License](LICENSE) 