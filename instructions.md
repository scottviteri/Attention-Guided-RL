Below is a comprehensive blueprint document titled **"Attention-Guided Reinforcement Learning for Self-Directed Language Model Training"**. This document details the motivation, technical design, and implementation plan—enough to guide the development of a full PyTorch/pytest GitHub repository.

---

# Attention-Guided Reinforcement Learning for Self-Directed Language Model Training

## Abstract

This project proposes an RL-based active learning framework for language models that guides the ordering of training data to maximize learning efficiency. In our setup, a base language model (meta-llama/Llama-3.2-3.B-Instruct) generates queries whose embeddings (via OpenAI's ADA) are used to compute attention scores against a fixed set of key–value pairs derived from a Wikipedia article. As the RL agent chooses the order in which these pairs are added to the context window, it receives a reward defined as the average conditional log probability over the entire trajectory (normalized against an untrained baseline). Policy gradients—regularized with a KL divergence term comparing successive model outputs—are used to update the query-generation policy. Detailed logging and a filtering mechanism (only trajectories >1 standard deviation above the running mean, computed after a warm-up phase) ensure robust training and debugging.

## 1. Motivation

- **Bridging Natural Language and Active Learning:**\
  Traditional active learning with language models often uses off-distribution tasks (e.g., Boolean formulas) that demand abstract reasoning. Our method shifts to natural language by using a fixed Wikipedia article. This leverages the language model's familiarity with natural text while still requiring self-directed learning.

- **Efficient Use of Context Windows:**\
  By pre-embedding key–value pairs (consecutive sentences) from the article and then ordering them to fill the available context window, we avoid wasting tokens on indices or redundant text. Each pair is used exactly once, and once selected, its key is removed from the lookup table.

- **Self-Directed Curriculum via RL:**\
  The system trains the model to sequence its own training examples. The RL agent (acting as a "teacher") learns to select an ordering that maximizes learning progress, measured by the average log probability on subsequent queries. This is achieved by reward feedback, with a filtering mechanism to focus on trajectories that exceed a dynamically computed threshold.

- **Stable Training through Regularization:**\
  A KL divergence penalty is applied during gradient updates to ensure that the new model's probability distribution does not drift too far from the previous one, stabilizing the RL training process.

## 2. System Architecture

### 2.1. Components

- **Base Language Model:**

  - *Model:* meta-llama/Llama-3.2-3.B-Instruct
  - *Role:* Generate queries given the current context; its output guides data ordering.

- **Key–Value Memory Database:**

  - *Source:* A single Wikipedia article is segmented into \(n\) pairs of consecutive sentences.
  - *Keys:* Each key is an ADA embedding of the first sentence in the pair.
  - *Values:* The corresponding second sentence of the pair.
  - *Mechanism:* When a key–value pair is selected, its key is removed from the database to ensure non-redundant ordering.

- **Query and Key Embedding:**

  - *Embedding Model:* OpenAI's ADA embedding service.
  - *Usage:* Both queries (from the language model) and keys (pre-computed from Wikipedia) are embedded. The similarity is computed via the inner product, and a softmax over these inner products produces a probability distribution.

- **RL Agent:**

  - *Action:* Choosing the order in which the key–value pairs are added to the context.
  - *Reward:* The average conditional log probability over queries in the entire trajectory. Each value's log probability is normalized by subtracting the log probability obtained when an untrained model generated the previous query.
  - *Optimization:* Policy gradient updates are applied only on trajectories that are more than 1 standard deviation above the running mean reward (after a 15-datapoint warm-up). A KL penalty between the old and new model distributions is incorporated during the gradient calculation.

## 3. Methodology

### 3.1. Data Flow and Training Loop

1. **Initialization:**

   - Load the base LM (meta-llama/Llama-3.2-3.B-Instruct).
   - Pre-compute ADA embeddings for the keys extracted from the chosen Wikipedia article.
   - Initialize the RL policy (implicitly represented by the query generation of the LM).
   - Reserve the first 15 data points (or trajectories) for warm-up to compute the initial mean and standard deviation of rewards.

2. **Trajectory Generation:**

   - For each episode:
     - The current context starts empty.
     - For each step until the context window is full (i.e. all \(n\) pairs have been selected):
       1. The LM generates a query (which can condition on previous words and previous query–value pairs already in context).
       2. Embed the query using ADA.
       3. Compute the inner products with all remaining keys.
       4. Apply a softmax to obtain a probability distribution over the remaining keys.
       5. Sample an action (i.e., choose one key–value pair) according to this distribution.
       6. Remove the chosen key from the database.
       7. Append the query and its corresponding value to the context.
   - Record the trajectory (sequence of queries, chosen keys, and values).

3. **Reward Computation:**

   - After the full trajectory is built, compute the **reward** as the average conditional log probability over all queries in the trajectory.
   - Normalize each value's log probability by subtracting the log probability that would be assigned if the untrained LM had generated the preceding query.

4. **Trajectory Filtering:**

   - Only trajectories with a reward greater than 1 standard deviation above the current cumulative mean (computed after the initial 15 trajectories) are used for policy updates.

5. **Policy Update:**

   - Use the filtered trajectories to perform a policy gradient update.
   - Include a KL penalty between the old and new LM probability distributions for the tokens in the retrieved values.
   - Update the policy parameters using standard gradient descent (PyTorch's optimizer).

6. **Logging and Monitoring:**

   - Log trajectories (queries, keys, values).
   - Track gradient norms, reward evolution per trajectory step, loss vs. batch index, and the fraction of trajectories passing the filter.
   - Periodically print these logs for debugging and analysis.

### 3.2. Pseudocode Overview

```python
# Pseudocode for training loop

# Initialize base LM, ADA embedding service, key-value database
lm = load_model("meta-llama/Llama-3.2-3.B-Instruct")
keys, values = load_wikipedia_key_value_pairs(wiki_article)
keys = {k: ada_embedding(sentence) for k, sentence in keys.items()}  # Pre-computed
database = {key: value for key, value in zip(keys, values)}
optimizer = torch.optim.Adam(lm.parameters(), lr=LEARNING_RATE)
baseline_rewards = []  # for warm-up and running mean/std

for episode in range(num_episodes):
    context = []  # Start with empty context
    available_keys = database.copy()
    trajectory = []
    
    # Generate trajectory: fill the context window with all pairs
    while available_keys:
        # Generate query from LM, conditioning on current context
        query = lm.generate(context)
        query_embedding = ada_embedding(query)
        
        # Compute similarity (inner product) with remaining keys
        similarities = {k: torch.dot(query_embedding, embedding) 
                        for k, embedding in available_keys.items()}
        probs = softmax(similarities)
        
        # Sample a key based on the probability distribution
        selected_key = sample_from_distribution(probs)
        selected_value = available_keys[selected_key]
        
        # Append the query and value to the context
        context.append(query)
        context.append(selected_value)
        
        # Log the step
        trajectory.append({
            "query": query,
            "selected_key": selected_key,
            "selected_value": selected_value,
            "probs": probs
        })
        
        # Remove the selected key to avoid repetition
        del available_keys[selected_key]
    
    # Compute reward: average conditional log probability over trajectory
    reward = compute_average_log_prob(lm, trajectory, normalization=True)
    
    # Warm-up: collect initial rewards before applying filtering
    if episode < 15:
        baseline_rewards.append(reward)
        continue  # No policy update yet
    
    # Compute running mean and std from baseline_rewards + subsequent episodes
    running_mean, running_std = compute_running_stats(baseline_rewards)
    
    if reward > running_mean + running_std:
        # Calculate policy gradient loss: -log(prob(action)) * (reward - baseline)
        loss = compute_policy_loss(trajectory, reward, baseline=running_mean)
        
        # Add KL divergence regularization (between old and new LM outputs on value tokens)
        kl_loss = compute_kl_divergence(old_lm_outputs, new_lm_outputs)
        total_loss = loss + KL_WEIGHT * kl_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update baseline_rewards with the new reward
        baseline_rewards.append(reward)
    
    # Log trajectory, gradient norm, loss, reward, and filter pass rate
    log_metrics(trajectory, reward, total_loss, optimizer)
```

## 4. Implementation Details

### 4.1. Environment Setup and Repository Structure

A suggested repository structure:

```
attention_rl_lm_training/
├── README.md                # Project overview and instructions
├── requirements.txt         # PyTorch, transformers, OpenAI API, etc.
├── setup.py                 # Packaging script (if needed)
├── src/
│   ├── main.py              # Main training loop
│   ├── model.py             # LM loading and query generation utilities
│   ├── rl_agent.py          # Policy gradient functions and RL utilities
│   ├── embeddings.py        # Functions for ADA embedding calls
│   ├── data.py              # Functions to load and pre-process the Wikipedia article, extract key-value pairs
│   ├── utils.py             # Utility functions: softmax, sampling, logging, stats computation, etc.
│   └── config.py            # Hyperparameters (LEARNING_RATE, KL_WEIGHT, num_episodes, etc.)
└── tests/
    ├── test_model.py        # Pytest tests for model functions
    ├── test_rl_agent.py     # Tests for policy update functions
    ├── test_embeddings.py   # Tests for ADA embedding functions (can use mocks)
    └── test_data.py         # Tests for data loading and processing
```

### 4.2. Key Functions and Modules

- **Model Loading and Query Generation (model.py):**\
  – Function to load meta-llama/Llama-3.2-3.B-Instruct.\
  – Function `generate(context)` that produces a query given the current context.

- **Embeddings (embeddings.py):**\
  – Function `ada_embedding(text)` which wraps the OpenAI API call (or a local implementation) to compute text embeddings.\
  – Ensure that both keys and queries are processed consistently (with normalization).

- **Data Preparation (data.py):**\
  – Function to load a specific Wikipedia article and extract consecutive sentence pairs.\
  – Pre-compute embeddings for keys and store them in a dictionary.

- **RL Agent and Policy Update (rl_agent.py):**\
  – Function `compute_policy_loss(trajectory, reward, baseline)` that computes the negative log-probability loss weighted by the reward difference.\
  – Function `compute_kl_divergence(old_outputs, new_outputs)` to compute the KL penalty.\
  – The training loop integrates these to update the LM policy using PyTorch's optimizer.

- **Utilities (utils.py):**\
  – Functions for softmax, sampling from a probability distribution, computing running mean and standard deviation, and logging metrics.\
  – Logging should include gradient norms, trajectory details, reward progression, loss per batch, and filter pass fraction.

### 4.3. Hyperparameters and Training Settings (config.py)

- **Learning Rate:** e.g., 1e-5
- **KL Weight:** a scalar coefficient to balance the KL penalty
- **Number of Episodes:** Total episodes for training
- **Warm-up Count:** 15 episodes
- **Reward Filtering:** Threshold set to running mean + 1 SD
- **Batch Size:** 1 (each episode is a full trajectory)
- **Optimizer Settings:** Adam or similar

### 4.4. Testing with Pytest

- Write unit tests for each function:
  - **Embedding Tests:** Ensure `ada_embedding(text)` returns a vector of expected shape and is normalized.
  - **Data Tests:** Validate that the Wikipedia article loader correctly splits text into non-overlapping key–value pairs.
  - **Model Tests:** Mock the LM's generation to ensure queries are produced and that the policy's action distribution is correctly computed.
  - **RL Agent Tests:** Test that `compute_policy_loss` correctly scales the negative log probability by the reward difference, and that `compute_kl_divergence` returns a non-negative scalar.
  - **Utilities Tests:** Verify softmax, sampling, and statistics functions using known inputs.

### 4.5. Logging and Debugging

- Use Python's logging module or TensorBoard to record:
  - **Trajectory Logs:** Save each episode's sequence of queries, chosen keys, and corresponding values.
  - **Metrics:** Gradient norm, per-episode loss, reward, and filter pass rate.
  - **Periodic Console Prints:** For quick debugging during training, print summaries every N episodes.

### 4.6. Llama-Based Embeddings

This project supports generating embeddings directly from the Llama model rather than relying on external API calls to OpenAI. This approach offers several advantages:

#### Benefits

1. **Cost Efficiency**: Eliminates the need for external API calls to OpenAI's embedding service, saving on API costs
2. **Speed Improvement**: Generates embeddings locally, reducing network latency
3. **Semantic Alignment**: Embeddings are derived from the same model used for generation, ensuring better alignment between query understanding and embedding space
4. **Privacy**: All text processing happens locally without sending data to external services
5. **Graceful Fallback**: Maintains compatibility with OpenAI's embedding API as a fallback option

#### Technical Implementation

The implementation follows these steps:

1. Extracts hidden states from all layers of the Llama model
2. Averages these hidden states across layers and token positions
3. Normalizes the resulting vector to create a unit vector embedding

This approach captures the distributed semantic representation across the entire model rather than from a specific layer, providing rich contextual embeddings.

#### Embedding Service Design

The `EmbeddingService` class provides a consistent interface for embedding functionality:

- **Centralized Caching**: Implements cache management for all embedding operations
- **Error Handling**: Provides graceful fallbacks when embedding computation fails
- **Backend Switching**: Allows easy transition between different embedding providers
- **Batch Processing**: Optimizes performance through batched embedding generation

#### Usage

The embedding functionality is transparent to the rest of the codebase:

1. First attempts to use Llama-based embeddings
2. Falls back to OpenAI embeddings if Llama embeddings are not available
3. Supports caching mechanisms to improve performance

##### Command Line Options

- `--no-cache`: Disables embedding caching
- `--verbose`: Enables detailed logging, including information about embedding computation and cache hits/misses

#### Performance Characteristics

- **Dimensionality**: Embeddings have the same dimensionality as the Llama model's hidden size (3072 for Llama-3.2-3B)
- **Computation Time**: Typically ~30-50ms per embedding on GPU
- **Similarity Metrics**: Shows high discriminative power between different concepts

#### Future Improvements

Potential enhancements:

1. Experiment with attention-weighted pooling instead of simple averaging
2. Add layer-specific embedding options to extract from specific parts of the model
3. Add quantization options for smaller embedding footprints

## 5. Conclusion

This blueprint outlines a method—**Attention-Guided Reinforcement Learning for Self-Directed Language Model Training**—that trains a language model to self-organize its learning by optimally ordering key–value pairs from a Wikipedia article. It uses ADA embeddings for natural language retrieval, policy gradients with KL divergence regularization, and a reward filtering mechanism based on cumulative statistics. The design emphasizes reproducibility and debuggability, with thorough logging and testing practices recommended throughout the PyTorch codebase. By following this blueprint, a developer should be able to implement and experiment with this novel active learning strategy in a GitHub repository, using PyTorch and pytest.

---

This document should serve as a detailed roadmap. If any additional clarifications or refinements are needed during implementation, consider iteratively updating the design based on observed training dynamics and debugging feedback.
