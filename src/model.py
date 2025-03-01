"""
Functions for loading language models and generating queries.
"""

import os
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)

from src.config import (
    MODEL_NAME,
    QUERY_TOKEN_COUNT,
    USER_START,
    ASSISTANT_START,
    EOT_TOKEN,
    QUERY_INSTRUCTIONS
)
from src.utils import format_query_prompt, format_reward_context, ensure_dir

# Set up logging
logger = logging.getLogger(__name__)

class LanguageModel:
    """
    Class for loading and using language models.
    """
    
    def __init__(
        self, 
        model_name: str = MODEL_NAME,
        device: str = None,
        cache_dir: str = "model_cache",
        load_in_8bit: bool = False 
    ):
        """
        Initialize the language model.
        
        Args:
            model_name: Name of the language model
            device: Device to load the model on (None for auto-detection)
            cache_dir: Directory to cache models
            load_in_8bit: Whether to load model in 8-bit precision for memory efficiency
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        ensure_dir(self.cache_dir)
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading model {model_name} on {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Ensure the tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Set padding side to left for decoder-only models
        self.tokenizer.padding_side = 'left'
        
        # Load model
        model_kwargs = {"cache_dir": cache_dir}
        if load_in_8bit and self.device == "cuda":
            model_kwargs["load_in_8bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            **model_kwargs
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        # Set up generation config
        self.gen_config = GenerationConfig.from_pretrained(model_name)
        
        logger.info(f"Model loaded successfully: {model_name}")
    
    def generate_query(
        self, 
        context: str,
        instructions: str = QUERY_INSTRUCTIONS,
        fixed_token_count: int = QUERY_TOKEN_COUNT,
        temperature: float = 0.7,
        article_title: str = None
    ) -> str:
        """
        Generate a query given the context.
        
        Args:
            context: Context for generation (previous query-value pairs)
            instructions: Instructions for query generation
            fixed_token_count: Fixed number of tokens to generate
            temperature: Temperature for generation
            article_title: Title of the Wikipedia article (optional)
            
        Returns:
            Generated query
        """
        # Format the prompt with INSTRUCT markers
        prompt = format_query_prompt(context, instructions, article_title)
        
        # Tokenize with proper handling
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with exact token count
        with torch.no_grad():
            generation_output = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                min_new_tokens=fixed_token_count,
                max_new_tokens=fixed_token_count,
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=temperature,
                do_sample=True
            )
        
        # Extract only the assistant's response (excluding the prompt)
        query_tokens = generation_output[0][inputs.input_ids.shape[1]:]
        query = self.tokenizer.decode(query_tokens, skip_special_tokens=False)
        
        return query
    
    def generate_queries(
        self, 
        contexts: List[str],
        instructions: str = QUERY_INSTRUCTIONS,
        fixed_token_count: int = QUERY_TOKEN_COUNT,
        temperature: float = 0.7,
        article_titles: List[str] = None
    ) -> List[str]:
        """
        Generate queries for multiple contexts in batch.
        
        Args:
            contexts: List of context strings
            instructions: Instructions for query generation
            fixed_token_count: Fixed token count for each query
            temperature: Temperature for generation
            article_titles: List of article titles (optional)
            
        Returns:
            List of generated queries
        """
        if not contexts:
            return []
        
        # Handle article titles
        if article_titles is None:
            article_titles = ["Wikipedia Article"] * len(contexts)
        elif len(article_titles) != len(contexts):
            raise ValueError(f"Length mismatch: {len(article_titles)} titles vs {len(contexts)} contexts")
        
        # Format prompts
        prompts = []
        for context, title in zip(contexts, article_titles):
            prompt = format_query_prompt(context, instructions, title)
            prompts.append(prompt)
        
        # Generate all queries
        result_queries = []
        
        for prompt in prompts:
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Set up generation config with fixed token count
            gen_kwargs = {
                "max_new_tokens": fixed_token_count,
                "temperature": temperature,
                "pad_token_id": self.tokenizer.pad_token_id,
                "no_repeat_ngram_size": 3,
                "top_p": 0.9,
            }
            
            # Generate query
            with torch.no_grad():
                output = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **gen_kwargs
                )
            
            # Extract generated text (only the new tokens)
            output_ids = output[0, inputs.input_ids.shape[1]:]
            query = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # Clean up query if needed
            query = query.strip()
            
            if EOT_TOKEN in query:
                # Remove everything after EOT
                query = query.split(EOT_TOKEN)[0].strip()
            
            result_queries.append(query)
        
        return result_queries
    
    def generate_queries_batch(
        self, 
        contexts: List[str],
        instructions: str = QUERY_INSTRUCTIONS,
        fixed_token_count: int = QUERY_TOKEN_COUNT,
        temperature: float = 0.7,
        article_titles: List[str] = None,
        skip_padding_check: bool = False
    ) -> List[str]:
        """
        Generate queries for multiple contexts in true batch mode.
        
        This method processes all contexts in a single batch operation
        for maximum efficiency. All inputs are padded to the same length,
        and token generation is set to exactly QUERY_TOKEN_COUNT steps.
        
        Args:
            contexts: List of context strings
            instructions: Instructions for query generation
            fixed_token_count: Fixed token count for each query
            temperature: Temperature for generation
            article_titles: List of article titles (optional)
            skip_padding_check: Skip the check for padding tokens (for testing)
            
        Returns:
            List of generated queries
        """
        if not contexts:
            return []
        
        # Handle article titles
        if article_titles is None:
            article_titles = ["Wikipedia Article"] * len(contexts)
        elif len(article_titles) != len(contexts):
            raise ValueError(f"Length mismatch: {len(article_titles)} titles vs {len(contexts)} contexts")
        
        # Format prompts
        prompts = []
        for context, title in zip(contexts, article_titles):
            prompt = format_query_prompt(context, instructions, title)
            prompts.append(prompt)
        
        # Tokenize all prompts in a batch
        batch_inputs = self.tokenizer(
            prompts, 
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Create list of tokens that should be blocked to prevent early termination
        bad_words_ids = []
        for token in ["<", "</", "</query", "</query>"]:
            token_ids = self.tokenizer(token, add_special_tokens=False).input_ids
            bad_words_ids.append(token_ids)
        
        # Verify there are no padding tokens in the inputs except at the left side
        # where they are added intentionally for proper batching
        if not skip_padding_check:
            pad_token_id = self.tokenizer.pad_token_id
            for i, input_seq in enumerate(batch_inputs.input_ids):
                # Convert to numpy for easier processing
                input_seq_np = input_seq.cpu().numpy()
                # Find positions of all pad tokens
                pad_positions = np.where(input_seq_np == pad_token_id)[0]
                
                if len(pad_positions) > 0:
                    # Check if pad tokens are continuous from the start
                    # This allows padding tokens at the beginning (left-padding)
                    expected_positions = np.arange(len(pad_positions))
                    if not np.array_equal(pad_positions, expected_positions):
                        logger.warning(f"Warning: Padding tokens may be present in non-initial positions in sequence {i}")
                        # We don't fail the assertion in production, just log a warning
        
        # Generate all queries in a single batch
        with torch.no_grad():
            outputs = self.model.generate(
                batch_inputs.input_ids,
                attention_mask=batch_inputs.attention_mask,
                min_new_tokens=fixed_token_count,
                max_new_tokens=fixed_token_count,
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=temperature,
                do_sample=True,
                bad_words_ids=bad_words_ids,
                num_return_sequences=1
            )
        
        # Process outputs to extract the generated queries
        result_queries = []
        
        # Each batch element has its own input length
        for i, output in enumerate(outputs):
            # Get the original input length
            input_length = batch_inputs.input_ids[i].shape[0]
            
            # Extract only the new tokens for this batch item
            query_tokens = output[input_length:]
            
            # Decode the query
            query = self.tokenizer.decode(query_tokens, skip_special_tokens=False)
            
            # Clean up query if needed
            query = query.strip()
            
            # Remove special token markers if present
            if EOT_TOKEN in query:
                query = query.split(EOT_TOKEN)[0].strip()
            
            result_queries.append(query)
        
        return result_queries
    
    def compute_token_probabilities(
        self, 
        input_ids: torch.Tensor, 
        target_positions: List[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Compute token probabilities for input.
        
        Args:
            input_ids: Input token IDs
            target_positions: List of (start, end) positions to compute log probs for
            
        Returns:
            Log probabilities for the specified positions
        """
        inputs = {"input_ids": input_ids.to(self.device)}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Extract log probs for specific positions if specified
        if target_positions is not None:
            position_log_probs = []
            
            for start, end in target_positions:
                # For each position, get the log prob of the actual next token
                for pos in range(start, end):
                    next_token_id = input_ids[0, pos].item()
                    next_token_log_prob = log_probs[0, pos-1, next_token_id].item()
                    position_log_probs.append(next_token_log_prob)
            
            return torch.tensor(position_log_probs, device=self.device)
        
        # Otherwise return all log probs
        return log_probs
    
    def calculate_trajectory_rewards(
        self, 
        contexts: List[str], 
        baseline_model: Optional['LanguageModel'] = None
    ) -> List[float]:
        """
        Calculate rewards for a batch of trajectories.
        
        Args:
            contexts: List of formatted contexts with queries and values
            baseline_model: Optional model to calculate baseline probabilities
            
        Returns:
            List of reward values
        """
        if not contexts:
            return []
        
        rewards = []
        
        # Process each context
        for context in contexts:
            # Tokenize the context
            inputs = self.tokenizer(
                context, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Extract positions of values using patterns like </query> and </value>
            value_positions = self._extract_value_positions(inputs.input_ids)
            
            if not value_positions:
                logger.warning(f"No value positions found in context: {context[:100]}...")
                rewards.append(0.0)
                continue
            
            # Forward pass with our model
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask
                )
                logits = outputs.logits
            
            # Get baseline logits if baseline model is provided
            if baseline_model:
                with torch.no_grad():
                    baseline_outputs = baseline_model.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask
                    )
                    baseline_logits = baseline_outputs.logits
            
            # Calculate log probabilities for each value token
            log_probs = []
            
            for start_pos, end_pos in value_positions:
                for pos in range(start_pos, end_pos):
                    if pos >= inputs.input_ids.size(1) - 1:
                        continue
                    
                    # Get the actual next token
                    target_id = inputs.input_ids[0, pos + 1]
                    
                    # Get logits for the current position
                    current_logits = logits[0, pos, :]
                    
                    # Compute log probability of the actual next token
                    log_softmax = torch.nn.functional.log_softmax(current_logits, dim=0)
                    log_prob = log_softmax[target_id].item()
                    
                    # If baseline model exists, compute baseline log probability
                    if baseline_model:
                        baseline_log_softmax = torch.nn.functional.log_softmax(
                            baseline_logits[0, pos, :], 
                            dim=0
                        )
                        baseline_log_prob = baseline_log_softmax[target_id].item()
                        
                        # Compute normalized log probability
                        normalized_log_prob = log_prob - baseline_log_prob
                        log_probs.append(normalized_log_prob)
                    else:
                        log_probs.append(log_prob)
            
            # Average log probabilities for this context
            if log_probs:
                avg_log_prob = sum(log_probs) / len(log_probs)
                rewards.append(avg_log_prob)
            else:
                rewards.append(0.0)
        
        return rewards
    
    def _extract_value_positions(self, input_ids: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Extract positions of value tokens in a tokenized context.
        
        Args:
            input_ids: Tensor of token IDs
            
        Returns:
            List of (start, end) positions for value tokens
            
        Raises:
            ValueError: If an opening <value> tag doesn't have a corresponding </value> tag
        """
        batch_size = input_ids.size(0)
        all_batch_positions = []
        
        value_marker = " <value> "
        value_marker_ids = self.tokenizer.encode(value_marker, add_special_tokens=False)
        marker_len = len(value_marker_ids)
        value_end_marker = " </value> "
        value_end_marker_ids = self.tokenizer.encode(value_end_marker, add_special_tokens=False)
        end_marker_len = len(value_end_marker_ids)
        eot_token_ids = self.tokenizer.encode(EOT_TOKEN, add_special_tokens=False)
        eot_len = len(eot_token_ids)
        
        # Extract positions for each batch element
        for batch_idx in range(batch_size):
            value_positions = []
            
            for i in range(len(input_ids[batch_idx]) - marker_len + 1):
                if input_ids[batch_idx][i:i+marker_len].tolist() == value_marker_ids:
                    # Start position is after the value marker
                    start_pos = i + marker_len
                    
                    # Find the end of this value (</value> tag or EOT_TOKEN, whichever comes first)
                    end_pos = None
                    
                    # First try to find the end marker
                    for j in range(start_pos, len(input_ids[batch_idx]) - end_marker_len + 1):
                        if input_ids[batch_idx][j:j+end_marker_len].tolist() == value_end_marker_ids:
                            end_pos = j
                            break
                    
                    # If end marker not found, throw an error
                    if end_pos is None:
                        # Get more context for the error message
                        context_text = self.tokenizer.decode(input_ids[batch_idx])
                        raise ValueError(f"Found opening <value> tag without a corresponding </value> tag. Context: {context_text[:100]}...")
                        
                    value_positions.append((start_pos, end_pos))
            
            all_batch_positions.append(value_positions)
        
        # Assert that all batch elements have the same value positions
        if batch_size > 1:
            reference_positions = all_batch_positions[0]
            for batch_idx in range(1, batch_size):
                assert all_batch_positions[batch_idx] == reference_positions, \
                    f"Value positions are not synchronized across batch dimension. " \
                    f"Batch element 0: {reference_positions}, " \
                    f"Batch element {batch_idx}: {all_batch_positions[batch_idx]}"
        
        # Return positions from the first batch element only
        return all_batch_positions[0] if all_batch_positions else []

    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        ensure_dir(os.path.dirname(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model checkpoint saved to: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to load the checkpoint from
        """
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return
        
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map=self.device if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Ensure the tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model checkpoint loaded from: {path}")

def load_language_model(
    model_name: str = MODEL_NAME,
    device: str = None,
    load_in_8bit: bool = False,
    cache_dir: str = "model_cache",
    config: Optional['Config'] = None
) -> LanguageModel:
    """
    Load a language model as a pure function.
    
    Args:
        model_name: Name of the model to load
        device: Device to load the model on
        load_in_8bit: Whether to load in 8-bit precision
        cache_dir: Directory for caching model files
        config: Optional Config object containing model settings
        
    Returns:
        A new LanguageModel instance
    """
    # Use config parameters if provided
    if config is not None:
        model_name = config.model_name
    
    # Create and return a new LanguageModel instance
    return LanguageModel(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        load_in_8bit=load_in_8bit
    ) 