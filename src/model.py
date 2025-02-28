"""
Functions for loading language models and generating queries.
"""

import os
import torch
import logging
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
    
    def batch_generate_queries(
        self,
        contexts: List[str],
        instructions: str = QUERY_INSTRUCTIONS,
        fixed_token_count: int = QUERY_TOKEN_COUNT,
        temperature: float = 0.7,
        batch_size: int = 4,
        article_title: str = None
    ) -> List[str]:
        """
        Generate queries for multiple contexts in batches.
        
        Args:
            contexts: List of contexts
            instructions: Instructions for query generation
            fixed_token_count: Fixed number of tokens to generate
            temperature: Temperature for generation
            batch_size: Batch size for generation
            article_title: Title of the Wikipedia article (optional)
            
        Returns:
            List of generated queries
        """
        # Format prompts with INSTRUCT markers
        prompts = [format_query_prompt(ctx, instructions, article_title) for ctx in contexts]
        
        # Generate in batches
        all_queries = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Tokenize batch
            batch_inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                batch_outputs = self.model.generate(
                    batch_inputs.input_ids,
                    attention_mask=batch_inputs.attention_mask,
                    min_new_tokens=fixed_token_count,
                    max_new_tokens=fixed_token_count,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=temperature,
                    do_sample=True
                )
            
            # Extract and decode responses
            for j, output in enumerate(batch_outputs):
                input_length = batch_inputs.input_ids[j].shape[0]
                query_tokens = output[input_length:]
                query = self.tokenizer.decode(query_tokens, skip_special_tokens=False)
                all_queries.append(query)
        
        return all_queries
    
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
    
    def calculate_trajectory_reward(
        self,
        trajectory: List[Dict[str, Any]],
        system_prompt: str,
        baseline_model: Optional["LanguageModel"] = None
    ) -> float:
        """
        Calculate reward for a trajectory.
        
        Args:
            trajectory: List of (query, key, value) tuples
            system_prompt: System prompt for reward calculation
            baseline_model: Optional baseline model for normalization
            
        Returns:
            Average log probability reward
        """
        # Build a string with all query-value pairs
        query_value_pairs = ""
        for step in trajectory:
            query = step["query"]
            value = step["value"]
            query_value_pairs += f"Query: {query} {EOT_TOKEN} Value: {value} {EOT_TOKEN} "
        
        # Format the context
        context = format_reward_context(system_prompt, query_value_pairs)
        
        # Tokenize
        input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
        
        # Extract value positions
        value_positions = []
        value_marker = "Value:"
        value_marker_ids = self.tokenizer.encode(value_marker, add_special_tokens=False)
        marker_len = len(value_marker_ids)
        
        for i in range(input_ids.shape[1] - marker_len + 1):
            if input_ids[0, i:i+marker_len].tolist() == value_marker_ids:
                # Find the end of this value (next EOT_TOKEN or end of sequence)
                eot_token_ids = self.tokenizer.encode(EOT_TOKEN, add_special_tokens=False)
                eot_len = len(eot_token_ids)
                
                end_pos = None
                for j in range(i + marker_len, input_ids.shape[1] - eot_len + 1):
                    if input_ids[0, j:j+eot_len].tolist() == eot_token_ids:
                        end_pos = j
                        break
                
                if end_pos is None:
                    end_pos = input_ids.shape[1]
                    
                value_positions.append((i + marker_len, end_pos))
        
        # Compute log probabilities
        log_probs = self.compute_token_probabilities(input_ids, value_positions)
        
        # Normalize with baseline if provided
        if baseline_model is not None:
            baseline_log_probs = baseline_model.compute_token_probabilities(input_ids, value_positions)
            normalized_log_probs = log_probs - baseline_log_probs
            return normalized_log_probs.mean().item()
        
        return log_probs.mean().item()
    
    def calculate_batch_trajectory_rewards(
        self,
        trajectories: List[List[Dict[str, Any]]],
        system_prompt: str,
        baseline_model: Optional["LanguageModel"] = None
    ) -> List[float]:
        """
        Calculate rewards for multiple trajectories in a batched manner.
        
        Args:
            trajectories: List of trajectories, where each trajectory is a list of steps
            system_prompt: System prompt for reward calculation
            baseline_model: Optional baseline model for normalization
            
        Returns:
            List of average log probability rewards for each trajectory
        """
        rewards = []
        
        # Process in batches (can be optimized further with true batching if needed)
        for trajectory in trajectories:
            reward = self.calculate_trajectory_reward(
                trajectory=trajectory,
                system_prompt=system_prompt,
                baseline_model=baseline_model
            )
            rewards.append(reward)
            
        return rewards
    
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
    load_in_8bit: bool = False
) -> LanguageModel:
    """
    Load a language model.
    
    Args:
        model_name: Name of the model to load
        device: Device to load the model on
        load_in_8bit: Whether to load in 8-bit precision
        
    Returns:
        Loaded language model
    """
    return LanguageModel(
        model_name=model_name,
        device=device,
        load_in_8bit=load_in_8bit
    )

def create_baseline_model(main_model: LanguageModel) -> LanguageModel:
    """
    Create a baseline model for reward normalization.
    
    This could either:
    1. Use a separately initialized model
    2. Clone the main model and reset specific parameters
    3. Return None to skip normalization
    
    Args:
        main_model: The main language model
        
    Returns:
        Baseline language model or None
    """
    # Option 1: Create a new model instance
    baseline = LanguageModel(
        model_name=main_model.model_name,
        device=main_model.device
    )
    
    # Option 2: Clone and reset (more sophisticated approach)
    # This would involve more complex model parameter manipulation
    
    return baseline 