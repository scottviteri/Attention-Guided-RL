"""
Functions and classes for trajectory representation and manipulation.
With a focus on efficient batched operations for all tensors.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field

from src.config import (
    QUERY_TOKEN_COUNT,
    VALUE_TOKEN_COUNT,
    EOT_TOKEN,
    USER_START,
    ASSISTANT_START,
    SYSTEM_START,
    REWARD_SYSTEM_PROMPT
)


@dataclass(frozen=True)
class QueryValuePair:
    """
    A frozen dataclass representing a query-value pair in a trajectory.
    
    Attributes:
        query_tokens: Tokenized query content as a PyTorch tensor (QUERY_TOKEN_COUNT)
        value_tokens: Tokenized value content as a PyTorch tensor (VALUE_TOKEN_COUNT)
        query_embedding: Embedding vector for the query
        key_id: Identifier for the key that was selected
        key_probs: Probabilities for selecting each key
        raw_query: Raw string representation of the query
        raw_value: Raw string representation of the value
    """
    query_tokens: torch.Tensor
    value_tokens: torch.Tensor
    query_embedding: np.ndarray
    key_id: str
    key_probs: Dict[str, float]
    raw_query: str
    raw_value: str


class Trajectory:
    """
    Class for managing trajectory data with native batching support.
    All tensor data has a batch dimension by default.
    """
    
    def __init__(
        self, 
        batch_size: int = 1,
        device: str = "cpu"
    ):
        """
        Initialize a trajectory with batch support.
        
        Args:
            batch_size: Number of parallel trajectories
            device: Device for tensors
        """
        self.batch_size = batch_size
        self.device = device
        
        # Initialize empty tensors with batch dimension
        self.query_tokens = []  # List of tensors, each with shape [batch_size, QUERY_TOKEN_COUNT]
        self.value_tokens = []  # List of tensors, each with shape [batch_size, VALUE_TOKEN_COUNT]
        self.query_embeddings = []  # List of arrays, each with shape [batch_size, embedding_dim]
        self.key_ids = []  # List of lists, each with length batch_size
        self.key_probs = []  # List of dictionaries of probabilities
        self.raw_queries = []  # List of lists of strings, each with length batch_size
        self.raw_values = []  # List of lists of strings, each with length batch_size
        
        # Length of the trajectory (number of steps)
        self.length = 0
    
    def add_step(
        self,
        query_tokens: torch.Tensor,
        value_tokens: torch.Tensor,
        query_embeddings: np.ndarray,
        key_ids: List[str],
        key_probs: List[Dict[str, float]],
        raw_queries: List[str],
        raw_values: List[str]
    ):
        """
        Add a step to the trajectory for all batch elements.
        
        Args:
            query_tokens: Tensor of shape [batch_size, QUERY_TOKEN_COUNT]
            value_tokens: Tensor of shape [batch_size, VALUE_TOKEN_COUNT]
            query_embeddings: Array of shape [batch_size, embedding_dim]
            key_ids: List of key IDs of length batch_size
            key_probs: List of dictionaries mapping key IDs to probabilities
            raw_queries: List of query strings of length batch_size
            raw_values: List of value strings of length batch_size
        """
        # Validate batch dimensions match
        if query_tokens.size(0) != self.batch_size:
            raise ValueError(f"Expected batch size {self.batch_size}, got {query_tokens.size(0)}")
        
        # Append data
        self.query_tokens.append(query_tokens)
        self.value_tokens.append(value_tokens)
        self.query_embeddings.append(query_embeddings)
        self.key_ids.append(key_ids)
        self.key_probs.append(key_probs)
        self.raw_queries.append(raw_queries)
        self.raw_values.append(raw_values)
        
        # Increment length
        self.length += 1
    
    def get_token_tensor(self) -> torch.Tensor:
        """
        Convert the trajectory to a tensor of tokens.
        
        Returns:
            PyTorch tensor of shape [batch_size, num_steps * (QUERY_TOKEN_COUNT + VALUE_TOKEN_COUNT)]
        """
        if self.length == 0:
            return torch.tensor([], dtype=torch.long)
        
        # Concatenate all tokens for each batch element
        batch_tokens = []
        
        for b in range(self.batch_size):
            # Gather tokens for this batch element
            batch_element_tokens = []
            for step in range(self.length):
                batch_element_tokens.append(self.query_tokens[step][b])
                batch_element_tokens.append(self.value_tokens[step][b])
            
            # Concatenate into a single tensor for this batch element
            batch_tokens.append(torch.cat(batch_element_tokens).unsqueeze(0))
        
        # Concatenate batch elements
        return torch.cat(batch_tokens, dim=0)
    
    def get_query_contexts(self) -> List[str]:
        """
        Format the trajectory for query generation.
        
        Returns:
            List of formatted strings for generating the next query, one per batch element
        """
        contexts = [""] * self.batch_size
        
        for step in range(self.length):
            for b in range(self.batch_size):
                contexts[b] += f" <query> {self.raw_queries[step][b]} </query> {EOT_TOKEN} <value> {self.raw_values[step][b]} </value> {EOT_TOKEN} "
        
        return contexts
    
    def get_reward_contexts(self, system_prompt: str = REWARD_SYSTEM_PROMPT) -> List[str]:
        """
        Format the trajectory for reward calculation.
        
        Args:
            system_prompt: System prompt for reward calculation
            
        Returns:
            List of formatted strings for reward calculation, one per batch element
        """
        query_value_pairs = [""] * self.batch_size
        
        for step in range(self.length):
            for b in range(self.batch_size):
                query_value_pairs[b] += f" <query> {self.raw_queries[step][b]} </query> {EOT_TOKEN} <value> {self.raw_values[step][b]} </value> {EOT_TOKEN} "
        
        return [
            f"{SYSTEM_START} {system_prompt} {EOT_TOKEN} {USER_START} {pairs} {EOT_TOKEN}"
            for pairs in query_value_pairs
        ]
    
    def get_value_positions(self, tokenizer, tokenized_contexts: torch.Tensor) -> List[List[Tuple[int, int]]]:
        """
        Extract positions of value tokens in tokenized contexts.
        
        Args:
            tokenizer: Tokenizer to use for encoding markers
            tokenized_contexts: Tensor of tokenized contexts [batch_size, seq_length]
            
        Returns:
            List of lists of (start, end) positions for value tokens, one list per batch element
        """
        batch_value_positions = []
        value_marker = " <value> "
        value_marker_ids = tokenizer.encode(value_marker, add_special_tokens=False)
        marker_len = len(value_marker_ids)
        value_end_marker = " </value> "
        value_end_marker_ids = tokenizer.encode(value_end_marker, add_special_tokens=False)
        end_marker_len = len(value_end_marker_ids)
        eot_token_ids = tokenizer.encode(EOT_TOKEN, add_special_tokens=False)
        eot_len = len(eot_token_ids)
        
        for b in range(tokenized_contexts.size(0)):
            value_positions = []
            context = tokenized_contexts[b]
            
            for i in range(context.size(0) - marker_len + 1):
                if context[i:i+marker_len].tolist() == value_marker_ids:
                    # Start position is after the value marker
                    start_pos = i + marker_len
                    
                    # Find the end of this value (</value> tag or EOT_TOKEN, whichever comes first)
                    end_pos = None
                    
                    for j in range(start_pos, context.size(0) - end_marker_len + 1):
                        if context[j:j+end_marker_len].tolist() == value_end_marker_ids:
                            end_pos = j
                            break
                    
                    # If </value> tag not found, look for EOT_TOKEN
                    if end_pos is None:
                        for j in range(start_pos, context.size(0) - eot_len + 1):
                            if context[j:j+eot_len].tolist() == eot_token_ids:
                                end_pos = j
                                break
                    
                    # If neither found, use end of sequence
                    if end_pos is None:
                        end_pos = context.size(0)
                        
                    value_positions.append((start_pos, end_pos))
            
            batch_value_positions.append(value_positions)
        
        return batch_value_positions
    
    def to_dict_list(self) -> List[List[Dict[str, Any]]]:
        """
        Convert the trajectory to lists of dictionaries for backward compatibility.
        
        Returns:
            List of lists of dictionaries, where each inner list represents the trajectory
            for one batch element
        """
        # Initialize a list for each batch element
        dict_lists = [[] for _ in range(self.batch_size)]
        
        # Fill in the dictionaries
        for step in range(self.length):
            for b in range(self.batch_size):
                dict_lists[b].append({
                    "query": self.raw_queries[step][b],
                    "query_embedding": self.query_embeddings[step][b],
                    "key_id": self.key_ids[step][b],
                    "value": self.raw_values[step][b],
                    "probs": self.key_probs[step][b]
                })
        
        return dict_lists
    
    @classmethod
    def from_dict_lists(cls, dict_lists: List[List[Dict[str, Any]]], tokenizer) -> 'Trajectory':
        """
        Create a batched Trajectory from lists of dictionaries (backward compatibility).
        
        Args:
            dict_lists: List of lists of dictionaries, one list per batch element
            tokenizer: Tokenizer to convert strings to tokens
            
        Returns:
            New Trajectory instance with batch support
        """
        if not dict_lists:
            return cls(batch_size=0)
        
        # Create a new trajectory with the appropriate batch size
        batch_size = len(dict_lists)
        traj = cls(batch_size=batch_size)
        
        # Determine the minimum length across all batch elements
        min_length = min(len(dict_list) for dict_list in dict_lists)
        
        # Add each step
        for step in range(min_length):
            # Prepare data for this step
            raw_queries = []
            raw_values = []
            query_embeddings = []
            key_ids = []
            key_probs = []
            query_token_list = []
            value_token_list = []
            
            for b in range(batch_size):
                step_data = dict_lists[b][step]
                
                # Extract raw data
                raw_queries.append(step_data["query"])
                raw_values.append(step_data["value"])
                query_embeddings.append(step_data["query_embedding"])
                key_ids.append(step_data["key_id"])
                key_probs.append(step_data["probs"])
                
                # Tokenize query and value
                query_tokens = tokenizer(
                    step_data["query"],
                    padding="max_length",
                    max_length=QUERY_TOKEN_COUNT,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.squeeze(0)
                
                value_tokens = tokenizer(
                    step_data["value"],
                    padding="max_length",
                    max_length=VALUE_TOKEN_COUNT,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.squeeze(0)
                
                query_token_list.append(query_tokens.unsqueeze(0))
                value_token_list.append(value_tokens.unsqueeze(0))
            
            # Stack tensors
            query_tokens_batch = torch.cat(query_token_list, dim=0)
            value_tokens_batch = torch.cat(value_token_list, dim=0)
            query_embeddings_array = np.stack(query_embeddings)
            
            # Add step to trajectory
            traj.add_step(
                query_tokens=query_tokens_batch,
                value_tokens=value_tokens_batch,
                query_embeddings=query_embeddings_array,
                key_ids=key_ids,
                key_probs=key_probs,
                raw_queries=raw_queries,
                raw_values=raw_values
            )
        
        return traj
    
    def __len__(self):
        """Return the number of steps in the trajectory."""
        return self.length
    
    def get_batch_size(self):
        """Return the batch size of the trajectory."""
        return self.batch_size 