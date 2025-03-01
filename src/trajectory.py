"""
Functions and classes for trajectory representation and manipulation.
With a focus on efficient batched operations for all tensors.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any 
from dataclasses import dataclass

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
    Class for managing trajectory data.
    """
    
    def __init__(self, steps: List[Dict[str, Any]] = None, batch_size: int = 1):
        """
        Initialize a trajectory.
        
        Args:
            steps: List of steps, where each step is a dictionary containing:
                - 'query': The query string
                - 'key': The selected key string
                - 'value': The selected value string
                - 'context': The context at that step
            batch_size: Batch size for trajectory (default: 1 for backward compatibility)
        """
        self.steps = steps if steps is not None else []
        self.length = len(self.steps)
        self._batch_size = batch_size
        
        # Initialize batch data structures if using batch mode
        if batch_size > 0:
            self.queries = []
            self.key_ids = []
            self.value_tokens = []
            self.query_tokens = []
            self.query_embeddings = []
            self.key_probs = []
            self.raw_queries = []
            self.raw_values = []
    
    def add_step(self, query: str = None, key: str = None, value: str = None, context: str = None,
                query_tokens: torch.Tensor = None, value_tokens: torch.Tensor = None, 
                query_embeddings: np.ndarray = None, key_ids: List[str] = None, 
                key_probs: List[Dict[str, float]] = None, raw_queries: List[str] = None, 
                raw_values: List[str] = None):
        """
        Add a step to the trajectory. Supports both legacy and batch modes.
        
        Legacy mode (non-batch):
            query: Query string
            key: Selected key string
            value: Selected value string
            context: Context at this step
            
        Batch mode:
            query_tokens: Batch of tokenized queries [batch_size, QUERY_TOKEN_COUNT]
            value_tokens: Batch of tokenized values [batch_size, VALUE_TOKEN_COUNT]
            query_embeddings: Batch of query embeddings [batch_size, embedding_dim]
            key_ids: List of selected key IDs, one per batch element
            key_probs: List of dictionaries mapping key IDs to probabilities, one per batch element
            raw_queries: List of raw query strings, one per batch element
            raw_values: List of raw value strings, one per batch element
        """
        # Check if we're using legacy mode or batch mode
        if query is not None and key is not None and value is not None and context is not None:
            # Legacy mode
            self.steps.append({
                'query': query,
                'key': key,
                'value': value,
                'context': context
            })
            self.length += 1
        elif (query_tokens is not None and value_tokens is not None and 
              query_embeddings is not None and key_ids is not None and 
              key_probs is not None and raw_queries is not None and 
              raw_values is not None):
            # Batch mode
            self.query_tokens.append(query_tokens)
            self.value_tokens.append(value_tokens)
            self.query_embeddings.append(query_embeddings)
            self.key_ids.append(key_ids)
            self.key_probs.append(key_probs)
            self.raw_queries.append(raw_queries)
            self.raw_values.append(raw_values)
            self.length += 1
        else:
            raise ValueError("Invalid arguments for add_step. Must provide either legacy mode arguments or batch mode arguments.")
    
    def get_queries(self) -> List[str]:
        """Get all queries in the trajectory."""
        return [step['query'] for step in self.steps]
    
    def get_keys(self) -> List[str]:
        """Get all selected keys in the trajectory."""
        return [step['key'] for step in self.steps]
    
    def get_values(self) -> List[str]:
        """Get all selected values in the trajectory."""
        return [step['value'] for step in self.steps]
    
    def get_contexts(self) -> List[str]:
        """Get all contexts in the trajectory."""
        return [step['context'] for step in self.steps]
    
    def get_final_context(self) -> str:
        """Get the final context after all steps."""
        if not self.steps:
            return ""
        return self.steps[-1]['context']
    
    def to_dict(self) -> Dict[str, List[str]]:
        """Convert trajectory to dictionary format."""
        return {
            'queries': self.get_queries(),
            'keys': self.get_keys(),
            'values': self.get_values(),
            'contexts': self.get_contexts()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, List[str]]) -> 'Trajectory':
        """Create trajectory from dictionary format."""
        steps = []
        for i in range(len(data['queries'])):
            steps.append({
                'query': data['queries'][i],
                'key': data['keys'][i],
                'value': data['values'][i],
                'context': data['contexts'][i]
            })
        return cls(steps=steps)

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
        
        for b in range(self._batch_size):
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
        contexts = [""] * self._batch_size
        
        for step in range(self.length):
            for b in range(self._batch_size):
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
        query_value_pairs = [""] * self._batch_size
        
        for step in range(self.length):
            for b in range(self._batch_size):
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
        dict_lists = [[] for _ in range(self._batch_size)]
        
        # Fill in the dictionaries
        for step in range(self.length):
            for b in range(self._batch_size):
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
    
    def get_batch_size(self) -> int:
        """
        Get the batch size of this trajectory.
        
        Returns:
            Batch size (number of trajectories in this batch)
        """
        return self._batch_size 