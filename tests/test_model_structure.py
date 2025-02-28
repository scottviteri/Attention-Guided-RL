"""
Tests for the Llama model structure.
"""

import pytest
import torch
from src.model import LanguageModel

@pytest.fixture(scope="module")
def language_model():
    """Fixture to provide a language model for tests."""
    return LanguageModel(model_name="meta-llama/Llama-3.2-3B-Instruct", 
                          cache_dir="../model_cache", 
                          device="cpu")

def test_model_basic_structure(language_model):
    """Test the basic structure of the model."""
    model = language_model
    
    # Check model attributes
    assert hasattr(model, 'model')
    assert hasattr(model, 'tokenizer')
    assert hasattr(model, 'device')

def test_model_layers(language_model):
    """Test the model layers structure."""
    model = language_model
    
    # Check model.model structure
    assert hasattr(model.model, 'model')
    
    # Check model.model.model structure
    assert hasattr(model.model.model, 'layers')
    assert len(model.model.model.layers) > 0
    
    # Check first layer structure
    layer = model.model.model.layers[0]
    assert hasattr(layer, 'self_attn')
    assert hasattr(layer, 'mlp')
    
    # Check attention mechanism
    assert hasattr(layer.self_attn, 'k_proj')
    assert hasattr(layer.self_attn, 'q_proj')
    assert hasattr(layer.self_attn, 'v_proj')
    assert hasattr(layer.self_attn, 'o_proj')

def test_model_inference(language_model):
    """Test basic model inference."""
    model = language_model
    
    # Create input for a short test
    input_text = "Hello, world!"
    inputs = model.tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Run model with output_hidden_states=True
    with torch.inference_mode():
        outputs = model.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True
        )
    
    # Check outputs structure
    assert hasattr(outputs, 'hidden_states')
    assert isinstance(outputs.hidden_states, tuple)
    assert len(outputs.hidden_states) > 0
    
    # Check the shapes of hidden states
    for layer_output in outputs.hidden_states:
        assert isinstance(layer_output, torch.Tensor)
        assert layer_output.ndim == 3  # (batch_size, sequence_length, hidden_size)
        assert layer_output.shape[0] == 1  # batch_size = 1
        assert layer_output.shape[1] == inputs.input_ids.shape[1]  # sequence_length 