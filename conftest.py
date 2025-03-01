import os
import pytest
import torch

# Set CUDA device if available
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# Configuration for pytest
def pytest_configure(config):
    """Configure pytest for our environment."""
    # Set the environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    # Log info about CUDA availability
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"\nRunning with {device_count} CUDA device(s) available")
        for i in range(device_count):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("\nRunning without CUDA - tests will be slower")

@pytest.fixture(scope="session")
def language_model_cpu():
    """
    Fixture that provides a CPU-based language model for all tests.
    This is loaded once per test session for efficiency.
    """
    from src.model import LanguageModel
    model = LanguageModel("meta-llama/Llama-3.2-3B-Instruct", cache_dir="../model_cache", device="cpu")
    return model

@pytest.fixture(scope="session")
def language_model_cuda():
    """
    Fixture that provides a CUDA-based language model for all tests.
    This is loaded once per test session for efficiency.
    
    Tests that don't require CUDA should use language_model_cpu instead.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    from src.model import LanguageModel
    model = LanguageModel("meta-llama/Llama-3.2-3B-Instruct", cache_dir="../model_cache", device="cuda")
    return model 