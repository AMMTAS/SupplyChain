"""Configuration for Transformer-based Demand Predictor.

This module defines the configuration for the transformer architecture
that predicts demand across multiple horizons with uncertainty estimates.
"""

from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class TransformerConfig:
    """Configuration for the Transformer architecture."""
    # Input/Output dimensions
    input_dim: int  # Enhanced state dimension from ISN
    output_dim: int  # Demand prediction dimension
    forecast_horizon: int  # Number of future steps to predict
    history_length: int  # Length of historical sequence to consider
    
    # Architecture
    d_model: int = 256  # Transformer embedding dimension
    nhead: int = 8  # Number of attention heads
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    
    # Positional Encoding
    max_seq_length: int = 1000
    pos_encoding_type: str = 'sinusoidal'  # or 'learned'
    
    # Uncertainty Estimation
    uncertainty_type: str = 'probabilistic'  # or 'ensemble' or 'dropout'
    num_samples: int = 100  # Number of samples for uncertainty estimation
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 10  # Early stopping patience
    warmup_steps: int = 4000
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    checkpoint_dir: Optional[str] = None
    tensorboard_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate and adjust configurations."""
        # Validate attention heads
        assert self.d_model % self.nhead == 0, \
            "d_model must be divisible by nhead"
        
        # Validate sequence lengths
        assert self.history_length > 0, \
            "history_length must be positive"
        assert self.forecast_horizon > 0, \
            "forecast_horizon must be positive"
        
        # Validate uncertainty parameters
        if self.uncertainty_type == 'ensemble':
            assert self.num_samples > 1, \
                "num_samples must be > 1 for ensemble uncertainty"
