"""Actor-Critic Value Network Implementation.

This module implements a value network for the actor-critic architecture, designed to estimate
state values in the supply chain environment. The implementation focuses on stability and
accurate value estimation.

Key Components:
1. Value Network Architecture:
   - Multi-layer feed-forward network with ReLU activations
   - Configurable number of hidden layers for flexibility
   - Single output for state value estimation

2. Stability Features:
   - Orthogonal initialization with sqrt(2) gain for ReLU layers
   - Zero initialization for biases
   - No batch normalization to avoid shifting value estimates
   - Direct value output without activation for unbounded estimates

Usage:
    value_net = ValueNetwork(input_dim=29, hidden_dim=128, n_hidden=2)
    value_estimate = value_net(state)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List


class ValueNetwork(nn.Module):
    """Value network for state value estimation in supply chain management.
    
    This implementation uses a feed-forward neural network to estimate the expected
    return (discounted sum of rewards) from each state. The network is designed
    for stability and accurate value estimation.
    
    Attributes:
        input_dim: Input dimension (state size)
        hidden_dim: Hidden layer dimension
        n_hidden: Number of hidden layers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_hidden: int = 2
    ):
        """Initialize value network.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            n_hidden: Number of hidden layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden

        # Build network layers
        layers = []
        in_dim = input_dim
        for _ in range(n_hidden):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))

        self.value_net = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Value tensor of shape [batch_size, 1]
        """
        return self.value_net(x)
