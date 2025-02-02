"""Actor-Critic Policy Network Implementation.

This module implements a policy network for the actor-critic architecture, specifically designed
for the supply chain ordering decisions. The implementation includes several key stability features:

Key Components:
1. Policy Network Architecture:
   - Two-layer feed-forward network with ReLU activations
   - Separate heads for mean and log standard deviation
   - Bounded action space using tanh activation
   - Conservative exploration through bounded standard deviation

2. Stability Features:
   - Orthogonal initialization with small gains (0.5 for shared layers, 0.01 for heads)
   - Bounded standard deviation (0.1 to 0.5) to prevent extreme exploration
   - Initial bias towards smaller standard deviation (-1.0 bias in log_std head)
   - Reparameterization trick for better gradient flow

3. PPO-Specific Features:
   - Action sampling with reparameterization for stable training
   - Support for deterministic action selection during evaluation
   - Proper log probability and entropy calculations

Usage:
    policy = OrderingPolicy(input_dim=29, hidden_dim=128, action_dim=3)
    
    # Training mode
    action, log_prob = policy.sample_action(state)
    
    # Evaluation mode
    action, _ = policy.sample_action(state, deterministic=True)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Optional
import numpy as np


class OrderingPolicy(nn.Module):
    """Policy network for ordering decisions in supply chain management.
    
    This implementation uses a Gaussian policy with state-dependent mean and standard deviation.
    The network is designed for stability and conservative exploration, which is crucial for
    supply chain management where extreme actions can be costly.
    
    Attributes:
        input_dim: Input dimension (state size)
        hidden_dim: Hidden layer dimension
        action_dim: Action dimension (number of echelons)
        min_std: Minimum standard deviation to ensure exploration
        max_std: Maximum standard deviation to prevent extreme actions
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        min_std: float = 0.1,
        max_std: float = 0.5  # Reduced from 1.0 to 0.5 for more conservative exploration
    ):
        """Initialize policy network.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            action_dim: Action dimension (number of echelons)
            min_std: Minimum standard deviation
            max_std: Maximum standard deviation
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std

        # Policy network layers with smaller initialization
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Initialize weights with smaller values
        for m in self.policy_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0.0)

        # Mean and std heads with smaller initialization
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.log_std_head.bias, -1.0)  # Initialize to smaller std

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        """Forward pass through policy network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Normal distribution with learned mean and std
        """
        # Get features from policy network
        features = self.policy_net(x)
        
        # Get mean and std with tanh for bounded output
        mean = torch.tanh(self.mean_head(features))
        log_std = self.log_std_head(features)
        
        # Clamp std between min and max values
        std = torch.clamp(log_std.exp(), self.min_std, self.max_std)
        
        # Create normal distribution
        return torch.distributions.Normal(mean, std)

    def sample_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy.
        
        Args:
            state: State tensor
            deterministic: If True, return mean action
            
        Returns:
            Tuple of (action, log_prob)
        """
        dist = self(state)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()  # Use reparameterization trick
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actions.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Tuple of:
                - Log probability tensor [batch_size]
                - Entropy tensor [batch_size]
        """
        dist = self(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy
