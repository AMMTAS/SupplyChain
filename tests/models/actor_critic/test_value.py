"""Tests for value network."""

import pytest
import torch
import numpy as np
from src.models.actor_critic.value import ValueNetwork
from tests.test_results import TestResults
import time


@pytest.fixture
def value_net():
    """Create test value network."""
    return ValueNetwork(
        input_dim=128,
        hidden_dim=64,
        n_hidden=2
    )


def test_initialization(value_net):
    """Test value network initialization."""
    assert isinstance(value_net, ValueNetwork)
    assert value_net.input_dim == 128
    assert value_net.hidden_dim == 64
    assert value_net.n_hidden == 2


def test_forward_pass(value_net):
    """Test forward pass through value network."""
    batch_size = 32
    x = torch.randn(batch_size, 128)
    value = value_net(x)
    
    assert value.shape == (batch_size, 1)


def test_value_range(value_net):
    """Test value network output range."""
    batch_size = 32
    x = torch.randn(batch_size, 128)
    value = value_net(x)
    
    # Values should be finite
    assert torch.all(torch.isfinite(value))


def test_gradient_flow(value_net):
    """Test gradient flow through value network."""
    batch_size = 32
    x = torch.randn(batch_size, 128, requires_grad=True)
    value = value_net(x)
    loss = value.mean()
    loss.backward()
    
    # Check gradients
    for param in value_net.parameters():
        assert param.grad is not None
        assert torch.all(torch.isfinite(param.grad))


def test_value_network_performance():
    """Test and log value network performance."""
    # Create test data
    n_samples = 1000
    batch_size = 32
    input_dim = 29  # 24 demand + 2 fuzzy + 3 moea
    
    states = torch.randn(n_samples, input_dim)
    returns = torch.randn(n_samples, 1)  # Value targets
    
    # Initialize network
    network = ValueNetwork(
        input_dim=input_dim,
        hidden_dim=128,
        n_hidden=2
    )
    
    # Test forward pass
    values = network(states[:batch_size])
    assert values.shape == (batch_size, 1)
    assert torch.all(torch.isfinite(values))
    
    # Test gradient flow
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    value_loss = torch.nn.MSELoss()(values, returns[:batch_size])
    value_loss.backward()
    
    for param in network.parameters():
        assert param.grad is not None
        assert torch.all(torch.isfinite(param.grad))
