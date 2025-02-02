import pytest
import torch
import numpy as np
from src.models.information_sharing.network import (
    NodeConfig, NetworkConfig, InformationSharingNetwork
)

@pytest.fixture
def node_config():
    """Create base node configuration."""
    return NodeConfig(
        input_dim=3,  # inventory, backlog, demand
        hidden_dim=64,
        output_dim=32,
        activation='ReLU',
        dropout=0.1,
        delay=0,
        aggregation_type='attention'
    )

@pytest.fixture
def network_config(node_config):
    """Create network configuration."""
    return NetworkConfig(
        num_echelons=2,  # Two nodes in the network
        node_configs=[node_config] * 2,
        global_hidden_dim=64,
        topology_type='chain',
        noise_type='gaussian',
        noise_params={'std': 0.1}
    )

@pytest.fixture
def network(network_config):
    """Create test network."""
    return InformationSharingNetwork(network_config)

def test_basic_forward(network):
    """Test basic forward pass through network."""
    # Create sample inputs
    batch_size = 2
    states = [
        torch.randn(batch_size, 3)  # inventory, backlog, demand
        for _ in range(2)  # 2 nodes
    ]
    
    # Forward pass
    enhanced_states, global_state = network(states)
    
    # Check shapes
    assert len(enhanced_states) == 2, "Should have enhanced states for both nodes"
    assert enhanced_states[0].shape == (batch_size, 32), "Enhanced state should have correct dimension"
    assert enhanced_states[1].shape == (batch_size, 32), "Enhanced state should have correct dimension"
    assert global_state.shape == (batch_size, 64), "Global state should have correct dimension"
    
    # Check no NaN values
    for state in enhanced_states:
        assert not torch.isnan(state).any(), "Enhanced states should not contain NaN"
    assert not torch.isnan(global_state).any(), "Global state should not contain NaN"

def test_information_flow(network):
    """Test information flow between nodes."""
    # Create distinct states
    states = [
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),  # Node 1: only inventory
        torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)   # Node 2: only backlog
    ]
    
    # Forward pass
    enhanced_states, _ = network(states)
    
    # Check that information has flowed between nodes
    # Enhanced states should be different from each other
    # since they started with different inputs
    assert not torch.allclose(
        enhanced_states[0],
        enhanced_states[1],
        atol=1e-3
    ), "Enhanced states should be different due to information flow"
    
    # Check that output features are non-zero (information has propagated)
    for enhanced in enhanced_states:
        assert torch.any(enhanced.abs() > 1e-3), "Enhanced state should contain non-zero values"

def test_batch_processing(network):
    """Test processing multiple batches."""
    # Create batched inputs
    batch_size = 4
    states = [
        torch.randn(batch_size, 3)
        for _ in range(2)
    ]
    
    # Forward pass
    enhanced_states, global_state = network(states)
    
    # Check batch dimension is preserved
    assert enhanced_states[0].size(0) == batch_size
    assert enhanced_states[1].size(0) == batch_size
    assert global_state.size(0) == batch_size

def test_feature_consistency(network):
    """Test consistency of feature processing."""
    # Create identical states
    states = [
        torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    ]
    
    # Set network to eval mode to disable dropout
    network.eval()
    
    # Multiple forward passes
    enhanced_1, global_1 = network(states)
    enhanced_2, global_2 = network(states)
    
    # Check consistency
    for e1, e2 in zip(enhanced_1, enhanced_2):
        assert torch.allclose(e1, e2, atol=1e-6), "Same input should produce same output"
    assert torch.allclose(global_1, global_2, atol=1e-6), "Same input should produce same global state"
    
    # Set network back to train mode
    network.train()

def test_gradient_flow(network):
    """Test gradient flow through the network."""
    # Create sample input
    states = [
        torch.randn(2, 3, requires_grad=True)
        for _ in range(2)
    ]
    
    # Forward pass
    enhanced_states, global_state = network(states)
    
    # Compute loss and backward
    loss = sum(state.mean() for state in enhanced_states) + global_state.mean()
    loss.backward()
    
    # Check gradients exist
    for state in states:
        assert state.grad is not None, "Input should have gradients"
        assert not torch.isnan(state.grad).any(), "Gradients should not be NaN"

def test_delay_buffer(network_config, node_config):
    """Test delay buffer functionality."""
    # Create network with delay
    node_config.delay = 2
    delayed_network = InformationSharingNetwork(network_config)
    
    # Create sequence of states
    states = [
        torch.randn(1, 3)
        for _ in range(2)
    ]
    
    # Process multiple times
    outputs = []
    for _ in range(3):
        enhanced, _ = delayed_network(states)
        outputs.append([s.clone() for s in enhanced])
    
    # First outputs should be zero (from buffer initialization)
    assert torch.allclose(outputs[0][0], torch.zeros_like(outputs[0][0]))

def test_attention_mechanism(network_config, node_config):
    """Test attention-based aggregation."""
    # Create network with attention
    node_config.aggregation_type = 'attention'
    attention_network = InformationSharingNetwork(network_config)
    
    # Create states with clear patterns
    states = [
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    ]
    
    # Process with attention
    enhanced_states, _ = attention_network(states)
    
    # Check outputs are reasonable
    for state in enhanced_states:
        assert not torch.allclose(state, torch.zeros_like(state)), "Attention outputs should be non-zero"
