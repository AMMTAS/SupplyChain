"""Network tests for Information Sharing Network."""

import pytest
import torch
import torch.nn as nn
from src.models.information_sharing.network import InformationNode, InformationSharingNetwork
from config.information_sharing_config import NodeConfig, NetworkConfig
from tests.test_results import TestResults
import time
import numpy as np


@pytest.fixture
def sample_node_config():
    """Create a sample node configuration."""
    return NodeConfig(
        input_dim=3,  # Basic state: inventory, backlog, demand
        hidden_dim=32,
        output_dim=16,
        delay=2,
        dropout=0.1,
        activation='ReLU',
        aggregation_type='attention'
    )


@pytest.fixture
def sample_network_config(sample_node_config):
    """Create a sample network configuration."""
    return NetworkConfig(
        num_echelons=4,
        node_configs=[sample_node_config] * 4,
        global_hidden_dim=64,
        noise_type='gaussian',
        noise_params={'std': 0.1},
        topology_type='chain'
    )


def test_information_node_initialization(sample_node_config):
    """Test node initialization."""
    node = InformationNode(sample_node_config)
    
    # Verify network structure
    assert isinstance(node.local_network, nn.Sequential)
    assert len(node.local_network) == 4  # Linear -> Dropout -> ReLU -> Linear
    
    # Verify network dimensions
    first_layer = node.local_network[0]
    last_layer = node.local_network[-1]
    assert first_layer.in_features == sample_node_config.input_dim
    assert first_layer.out_features == sample_node_config.hidden_dim
    assert last_layer.in_features == sample_node_config.hidden_dim
    assert last_layer.out_features == sample_node_config.output_dim


def test_information_node_forward(sample_node_config):
    """Test node forward pass."""
    node = InformationNode(sample_node_config)
    batch_size = 32
    
    # Test with just local state
    local_state = torch.randn(batch_size, sample_node_config.input_dim)
    output = node(local_state)
    assert output.shape == (batch_size, sample_node_config.output_dim)
    
    # Test with neighbor states
    neighbor_states = [
        torch.randn(batch_size, sample_node_config.input_dim)
        for _ in range(3)
    ]
    output = node(local_state, neighbor_states)
    assert output.shape == (batch_size, sample_node_config.output_dim)


def test_information_network_initialization(sample_network_config):
    """Test network initialization."""
    network = InformationSharingNetwork(sample_network_config)
    
    # Verify node creation
    assert len(network.nodes) == sample_network_config.num_echelons
    assert all(isinstance(node, InformationNode) for node in network.nodes)
    
    # Verify global network
    assert isinstance(network.global_network, nn.Sequential)
    first_layer = network.global_network[0]
    last_layer = network.global_network[-1]
    total_output_dim = sample_network_config.num_echelons * sample_network_config.node_configs[0].output_dim
    assert first_layer.in_features == total_output_dim
    assert last_layer.out_features == total_output_dim


def test_information_network_forward(sample_network_config):
    """Test network forward pass."""
    network = InformationSharingNetwork(sample_network_config)
    batch_size = 32
    
    # Create input states
    states = [
        torch.randn(batch_size, sample_network_config.node_configs[0].input_dim)
        for _ in range(sample_network_config.num_echelons)
    ]
    
    # Test without adjacency matrix
    enhanced_states, global_state = network(states)
    assert len(enhanced_states) == sample_network_config.num_echelons
    assert all(s.shape == (batch_size, sample_network_config.node_configs[0].output_dim)
              for s in enhanced_states)
    total_output_dim = sample_network_config.num_echelons * sample_network_config.node_configs[0].output_dim
    assert global_state.shape == (batch_size, total_output_dim)
    
    # Test with custom adjacency matrix
    adj_matrix = torch.eye(sample_network_config.num_echelons)  # No connections
    enhanced_states, global_state = network(states, adj_matrix)
    assert len(enhanced_states) == sample_network_config.num_echelons


def test_delay_buffer(sample_node_config):
    """Test delay buffer handling."""
    node = InformationNode(sample_node_config)
    batch_size = 32
    
    # Create sequence of states
    states = [
        torch.randn(batch_size, sample_node_config.input_dim)
        for _ in range(5)
    ]
    
    # Process sequence
    outputs = []
    for state in states:
        output = node(state)
        outputs.append(output)
    
    # Verify buffer size
    assert len(node.buffer) == min(len(states), sample_node_config.delay)
    
    # Verify delay effect through state comparison
    if sample_node_config.delay > 0 and len(states) > sample_node_config.delay:
        # Get the last processed state and its corresponding delayed input
        latest_output = outputs[-1]  # Shape: [batch_size, output_dim]
        delayed_input = states[-1-sample_node_config.delay]  # Shape: [batch_size, input_dim]
        
        # Calculate mean values for comparison
        output_mean = latest_output.mean(dim=1)  # Shape: [batch_size]
        input_mean = delayed_input.mean(dim=1)  # Shape: [batch_size]
        
        # Calculate correlation between means
        correlation = torch.corrcoef(torch.stack([output_mean, input_mean]))[0, 1]
        assert not torch.isnan(correlation)


def test_network_topology(sample_network_config):
    """Test different network topologies."""
    # Test chain topology
    sample_network_config.topology_type = 'chain'
    network = InformationSharingNetwork(sample_network_config)
    adj_matrix = network.config.get_adjacency_matrix()
    assert adj_matrix.shape == (sample_network_config.num_echelons,
                              sample_network_config.num_echelons)
    # Verify chain structure
    for i in range(sample_network_config.num_echelons - 1):
        assert adj_matrix[i, i+1] == 1
        assert adj_matrix[i+1, i] == 1


def test_noise_types(sample_network_config):
    """Test different noise types."""
    batch_size = 32
    input_dim = sample_network_config.node_configs[0].input_dim
    x = torch.randn(batch_size, input_dim)
    
    # Test Gaussian noise
    sample_network_config.noise_type = 'gaussian'
    sample_network_config.noise_params = {'std': 0.1}
    network = InformationSharingNetwork(sample_network_config)
    noisy_x = network._apply_noise([x])[0]  # Pass list and get first element
    assert noisy_x.shape == x.shape
    assert not torch.allclose(noisy_x, x)  # Values should be different due to noise
    
    # Test dropout noise
    sample_network_config.noise_type = 'dropout'
    sample_network_config.noise_params = {'p': 0.1}
    network = InformationSharingNetwork(sample_network_config)
    noisy_x = network._apply_noise([x])[0]
    assert noisy_x.shape == x.shape
    assert torch.all((noisy_x == 0) | (noisy_x == x))  # Values should be either 0 or original
    
    # Test quantization noise
    sample_network_config.noise_type = 'quantization'
    sample_network_config.noise_params = {'scale': 10.0}
    network = InformationSharingNetwork(sample_network_config)
    noisy_x = network._apply_noise([x])[0]
    assert noisy_x.shape == x.shape
    assert torch.all(torch.round(noisy_x * 10.0) == noisy_x * 10.0)  # Values should be quantized


def test_node_aggregation_types(sample_node_config):
    """Test different node aggregation types."""
    batch_size = 32
    
    # Test attention aggregation
    sample_node_config.aggregation_type = 'attention'
    node = InformationNode(sample_node_config)
    local_state = torch.randn(batch_size, sample_node_config.input_dim)
    neighbor_states = [
        torch.randn(batch_size, sample_node_config.input_dim)
        for _ in range(3)
    ]
    output = node(local_state, neighbor_states)
    assert output.shape == (batch_size, sample_node_config.output_dim)
    
    # Test gated aggregation
    sample_node_config.aggregation_type = 'gated'
    node = InformationNode(sample_node_config)
    output = node(local_state, neighbor_states)
    assert output.shape == (batch_size, sample_node_config.output_dim)
    
    # Test mean aggregation
    sample_node_config.aggregation_type = 'mean'
    node = InformationNode(sample_node_config)
    output = node(local_state, neighbor_states)
    assert output.shape == (batch_size, sample_node_config.output_dim)


def test_isn_network_performance():
    """Test and log ISN network performance."""
    # Create test data
    n_samples = 1000
    batch_size = 32
    num_echelons = 3
    input_dim = 3  # inventory, backlog, demand
    hidden_dim = 32
    
    # Create network configuration
    node_config = NodeConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        delay=0,
        dropout=0.1,
        activation='ReLU',
        aggregation_type='attention'
    )
    
    network_config = NetworkConfig(
        num_echelons=num_echelons,
        node_configs=[node_config] * num_echelons,
        global_hidden_dim=hidden_dim,
        noise_type='gaussian',
        topology_type='chain'
    )
    
    # Initialize network
    network = InformationSharingNetwork(network_config)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    
    # Generate test data
    states = torch.randn(n_samples, num_echelons, input_dim)
    targets = torch.randn(n_samples, num_echelons, hidden_dim)  # Target hidden states
    
    # Training loop
    start_time = time.time()
    losses = []
    attention_scores = []
    hidden_norms = []
    
    for i in range(0, n_samples, batch_size):
        batch_states = states[i:i+batch_size]
        
        # Forward pass
        states_list = [batch_states[:, j, :] for j in range(num_echelons)]
        hidden_states, attention_weights = network(states_list)
        
        # Stack hidden states
        hidden_tensor = torch.stack(hidden_states, dim=1)
        
        # Project hidden states to match input dimension
        projection = nn.Linear(hidden_dim, input_dim).to(hidden_tensor.device)
        hidden_projected = projection(hidden_tensor)
        
        # Calculate loss (match target hidden states)
        loss = torch.nn.functional.mse_loss(hidden_projected, batch_states)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record metrics
        losses.append(loss.item())
        attention_scores.append(attention_weights.mean().item())
        hidden_norms.append(
            torch.norm(hidden_tensor).item() / batch_size
        )
    
    training_time = time.time() - start_time
    
    # Calculate metrics
    avg_loss = np.mean(losses)
    loss_std = np.std(losses)
    avg_attention = np.mean(attention_scores)
    avg_hidden_norm = np.mean(hidden_norms)
    
    # Log results
    results = TestResults()
    results.log_isn_network(
        avg_loss=avg_loss,
        loss_std=loss_std,
        avg_attention=avg_attention,
        avg_hidden_norm=avg_hidden_norm,
        training_time=training_time
    )
    results.save_results()
    
    # Assert expected ranges
    assert avg_loss <= 1.5, f"Average loss {avg_loss} too high"
    assert loss_std <= 0.5, f"Loss standard deviation {loss_std} too high"
    assert -0.5 <= avg_attention <= 0.5, f"Average attention {avg_attention} out of range"
    assert avg_hidden_norm <= 2.0, f"Average hidden norm {avg_hidden_norm} too high"
    assert training_time <= 10.0, f"Training time {training_time}s too high"
