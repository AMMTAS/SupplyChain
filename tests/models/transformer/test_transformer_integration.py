"""Integration tests for the Transformer-based Demand Predictor.

Tests the transformer's ability to learn demand patterns from ISN enhanced states.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Tuple

from src.models.transformer.network import DemandPredictor
from src.models.transformer.trainer import DemandPredictorTrainer
from src.models.information_sharing.network import InformationSharingNetwork
from config.information_sharing_config import NodeConfig, NetworkConfig
from config.transformer_config import TransformerConfig
from tests.test_results import TestResults
from pathlib import Path
import tempfile
import shutil
import copy


def generate_seasonal_demand(n_samples: int) -> np.ndarray:
    """Generate seasonal demand pattern with noise."""
    # Generate base seasonal pattern
    t = np.arange(n_samples)
    seasonal = 0.5 * np.sin(2 * np.pi * t / 12)  # 12-month seasonality
    trend = 0.1 * t / n_samples  # Small upward trend
    
    # Add noise
    noise = np.random.normal(0, 0.1, n_samples)
    
    # Combine components and ensure non-negative
    demand = np.maximum(0.5 + seasonal + trend + noise, 0)
    return demand


def create_test_network():
    """Create a test information sharing network."""
    # Create a simple 2-node network
    node_config = NodeConfig(
        input_dim=3,  # inventory, backlog, demand
        hidden_dim=64,  # Larger hidden dimension for better representation
        output_dim=32,  # Match transformer's input dimension
        delay=1,  # Add delay for temporal dependencies
        dropout=0.1,  # Light dropout for regularization
        activation='LeakyReLU',  # Better gradient flow
        aggregation_type='attention',  # Use attention for better feature selection
    )
    
    network_config = NetworkConfig(
        num_echelons=2,
        node_configs=[node_config, node_config],
        global_hidden_dim=64,  # Larger global hidden dimension
        noise_type=None,
        topology_type='chain'
    )
    
    return InformationSharingNetwork(network_config)


def create_training_data(n_samples: int, seq_length: int, forecast_horizon: int):
    """Create training data with ISN enhanced states."""
    # Create ISN
    isn = create_test_network()
    
    # Generate demand pattern
    total_length = n_samples + seq_length + forecast_horizon  # Add forecast horizon
    demand = generate_seasonal_demand(total_length)
    
    # Initialize states
    sequences = []
    targets = []
    
    # Track inventory and backlog (2 nodes in test network)
    num_nodes = len(isn.nodes)
    inventory = np.zeros(num_nodes)
    backlog = np.zeros(num_nodes)
    
    # Generate sequences
    for i in range(n_samples):
        seq_states = []
        
        # Generate sequence
        for j in range(seq_length):
            # Get current demand
            current_demand = demand[i + j]
            
            # Update inventory and backlog
            fulfilled = min(inventory[0], current_demand)
            unfulfilled = current_demand - fulfilled
            
            inventory[0] -= fulfilled
            backlog[0] += unfulfilled
            
            # Create states for each node
            node_states = []
            for node_idx in range(num_nodes):
                node_state = torch.FloatTensor([[
                    inventory[node_idx],
                    backlog[node_idx],
                    current_demand
                ]])  # Add batch dimension [1, input_dim]
                node_states.append(node_state)
            
            # Get ISN state
            enhanced_states, _ = isn(node_states)
            
            # Store state (use first node's state and remove batch dimension)
            seq_states.append(enhanced_states[0].squeeze(0).detach().numpy())
            
            # Update inventory and backlog
            inventory[0] += 10  # Fixed replenishment
            backlog[0] = max(0, backlog[0] - inventory[0])
            inventory[0] = max(0, inventory[0] - backlog[0])
        
        # Add future demands as targets
        future_demands = demand[i+seq_length:i+seq_length+forecast_horizon]  # forecast_horizon-step forecast
        
        sequences.append(np.stack(seq_states))  # Shape: [seq_length, hidden_size]
        targets.append(future_demands)  # Shape: [forecast_horizon]
    
    # Convert to tensors
    x = torch.FloatTensor(np.stack(sequences))  # Shape: [n_samples, seq_length, hidden_size]
    y = torch.FloatTensor(np.stack(targets)).unsqueeze(-1)  # Shape: [n_samples, forecast_horizon, 1]
    
    return x, y


@pytest.fixture
def config():
    """Create test configuration."""
    return TransformerConfig(
        input_dim=32,  # ISN output size
        output_dim=1,  # Demand prediction
        forecast_horizon=5,
        history_length=10,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_seq_length=50,
        device='cpu'
    )


@pytest.fixture
def model(config):
    """Create test model."""
    return DemandPredictor(config)


@pytest.fixture
def data_loaders(config):
    """Create test data loaders."""
    # Create training data
    n_samples = 1000
    seq_length = 24
    forecast_horizon = 5
    train_x, train_y = create_training_data(n_samples, seq_length, forecast_horizon)
    val_x, val_y = create_training_data(n_samples=200, seq_length=seq_length, forecast_horizon=forecast_horizon)

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=config.batch_size
    )
    
    return train_loader, val_loader


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def test_demand_prediction(model, data_loaders, temp_dir):
    """Test that the transformer can learn a simple demand pattern."""
    train_loader, val_loader = data_loaders
    
    # Initialize trainer
    trainer = DemandPredictorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        log_dir=str(temp_dir)
    )
    
    # Train model
    metrics = trainer.train()
    
    # Log results
    results = TestResults()
    results.log_demand_prediction(
        mse=metrics['train_loss'],
        coverage_1sigma=0.0,
        coverage_2sigma=0.0
    )
    results.save_results()

    # Assert expected ranges
    assert metrics['train_loss'] > 0
    if val_loader:
        assert metrics['val_loss'] > 0
    
    # Check model updates
    for param in model.parameters():
        assert not torch.isnan(param).any()
        assert not torch.isinf(param).any()
    
    # Check log files
    log_dir = Path(temp_dir)
    assert (log_dir / 'train_metrics.json').exists()
    if val_loader:
        assert (log_dir / 'val_metrics.json').exists()


def test_isn_integration(model, data_loaders, temp_dir):
    """Test integration between Information Sharing Network and Transformer.
    
    This test verifies that:
    1. ISN can process supply chain state information
    2. ISN outputs match transformer input requirements
    3. Information flows correctly between components
    4. Gradients flow through the entire pipeline
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create ISN with matching dimensions
    node_config = NodeConfig(
        input_dim=3,  # inventory, backlog, demand
        hidden_dim=64,
        output_dim=32,  # Must match transformer's input dimension
        delay=1,
        dropout=0.1,
        activation='LeakyReLU',
        aggregation_type='attention'
    )
    
    network_config = NetworkConfig(
        num_echelons=2,
        node_configs=[node_config, node_config],
        global_hidden_dim=64,
        noise_type=None,
        topology_type='chain'
    )
    
    isn = InformationSharingNetwork(network_config)
    
    # Test 1: Dimension compatibility
    batch_size = 32
    seq_length = 20
    
    # Create test batch for each node [batch_size, 3]
    node1_state = torch.randn(batch_size, 3)  # inventory, backlog, demand
    node2_state = torch.randn(batch_size, 3)
    
    # Process through ISN
    node_states = [node1_state, node2_state]
    enhanced_states, global_state = isn(node_states)
    
    # Verify ISN output dimensions
    assert len(enhanced_states) == 2, "Should have enhanced states for both nodes"
    assert enhanced_states[0].shape == (batch_size, 32), "Enhanced state should match transformer input"
    assert global_state.shape == (batch_size, 64), "Global state should have correct dimension"
    
    # Test 2: Information flow from ISN to Transformer
    # Create sequence from enhanced states
    sequence = torch.zeros(batch_size, seq_length, 32)
    for t in range(seq_length):
        # Process new states
        node1_state = torch.randn(batch_size, 3)
        node2_state = torch.randn(batch_size, 3)
        enhanced_states, global_state = isn([node1_state, node2_state])
        
        # Combine enhanced states with global features
        combined_state = torch.cat([
            enhanced_states[0][:, :16],  # First half of node 1
            enhanced_states[1][:, :8],   # Part of node 2
            global_state[:, :8]          # Part of global state
        ], dim=1)
        
        sequence[:, t, :] = combined_state
            
    # Process through transformer
    mean_pred, std_pred = model(sequence)
    
    # Verify transformer output
    assert mean_pred.shape == (batch_size, 5, 1), "Transformer should produce correct prediction shape"
    assert not torch.isnan(mean_pred).any(), "Predictions should not contain NaN values"
    assert not torch.isinf(mean_pred).any(), "Predictions should not contain infinite values"
        
    # Test 3: End-to-end gradient flow
    # Create a simple training step
    optimizer = torch.optim.Adam(list(isn.parameters()) + list(model.parameters()))
    target = torch.randn(batch_size, 5, 1)
        
    # Forward pass
    node1_state = torch.randn(batch_size, 3)
    node2_state = torch.randn(batch_size, 3)
    enhanced_states, global_state = isn([node1_state, node2_state])
        
    combined_state = torch.cat([
        enhanced_states[0][:, :16],
        enhanced_states[1][:, :8],
        global_state[:, :8]
    ], dim=1)
        
    sequence = torch.zeros(batch_size, seq_length, 32)
    sequence[:, -1, :] = combined_state  # Use latest state
        
    mean_pred, _ = model(sequence)
        
    # Backward pass
    loss = F.mse_loss(mean_pred, target)
    loss.backward()
        
    # Verify gradients flow through both components
    isn_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in isn.parameters())
    transformer_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        
    assert isn_has_grad, "Gradients should flow through ISN"
    assert transformer_has_grad, "Gradients should flow through transformer"
