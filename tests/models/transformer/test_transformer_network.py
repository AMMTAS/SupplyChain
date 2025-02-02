"""Tests for the Transformer-based Demand Predictor network."""

import pytest
import torch
import math
import numpy as np
from src.models.transformer.network import DemandPredictor, PositionalEncoding
from config.transformer_config import TransformerConfig
import time
from tests.test_results import TestResults


@pytest.fixture
def config():
    """Create a test configuration."""
    return TransformerConfig(
        input_dim=3,  # inventory, backlog, demand
        output_dim=1,  # demand prediction
        forecast_horizon=5,
        history_length=10,
        d_model=32,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.1
    )


@pytest.fixture
def model(config):
    """Create a test model."""
    return DemandPredictor(config)


def test_positional_encoding():
    """Test positional encoding module."""
    d_model = 32
    max_len = 100
    pos_encoder = PositionalEncoding(d_model, max_len)
    
    # Test output shape
    x = torch.randn(10, 50, d_model)
    out = pos_encoder(x)
    assert out.shape == x.shape
    
    # Test positional encoding properties
    pe = pos_encoder.pe[0]  # [max_len, d_model]
    
    # Test alternating sine and cosine
    assert torch.allclose(
        pe[1:, 0::2],
        torch.sin(torch.arange(1, max_len).float().unsqueeze(1) * torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )),
        atol=1e-6
    )


def test_model_initialization(model, config):
    """Test model initialization."""
    # Test model parameters
    assert isinstance(model, DemandPredictor)
    assert model.config == config
    
    # Test input projection
    assert model.input_proj.in_features == config.input_dim
    assert model.input_proj.out_features == config.d_model
    
    # Test transformer layers
    assert len(model.transformer_encoder.layers) == config.num_encoder_layers
    assert len(model.transformer_decoder.layers) == config.num_decoder_layers


def test_model_forward(model, config):
    """Test model forward pass."""
    batch_size = 16
    seq_len = 10
    
    # Create dummy input
    src = torch.randn(batch_size, seq_len, config.input_dim)
    tgt = torch.randn(batch_size, seq_len).unsqueeze(-1)  # Add feature dimension
    
    # Test training forward pass
    mean, std = model(src, tgt)
    
    # Check output shapes
    assert mean.shape == (batch_size, seq_len, config.output_dim)
    if config.uncertainty_type == 'probabilistic':
        assert std.shape == (batch_size, seq_len, config.output_dim)
        assert torch.all(std > 0)  # Standard deviations should be positive


def test_sequence_generation(model, config):
    """Test autoregressive sequence generation."""
    batch_size = 16
    seq_len = 10
    
    # Create dummy input
    src = torch.randn(batch_size, seq_len, config.input_dim)
    
    # Test inference forward pass
    mean, std = model(src)
    
    # Check output shapes
    assert mean.shape == (batch_size, config.forecast_horizon, config.output_dim)
    if config.uncertainty_type == 'probabilistic':
        assert std.shape == (batch_size, config.forecast_horizon, config.output_dim)
        assert torch.all(std > 0)


def test_attention_mask(model, config):
    """Test attention masking."""
    batch_size = 16
    seq_len = 10
    
    # Create dummy input
    src = torch.randn(batch_size, seq_len, config.input_dim)
    tgt = torch.randn(batch_size, seq_len).unsqueeze(-1)  # Add feature dimension
    
    # Create attention mask
    mask = model._generate_square_subsequent_mask(seq_len)
    
    # Test masked forward pass
    mean, std = model(src, tgt, tgt_mask=mask)
    
    # Check output shapes
    assert mean.shape == (batch_size, seq_len, config.output_dim)
    if config.uncertainty_type == 'probabilistic':
        assert std.shape == (batch_size, seq_len, config.output_dim)


def test_probabilistic_output(model, config):
    """Test probabilistic output properties."""
    if config.uncertainty_type != 'probabilistic':
        pytest.skip("Test only applicable for probabilistic models")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    batch_size = 16
    seq_len = 10
    
    # Create dummy input
    src = torch.randn(batch_size, seq_len, config.input_dim)
    
    # Set model to eval mode for deterministic behavior
    model.eval()
    
    # Generate multiple predictions
    means, stds = [], []
    with torch.no_grad():
        for _ in range(10):
            mean, std = model(src)
            means.append(mean)
            stds.append(std)
    
    means = torch.stack(means)
    stds = torch.stack(stds)
    
    # Check consistency of uncertainty estimates
    assert torch.allclose(stds[0], stds[1], rtol=1e-2)  # Should be deterministic
    assert torch.all(stds > 0)  # All standard deviations should be positive


def test_transformer_network_performance():
    """Test and log transformer network performance."""
    # Create test data
    batch_size = 32
    seq_len = 24
    input_dim = 32
    output_dim = 1
    forecast_horizon = 5
    n_samples = 1000
    n_epochs = 10  # Increase epochs for better convergence

    inputs = torch.randn(n_samples, seq_len, input_dim)
    targets = torch.randn(n_samples, forecast_horizon, output_dim)  # Match forecast horizon

    # Initialize network
    config = TransformerConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        forecast_horizon=forecast_horizon,
        history_length=seq_len,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=128,
        dropout=0.1
    )
    network = DemandPredictor(config)

    # Training loop
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    losses = []
    attention_scores = []

    start_time = time.time()
    for epoch in range(n_epochs):
        epoch_losses = []
        for i in range(0, n_samples, batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]

            # Forward pass
            mean, std = network(batch_inputs)
            loss = torch.nn.functional.mse_loss(mean, batch_targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            attention_scores.append(network._get_attention_weights().mean().item() if network._get_attention_weights() is not None else 0.0)
        
        # Average loss for this epoch
        avg_epoch_loss = np.mean(epoch_losses)
        losses.append(avg_epoch_loss)

    training_time = time.time() - start_time

    # Calculate metrics
    avg_loss = np.mean(losses)
    loss_std = np.std(losses)
    avg_attention = np.mean(attention_scores)

    # Log results
    results = TestResults()
    results.log_transformer_network(
        avg_loss=avg_loss,
        loss_std=loss_std,
        avg_attention=avg_attention,
        training_time=training_time
    )
    results.save_results()

    # Assert expected ranges - allow higher loss since we're using random data
    assert avg_loss <= 10.0, f"Average loss {avg_loss} too high"
    assert loss_std <= 5.0, f"Loss standard deviation {loss_std} too high"
