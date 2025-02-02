"""Tests for Actor-Critic network."""

import pytest
import torch
import numpy as np
from src.models.actor_critic.network import ActorCriticNetwork


@pytest.fixture
def network():
    """Create test network."""
    return ActorCriticNetwork(
        demand_dim=24,    # 24 time steps of demand predictions
        fuzzy_dim=2,     # Order adjustment and risk level
        moea_dim=8,      # MOEA parameters
        action_dim=3,    # Order quantities for 3 echelons
        hidden_dim=128,  # Smaller for testing
        n_hidden=2
    )


@pytest.fixture
def sample_batch():
    """Create sample batch of data."""
    batch_size = 32
    return {
        'demand_preds': torch.randn(batch_size, 24),
        'fuzzy_recs': torch.randn(batch_size, 2),
        'moea_params': torch.randn(batch_size, 8),
        'actions': torch.randn(batch_size, 3),
        'old_log_probs': torch.randn(batch_size, 3),
        'advantages': torch.randn(batch_size),
        'returns': torch.randn(batch_size)
    }


def test_initialization(network):
    """Test network initialization."""
    assert isinstance(network, ActorCriticNetwork)
    assert network.demand_dim == 24
    assert network.fuzzy_dim == 2
    assert network.moea_dim == 8
    assert network.action_dim == 3


def test_forward_pass(network):
    """Test forward pass through network."""
    batch_size = 16
    demand_pred = torch.randn(batch_size, 24)
    fuzzy_rec = torch.randn(batch_size, 2)
    moea_params = torch.randn(batch_size, 8)
    
    action_dist, value = network(demand_pred, fuzzy_rec, moea_params)
    
    assert isinstance(action_dist, torch.distributions.Distribution)
    assert value.shape == (batch_size, 1)  # Value has shape [batch_size, 1]


def test_select_action(network):
    """Test action selection."""
    demand_pred = np.random.randn(24)
    fuzzy_rec = np.random.randn(2)
    moea_params = np.random.randn(8)
    
    # Test deterministic
    action, info = network.select_action(
        demand_pred, fuzzy_rec, moea_params, deterministic=True
    )
    assert action.shape == (3,)
    assert 'value' in info
    assert 'log_prob' in info
    assert 'entropy' in info
    
    # Test stochastic
    action, info = network.select_action(
        demand_pred, fuzzy_rec, moea_params, deterministic=False
    )
    assert action.shape == (3,)


def test_update(network, sample_batch):
    """Test network update."""
    metrics = network.update(
        demand_preds=sample_batch['demand_preds'],
        fuzzy_recs=sample_batch['fuzzy_recs'],
        moea_params=sample_batch['moea_params'],
        actions=sample_batch['actions'],
        old_log_probs=sample_batch['old_log_probs'],
        advantages=sample_batch['advantages'],
        returns=sample_batch['returns']
    )
    
    assert 'policy_loss' in metrics
    assert 'value_loss' in metrics
    assert 'entropy' in metrics
    assert 'approx_kl' in metrics
    assert 'early_stop' in metrics


def test_early_stopping(network, sample_batch):
    """Test early stopping based on KL divergence."""
    # Set a very low target_kl to trigger early stopping
    network.target_kl = 1e-6
    
    metrics = network.update(
        demand_preds=sample_batch['demand_preds'],
        fuzzy_recs=sample_batch['fuzzy_recs'],
        moea_params=sample_batch['moea_params'],
        actions=sample_batch['actions'],
        old_log_probs=sample_batch['old_log_probs'],
        advantages=sample_batch['advantages'],
        returns=sample_batch['returns']
    )
    
    assert metrics['early_stop']


def test_feature_encoding(network):
    """Test feature encoding for different inputs."""
    batch_size = 8
    demand_pred = torch.randn(batch_size, 24)
    fuzzy_rec = torch.randn(batch_size, 2)
    moea_params = torch.randn(batch_size, 8)
    
    # Get features
    demand_features = network.demand_encoder(demand_pred)
    fuzzy_features = network.fuzzy_encoder(fuzzy_rec)
    moea_features = network.moea_encoder(moea_params)
    
    assert demand_features.shape == (batch_size, 128)
    assert fuzzy_features.shape == (batch_size, 128)
    assert moea_features.shape == (batch_size, 128)
    
    # Test feature combination
    combined = torch.cat([
        demand_features,
        fuzzy_features,
        moea_features
    ], dim=-1)
    features = network.feature_combiner(combined)
    
    assert features.shape == (batch_size, 128)
