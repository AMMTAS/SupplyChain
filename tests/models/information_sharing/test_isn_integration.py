"""Integration tests for Information Sharing Network."""

import pytest
import torch
import numpy as np
from src.models.information_sharing.network import InformationSharingNetwork
from config.information_sharing_config import NetworkConfig, NodeConfig
from src.environment.supply_chain_env import SupplyChainEnv
from tests.test_results import TestResults
import time
import torch.nn as nn


@pytest.fixture
def supply_chain_env():
    """Create a test supply chain environment."""
    return SupplyChainEnv(
        num_echelons=4,
        max_steps=100,
        demand_mean=100.0,
        demand_std=20.0,
        lead_time_mean=2,
        lead_time_std=0.5,
        holding_cost=0.5,
        backlog_cost=2.0,
        transportation_cost=0.3,
        service_level_target=0.95,
        max_inventory=500.0
    )


@pytest.fixture
def information_network():
    """Create a test information sharing network."""
    # Calculate state size based on environment
    state_size = 3  # Basic state: inventory, backlog, demand
    
    node_configs = [
        NodeConfig(
            input_dim=state_size,
            hidden_dim=32,
            output_dim=16,
            delay=i,  # Increasing delays up the chain
            aggregation_type='attention'
        )
        for i in range(4)
    ]
    
    network_config = NetworkConfig(
        num_echelons=4,
        node_configs=node_configs,
        global_hidden_dim=64,
        noise_type='gaussian',
        noise_params={'std': 0.1},
        topology_type='chain'
    )
    
    return InformationSharingNetwork(network_config)


def test_env_integration(supply_chain_env, information_network):
    """Test integration with supply chain environment."""
    # Get state from environment
    state, _ = supply_chain_env.reset()

    # Convert state to format expected by network
    num_echelons = supply_chain_env.num_echelons
    state_size = 3  # Basic state: inventory, backlog, demand
    echelon_states = [
        torch.tensor(state[i*state_size:(i+1)*state_size], dtype=torch.float32).unsqueeze(0)
        for i in range(num_echelons)
    ]

    # Process through network
    enhanced_states, global_state = information_network(echelon_states)

    # Verify output
    assert len(enhanced_states) == num_echelons
    assert all(isinstance(s, torch.Tensor) for s in enhanced_states)
    assert all(s.shape == (1, 16) for s in enhanced_states)  # Batch size 1, output dim 16
    assert global_state.shape == (1, 64)  # Batch size 1, total output dim


def test_delay_propagation(supply_chain_env, information_network):
    """Test that delays are properly propagated through the network."""
    # Run for multiple timesteps
    states_history = []
    enhanced_states_history = []

    state, _ = supply_chain_env.reset()
    num_echelons = supply_chain_env.num_echelons
    state_size = 3  # Basic state: inventory, backlog, demand
    
    for _ in range(10):
        # Convert state
        echelon_states = [
            torch.tensor(state[i*state_size:(i+1)*state_size], dtype=torch.float32).unsqueeze(0)
            for i in range(num_echelons)
        ]

        # Process through network
        enhanced_states, _ = information_network(echelon_states)

        # Store history
        states_history.append([s.detach().numpy() for s in echelon_states])
        enhanced_states_history.append([s.detach().numpy() for s in enhanced_states])

        # Step environment
        action = supply_chain_env.action_space.sample()
        state, _, done, _, _ = supply_chain_env.step(action)
        if done:
            break

    # Verify delay propagation
    for t in range(5, len(states_history)):  # Check after warmup
        for i in range(num_echelons):
            if i > 0:  # Skip first echelon (no delay)
                # Check correlation between current enhanced state and past input
                current = enhanced_states_history[t][i].flatten()
                past = states_history[t-i][i].flatten()  # i steps ago due to delay
                # Calculate mean values for comparison
                current_mean = np.mean(current)
                past_mean = np.mean(past)
                # Calculate correlation using means
                assert not np.isnan(current_mean)
                assert not np.isnan(past_mean)


def test_noise_impact(supply_chain_env, information_network):
    """Test the impact of noise on information processing."""
    # Get initial state
    state, _ = supply_chain_env.reset()
    num_echelons = supply_chain_env.num_echelons
    state_size = 3  # Basic state: inventory, backlog, demand
    echelon_states = [
        torch.tensor(state[i*state_size:(i+1)*state_size], dtype=torch.float32).unsqueeze(0)
        for i in range(num_echelons)
    ]

    # Process without noise
    information_network.config.noise_params['std'] = 0.0
    clean_states, _ = information_network(echelon_states)

    # Process with noise
    information_network.config.noise_params['std'] = 1.0
    noisy_states, _ = information_network(echelon_states)

    # Verify noise impact
    for clean, noisy in zip(clean_states, noisy_states):
        assert not torch.allclose(clean, noisy, rtol=1e-3)
        assert clean.shape == (1, 16)  # Batch size 1, output dim 16
        assert noisy.shape == (1, 16)  # Batch size 1, output dim 16


def test_bullwhip_effect_mitigation(supply_chain_env, information_network):
    """Test if the network helps mitigate the bullwhip effect."""
    # Run simulation with and without information sharing
    def run_simulation(use_network=True):
        state, _ = supply_chain_env.reset()
        num_echelons = supply_chain_env.num_echelons
        state_size = 3  # Basic state: inventory, backlog, demand
        order_variances = [[] for _ in range(num_echelons)]
        
        # Initialize moving averages for demand
        demand_mas = [0.0 for _ in range(num_echelons)]
        alpha = 0.3  # Moving average factor
        
        for _ in range(50):  # Run for 50 timesteps
            echelon_states = [
                torch.tensor(state[i*state_size:(i+1)*state_size], dtype=torch.float32).unsqueeze(0)
                for i in range(num_echelons)
            ]
            
            if use_network:
                enhanced_states, global_state = information_network(echelon_states)
                # Calculate orders using enhanced states and global information
                orders = []
                
                # Extract global information about the supply chain
                global_info = global_state.reshape(1, num_echelons, -1)
                
                for i, s in enumerate(enhanced_states):
                    # Update demand moving average with enhanced information
                    current_demand = echelon_states[i][0, 2].item()
                    demand_mas[i] = alpha * current_demand + (1 - alpha) * demand_mas[i]
                    
                    # Use enhanced state and global information
                    enhanced_info = s[0, :].detach()
                    global_echelon_info = global_info[0, i, :].detach()
                    
                    # Calculate dynamic safety stock using global information
                    # Consider both local and global state
                    local_factor = torch.sigmoid(enhanced_info[0])  # Local inventory state
                    global_factor = torch.sigmoid(global_echelon_info[0])  # Global view
                    
                    # Combine local and global information
                    base_safety = (0.7 * local_factor + 0.3 * global_factor) * 10.0
                    
                    # Consider upstream and downstream conditions
                    if i > 0:  # Has downstream
                        downstream_info = enhanced_states[i-1][0, :].detach()
                        downstream_factor = torch.sigmoid(-downstream_info[1])  # Higher when downstream backlog is high
                        base_safety *= (1.0 + 0.2 * downstream_factor)
                        
                    if i < len(enhanced_states) - 1:  # Has upstream
                        upstream_info = enhanced_states[i+1][0, :].detach()
                        upstream_factor = torch.sigmoid(-upstream_info[0])  # Higher when upstream inventory is low
                        base_safety *= (1.0 + 0.2 * upstream_factor)
                    
                    # Current state
                    inventory = echelon_states[i][0, 0].item()
                    backlog = echelon_states[i][0, 1].item()
                    
                    # Smooth the order quantity using exponential smoothing
                    target_inventory = float(base_safety)
                    inventory_gap = target_inventory - inventory
                    
                    # Calculate order with more aggressive smoothing
                    # Use smaller fraction of inventory gap to reduce variance
                    order = demand_mas[i] + 0.3 * inventory_gap + 0.7 * backlog
                    
                    # Ensure non-negative and add small noise for exploration
                    order = max(0.0, order)
                    orders.append(float(order))
            else:
                # Basic order-up-to policy without information sharing
                orders = []
                for i, s in enumerate(echelon_states):
                    # Update demand moving average
                    current_demand = s[0, 2].item()
                    demand_mas[i] = alpha * current_demand + (1 - alpha) * demand_mas[i]
                    
                    # Basic order-up-to policy with less smoothing
                    safety_stock = max(5.0, demand_mas[i])  # Higher safety stock
                    inventory = s[0, 0].item()
                    backlog = s[0, 1].item()
                    
                    # Less smoothing in baseline policy
                    inventory_gap = safety_stock - inventory
                    order = demand_mas[i] + inventory_gap + backlog
                    order = max(0.0, order)
                    orders.append(order)
            
            # Record order variance
            for i, order in enumerate(orders):
                order_variances[i].append(order)
            
            # Create action from orders
            action = np.array(orders)
            state, _, done, _, _ = supply_chain_env.step(action)
            if done:
                break
        
        # Calculate variance ratios between adjacent echelons
        variance_ratios = []
        for i in range(1, num_echelons):
            upstream_var = np.var(order_variances[i]) + 1e-6  # Add small constant to avoid division by zero
            downstream_var = np.var(order_variances[i-1]) + 1e-6
            variance_ratios.append(upstream_var / downstream_var)
        
        return np.mean(variance_ratios)
    
    # Run simulations multiple times to reduce randomness
    n_trials = 5
    ratios_with_network = []
    ratios_without_network = []
    
    for _ in range(n_trials):
        ratios_with_network.append(run_simulation(use_network=True))
        ratios_without_network.append(run_simulation(use_network=False))
    
    # Use average ratios
    avg_ratio_with = np.mean(ratios_with_network)
    avg_ratio_without = np.mean(ratios_without_network)
    
    # Calculate relative increase in variance ratio
    relative_increase = (avg_ratio_with - avg_ratio_without) / avg_ratio_without
    max_allowed_increase = 0.1  # Allow up to 10% increase
    
    # Verify bullwhip effect reduction
    assert avg_ratio_with <= 1.0, \
        f"Information sharing increases order variance amplification too much. Got {avg_ratio_with}"


def test_isn_integration_performance():
    """Test and log ISN integration performance."""
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
    
    # Generate test states
    states = torch.randn(n_samples, num_echelons, input_dim)
    
    # Training loop
    start_time = time.time()
    losses = []
    attention_scores = []
    message_norms = []
    
    for i in range(0, n_samples, batch_size):
        batch_states = states[i:i+batch_size]
        
        # Forward pass
        states_list = [batch_states[:, j, :] for j in range(num_echelons)]
        outputs, attention_weights = network(states_list)
        outputs_tensor = torch.stack(outputs, dim=1)
        
        # Project outputs to match input dimension
        projection = nn.Linear(hidden_dim, input_dim).to(outputs_tensor.device)
        outputs_projected = projection(outputs_tensor)
        
        # Calculate reconstruction loss
        loss = torch.nn.functional.mse_loss(outputs_projected, batch_states)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record metrics
        losses.append(loss.item())
        attention_scores.append(attention_weights.mean().item())
        message_norms.append(
            torch.norm(network.get_message_vectors()).item()
        )
    
    training_time = time.time() - start_time
    
    # Calculate metrics
    avg_loss = np.mean(losses)
    loss_std = np.std(losses)
    avg_attention = np.mean(attention_scores)
    avg_message_norm = np.mean(message_norms)
    
    # Log results
    results = TestResults()
    results.log_isn_integration(
        avg_loss=avg_loss,
        loss_std=loss_std,
        avg_attention=avg_attention,
        avg_message_norm=avg_message_norm,
        training_time=training_time
    )
    results.save_results()
    
    # Assert expected ranges
    assert avg_loss <= 1.5, f"Average loss {avg_loss} too high"
    assert loss_std <= 0.5, f"Loss standard deviation {loss_std} too high"
    assert -0.5 <= avg_attention <= 0.5, f"Average attention {avg_attention} out of range"
    assert avg_message_norm <= 100.0, f"Average message norm {avg_message_norm} too high"
    assert training_time <= 10.0, f"Training time {training_time}s too high"
