"""Test integration between ISN and Fuzzy Controller."""

import pytest
import torch
import numpy as np
from src.models.information_sharing.network import (
    InformationSharingNetwork,
    NetworkConfig,
    NodeConfig
)
from src.models.fuzzy.controller import FuzzyController, FuzzyControllerConfig

@pytest.fixture
def isn():
    """Create ISN for testing."""
    node_config = NodeConfig(
        input_dim=3,  # inventory, backlog, demand
        hidden_dim=64,
        output_dim=32,
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
    
    return InformationSharingNetwork(network_config)

@pytest.fixture
def fuzzy_controller():
    """Create fuzzy controller for testing."""
    config = FuzzyControllerConfig(
        input_dim=32,  # Match ISN output
        n_membership_functions=3,
        universe_range=(-1.0, 1.0),
        defuzz_method='centroid'
    )
    return FuzzyController(config)

def test_isn_fuzzy_integration(isn, fuzzy_controller):
    """Test integration between ISN and Fuzzy Controller.
    
    This test verifies that:
    1. ISN can process supply chain states
    2. ISN outputs can be used by fuzzy controller
    3. Information flows correctly between components
    4. Fuzzy rules are properly applied to ISN-enhanced states
    """
    # Test 1: Dimension compatibility
    batch_size = 32
    
    # Create test batch for each node
    node1_state = torch.randn(batch_size, 3)  # inventory, backlog, demand
    node2_state = torch.randn(batch_size, 3)
    
    # Process through ISN
    node_states = [node1_state, node2_state]
    enhanced_states, global_state = isn(node_states)
    
    # Verify ISN output dimensions
    assert len(enhanced_states) == 2, "Should have enhanced states for both nodes"
    assert enhanced_states[0].shape == (batch_size, 32), "Enhanced state should match fuzzy input"
    assert global_state.shape == (batch_size, 64), "Global state should have correct dimension"
    
    # Test 2: Information flow from ISN to Fuzzy
    # Convert enhanced states to numpy for fuzzy controller
    enhanced_state_np = enhanced_states[0].detach().numpy()
    
    # Process through fuzzy controller
    fuzzy_outputs = []
    for state in enhanced_state_np:
        # Extract first three components as inventory, demand, service level
        inventory = state[0]
        demand = state[1]
        service = state[2]
        
        # Create state vector
        state_vector = np.array([inventory, demand, service])
        recommendation = fuzzy_controller.process_state(state_vector)
        fuzzy_outputs.append([
            recommendation['order_adjustment'],
            recommendation['risk_level']
        ])
    fuzzy_outputs = np.array(fuzzy_outputs)
    
    # Verify fuzzy controller outputs
    assert fuzzy_outputs.shape == (batch_size, 2), "Should output order adjustment and risk level"
    assert np.all(fuzzy_outputs[:, 1] >= 0) and np.all(fuzzy_outputs[:, 1] <= 1), "Risk level should be between 0 and 1"
    
    # Test 3: State change propagation
    # Create states with very different inventory levels
    very_low_inventory = np.array([-0.9, 0.0, 0.0])  # Very low inventory
    very_high_inventory = np.array([0.9, 0.0, 0.0])  # Very high inventory
    
    # Process through fuzzy controller
    fuzzy_low = fuzzy_controller.process_state(very_low_inventory)
    fuzzy_high = fuzzy_controller.process_state(very_high_inventory)
    
    # Verify fuzzy controller responds to inventory changes
    # The controller should recommend increase_lot for very low inventory
    # and decrease_lot for very high inventory
    assert fuzzy_low['order_adjustment'] > 0.5, \
        "Should recommend significant increase for very low inventory"
    assert fuzzy_high['order_adjustment'] < -0.5, \
        "Should recommend significant decrease for very high inventory"
    assert fuzzy_low['risk_level'] > 0.6, \
        "Risk should be high when inventory is very low"
    assert fuzzy_high['risk_level'] > 0.6, \
        "Risk should be high when inventory is very high"
