"""Example configurations and usage of Information Sharing Network."""

import torch
from typing import List
from .config import NodeConfig, NetworkConfig
from .network import InformationSharingNetwork


def create_basic_chain_network(
    num_echelons: int,
    input_dim: int,
    hidden_dim: int = 64,
    output_dim: int = 32
) -> InformationSharingNetwork:
    """Create a basic chain topology network.
    
    Args:
        num_echelons: Number of echelons in supply chain
        input_dim: Input dimension for each node
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension for each node
    
    Returns:
        Configured network instance
    """
    # Create node configs
    node_configs = [
        NodeConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            delay=1 if i > 0 else 0,  # Add delay to non-retailer nodes
            dropout=0.1,
            activation='ReLU',
            aggregation_type='attention'
        )
        for i in range(num_echelons)
    ]
    
    # Create network config
    network_config = NetworkConfig(
        num_echelons=num_echelons,
        node_configs=node_configs,
        global_hidden_dim=hidden_dim * 2,
        noise_type='gaussian',
        noise_params={'mean': 0.0, 'std': 0.1},
        topology_type='chain'
    )
    
    return InformationSharingNetwork(network_config)


def create_fully_connected_network(
    num_echelons: int,
    input_dim: int,
    hidden_dim: int = 64,
    output_dim: int = 32
) -> InformationSharingNetwork:
    """Create a fully connected network topology.
    
    Args:
        num_echelons: Number of echelons in supply chain
        input_dim: Input dimension for each node
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension for each node
    
    Returns:
        Configured network instance
    """
    # Create node configs with gated aggregation
    node_configs = [
        NodeConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            delay=0,  # No delays in fully connected network
            dropout=0.1,
            activation='ReLU',
            aggregation_type='gated'
        )
        for _ in range(num_echelons)
    ]
    
    # Create network config
    network_config = NetworkConfig(
        num_echelons=num_echelons,
        node_configs=node_configs,
        global_hidden_dim=hidden_dim * 2,
        noise_type='dropout',
        noise_params={'p': 0.1},
        topology_type='fully_connected'
    )
    
    return InformationSharingNetwork(network_config)


def example_usage():
    """Example of how to use the network."""
    # Create a 4-echelon supply chain network
    network = create_basic_chain_network(
        num_echelons=4,
        input_dim=10  # Each state has 10 features
    )
    
    # Create sample batch of states
    batch_size = 32
    states = [
        torch.randn(batch_size, 10)  # [batch_size, input_dim]
        for _ in range(4)
    ]
    
    # Process states through network
    enhanced_states, global_state = network(states)
    
    # Print shapes
    print("Enhanced states shapes:")
    for i, state in enumerate(enhanced_states):
        print(f"Echelon {i}: {state.shape}")
    print(f"Global state shape: {global_state.shape}")


if __name__ == "__main__":
    example_usage()
