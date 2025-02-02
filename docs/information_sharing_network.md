# Information Sharing Network (ISN)

## Overview
The Information Sharing Network (ISN) is a key component of the supply chain system that enables efficient information flow between different echelons. It processes and shares inventory, backlog, and demand information across the supply chain network.

## Components

### InformationNode
The `InformationNode` class represents a single node in the network. Each node:
- Processes local state information (inventory, backlog, demand)
- Normalizes input data
- Extracts features using a neural network
- Aggregates information from neighboring nodes

Key features:
- Input normalization using LayerNorm
- Configurable feature network architecture
- Support for different aggregation types (mean, attention)

### InformationSharingNetwork
The `InformationSharingNetwork` class manages the overall network topology and information flow. It:
- Coordinates multiple information nodes
- Handles network topology (chain, custom)
- Processes global state information
- Applies noise for robustness

Configuration options:
- Number of echelons
- Node configurations
- Global hidden dimensions
- Noise type and parameters
- Network topology

## Usage

```python
# Create node configuration
node_config = NodeConfig(
    input_dim=3,  # inventory, backlog, demand
    hidden_dim=64,
    output_dim=32,
    dropout=0.1,
    activation='LeakyReLU',
    aggregation_type='attention'
)

# Create network configuration
network_config = NetworkConfig(
    num_echelons=4,
    node_configs=[node_config] * 4,  # One config per node
    global_hidden_dim=64,
    noise_type='gaussian',
    noise_params={'std': 0.1},
    topology_type='chain'
)

# Initialize network
network = InformationSharingNetwork(network_config)

# Process states
node_states = [torch.randn(1, 3) for _ in range(4)]  # One state per node
enhanced_states, global_state = network(node_states)
```

## Testing
The ISN implementation includes comprehensive tests:
- Basic forward pass tests
- Information flow verification
- Batch processing capability
- Feature consistency checks
- Gradient flow tests

## Integration
The ISN can be integrated with:
- Supply chain environment
- Fuzzy controllers
- Transformer models for demand prediction
- Actor-critic networks for policy learning

## Performance Considerations
- Use batch processing when possible
- Consider network topology for scalability
- Monitor memory usage with large networks
- Tune noise parameters for robustness
