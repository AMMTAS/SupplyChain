"""Configuration for Information Sharing Network."""

from dataclasses import dataclass
from typing import List, Optional, Dict
import torch


@dataclass
class NodeConfig:
    """Configuration for a single information node.
    
    Args:
        input_dim: Input dimension for the node
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        delay: Information delay in timesteps
        dropout: Dropout rate for regularization
        activation: Activation function to use
        aggregation_type: How to aggregate neighbor information
            ('mean', 'attention', 'gated')
    """
    input_dim: int
    hidden_dim: int
    output_dim: int
    delay: int = 0
    dropout: float = 0.1
    activation: str = 'ReLU'
    aggregation_type: str = 'attention'


@dataclass
class NetworkConfig:
    """Configuration for the entire Information Sharing Network.
    
    Args:
        num_echelons: Number of echelons in the supply chain
        node_configs: List of configurations for each node
        global_hidden_dim: Hidden dimension for global processing
        noise_type: Type of information noise to apply
            ('gaussian', 'dropout', 'quantization')
        noise_params: Parameters for the noise model
        topology_type: How nodes are connected
            ('chain', 'fully_connected', 'custom')
        custom_adjacency: Custom adjacency matrix if topology_type is 'custom'
    """
    num_echelons: int
    node_configs: List[NodeConfig]
    global_hidden_dim: int
    noise_type: str = 'gaussian'
    noise_params: Dict = None
    topology_type: str = 'chain'
    custom_adjacency: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert len(self.node_configs) == self.num_echelons, \
            "Number of node configs must match number of echelons"
        
        if self.noise_params is None:
            self.noise_params = {'mean': 0.0, 'std': 0.1}
        
        if self.topology_type == 'custom':
            assert self.custom_adjacency is not None, \
                "Must provide custom_adjacency when topology_type is 'custom'"
            assert self.custom_adjacency.shape == (self.num_echelons, self.num_echelons), \
                "Custom adjacency matrix must be square with size num_echelons"
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        """Generate adjacency matrix based on topology type.
        
        Returns:
            Binary adjacency matrix defining node connections
        """
        adj = torch.zeros(self.num_echelons, self.num_echelons)
        
        if self.topology_type == 'chain':
            # Each node connected to adjacent nodes
            for i in range(self.num_echelons - 1):
                adj[i, i+1] = 1
                adj[i+1, i] = 1
        
        elif self.topology_type == 'fully_connected':
            # All nodes connected to all other nodes
            adj = torch.ones(self.num_echelons, self.num_echelons)
            adj.fill_diagonal_(0)  # No self-connections
        
        elif self.topology_type == 'custom':
            adj = self.custom_adjacency
        
        return adj
