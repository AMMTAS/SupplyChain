import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, field
from config.information_sharing_config import NodeConfig, NetworkConfig

@dataclass
class NodeConfig:
    input_dim: int
    hidden_dim: int 
    output_dim: int
    activation: str = 'ReLU'
    dropout: float = 0.1
    delay: int = 0  # Optional delay parameter
    aggregation_type: str = 'mean'  # Options: mean, attention

@dataclass
class NetworkConfig:
    # Required parameters
    global_hidden_dim: int
    
    # Optional parameters - either specify num_nodes + node_config OR num_echelons + node_configs
    num_nodes: Optional[int] = None
    node_config: Optional[NodeConfig] = None
    num_echelons: Optional[int] = None
    node_configs: Optional[List[NodeConfig]] = None
    
    # Network topology
    topology_type: str = 'chain'  # Options: chain, custom
    custom_adjacency: Optional[List[List[int]]] = None
    
    # Noise parameters
    noise_type: str = 'none'  # Options: none, gaussian
    noise_params: Dict = field(default_factory=lambda: {'mean': 0.0, 'std': 0.1})
    
    # Derived attributes
    num_nodes_final: Optional[int] = None
    node_configs_final: Optional[List[NodeConfig]] = None
    topology_final: Optional[List[List[int]]] = None
    
    def __post_init__(self):
        """Validate configuration and set up derived attributes."""
        # Set node configs
        if self.num_nodes is not None and self.node_config is not None:
            self.num_nodes_final = self.num_nodes
            self.node_configs_final = [self.node_config] * self.num_nodes
        elif self.num_echelons is not None and self.node_configs is not None:
            self.num_nodes_final = self.num_echelons
            self.node_configs_final = self.node_configs
        else:
            raise ValueError("Must specify either (num_nodes, node_config) or (num_echelons, node_configs)")
        
        # Validate node configs
        if not self.node_configs_final or len(self.node_configs_final) != self.num_nodes_final:
            raise ValueError("Invalid node configurations")
        
        # Set up adjacency list based on topology
        if self.topology_type == 'chain':
            self.topology_final = []
            for i in range(self.num_nodes_final):
                neighbors = []
                if i > 0:
                    neighbors.append(i-1)
                if i < self.num_nodes_final - 1:
                    neighbors.append(i+1)
                self.topology_final.append(neighbors)
        elif self.topology_type == 'custom' and self.custom_adjacency is not None:
            if len(self.custom_adjacency) != self.num_nodes_final:
                raise ValueError("Custom adjacency list must match number of nodes")
            self.topology_final = self.custom_adjacency
        else:
            raise ValueError("Invalid topology configuration")

class InformationNode(nn.Module):
    """Single node in the Information Sharing Network."""
    
    def __init__(self, config: NodeConfig):
        super().__init__()
        self.config = config
        
        # Feature processing network (also accessible as local_network for backward compatibility)
        self.local_network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
            getattr(nn, config.activation)(),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
        # Input normalization
        self.input_norm = nn.LayerNorm(config.input_dim)
        
        # Delay buffer if needed
        self.delay = config.delay
        if self.delay > 0:
            self.buffer = []  # List for storing delayed states
        
        # Attention for neighbor aggregation if needed
        if config.aggregation_type == 'attention':
            self.attention = nn.Linear(config.output_dim, 1)
    
    def _process_state(self, state: torch.Tensor) -> torch.Tensor:
        """Process a single state through the network."""
        # Ensure state has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        normalized_state = self.input_norm(state)
        return self.local_network(normalized_state)
    
    def _update_delay_buffer(self, state: torch.Tensor) -> torch.Tensor:
        """Update delay buffer and return delayed state."""
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Update buffer
        self.buffer.append(state.detach().cpu())
        if len(self.buffer) > self.delay:
            delayed_state = self.buffer.pop(0)
            return delayed_state.to(state.device)
        
        return torch.zeros_like(state)
    
    def _aggregate_neighbors(self, neighbor_features: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate neighbor features using mean or attention."""
        if not neighbor_features:
            return None
            
        # Stack features for processing
        stacked_features = torch.stack(neighbor_features, dim=1)  # [batch_size, num_neighbors, feature_dim]
        
        if self.config.aggregation_type == 'attention':
            # Calculate attention scores
            scores = self.attention(stacked_features)  # [batch_size, num_neighbors, 1]
            attention_weights = torch.softmax(scores, dim=1)  # [batch_size, num_neighbors, 1]
            
            # Apply attention weights
            weighted_features = stacked_features * attention_weights
            return weighted_features.sum(dim=1)  # [batch_size, feature_dim]
        else:
            # Mean aggregation
            return stacked_features.mean(dim=1)  # [batch_size, feature_dim]
    
    def forward(self, state: torch.Tensor, neighbor_states: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Process state and aggregate with neighbor states if available."""
        # Handle delay if needed
        if self.delay > 0 and self.training:
            state = self._update_delay_buffer(state)
        
        # Process local state
        local_features = self._process_state(state)
        
        # Process and aggregate neighbor states if available
        if neighbor_states:
            neighbor_features = [self._process_state(s) for s in neighbor_states]
            neighbor_agg = self._aggregate_neighbors(neighbor_features)
            if neighbor_agg is not None:
                local_features = local_features + neighbor_agg
        
        return local_features

class InformationSharingNetwork(nn.Module):
    """Network for processing and sharing information between supply chain nodes."""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        # Call post_init if not already called
        if not hasattr(config, 'num_nodes_final'):
            config.__post_init__()
            
        self.config = config
        
        # Create nodes
        self.nodes = nn.ModuleList([
            InformationNode(config.node_configs[i])
            for i in range(config.num_echelons)
        ])
        
        # Global feature network
        total_features = config.num_echelons * config.node_configs[0].output_dim
        self.global_network = nn.Sequential(
            nn.Linear(total_features, config.global_hidden_dim),
            nn.LayerNorm(config.global_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.global_hidden_dim, total_features)
        )
    
    def _get_neighbor_states(self, node_idx: int, node_states: List[torch.Tensor], adj_matrix: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Get states of neighboring nodes based on topology."""
        if adj_matrix is not None:
            # Use provided adjacency matrix
            neighbors = (adj_matrix[node_idx] == 1).nonzero().squeeze(1).tolist()
        else:
            # Use topology based on configuration
            if self.config.topology_type == 'chain':
                neighbors = []
                if node_idx > 0:
                    neighbors.append(node_idx - 1)
                if node_idx < self.config.num_echelons - 1:
                    neighbors.append(node_idx + 1)
            elif self.config.topology_type == 'custom' and self.config.custom_adjacency is not None:
                neighbors = self.config.custom_adjacency[node_idx]
            else:
                # Default to no neighbors
                neighbors = []
        
        return [node_states[i] for i in neighbors]
    
    def _apply_noise(self, states: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply noise to states if configured."""
        if not self.training or self.config.noise_type == 'none':
            return states
            
        noisy_states = []
        for state in states:
            if self.config.noise_type == 'gaussian':
                noise = torch.randn_like(state) * self.config.noise_params.get('std', 0.1)
                noisy_states.append(state + noise)
            elif self.config.noise_type == 'dropout':
                mask = torch.bernoulli(
                    torch.ones_like(state) * (1 - self.config.noise_params.get('p', 0.1))
                )
                noisy_states.append(state * mask)
            elif self.config.noise_type == 'quantization':
                scale = self.config.noise_params.get('scale', 10.0)
                noisy_states.append(torch.round(state * scale) / scale)
            else:
                noisy_states.append(state)
        
        return noisy_states
    
    def forward(self, node_states: List[torch.Tensor], adj_matrix: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Process states through the network.
        
        Args:
            node_states: List of node state tensors [batch_size, input_dim]
            adj_matrix: Optional adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Tuple of:
                - List of enhanced node states [batch_size, output_dim]
                - Global state tensor [batch_size, global_hidden_dim]
        """
        # Apply noise if configured
        if self.training:
            node_states = self._apply_noise(node_states)
        
        # Process each node
        enhanced_states = []
        for i in range(self.config.num_echelons):
            # Get neighbor states based on topology
            neighbor_states = self._get_neighbor_states(i, node_states, adj_matrix)
            
            # Process through node
            enhanced_state = self.nodes[i](node_states[i], neighbor_states)
            enhanced_states.append(enhanced_state)
        
        # Combine enhanced states
        combined_state = torch.cat(enhanced_states, dim=1)
        
        # Process through global network
        global_state = self.global_network(combined_state)
        
        return enhanced_states, global_state
    
    def get_message_vectors(self) -> torch.Tensor:
        """Get the current message vectors being passed between nodes.
        
        Returns:
            Tensor containing message vectors for each node connection
        """
        message_vectors = []
        
        # Iterate through nodes and collect their message vectors
        for i, node in enumerate(self.nodes):
            # Get neighbors based on topology
            neighbors = self.config.topology_final[i]
            
            if not neighbors:
                continue
                
            # Get node's hidden state
            if hasattr(node, 'hidden_state'):
                hidden = node.hidden_state
            else:
                hidden = torch.zeros(1, self.config.node_configs[i].hidden_dim)
                
            # Add message vector for each neighbor
            for neighbor in neighbors:
                message = hidden.detach().clone()  # Detach to avoid gradient issues
                message_vectors.append(message)
                
        # Stack all message vectors
        if message_vectors:
            return torch.cat(message_vectors, dim=0)
        else:
            # Return empty tensor if no messages
            return torch.zeros(0, self.config.node_configs[0].hidden_dim)
