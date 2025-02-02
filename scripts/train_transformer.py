"""Script to train the transformer model on real supply chain data."""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.models.transformer.trainer import DemandPredictorTrainer
from src.models.information_sharing.network import InformationSharingNetwork
from config.transformer_config import TransformerConfig
from config.information_sharing_config import NodeConfig, NetworkConfig
from src.environment.supply_chain_env import SupplyChainEnv

def create_isn():
    """Create information sharing network."""
    node_config = NodeConfig(
        input_dim=3,  # inventory, backlog, demand
        hidden_dim=64,
        output_dim=64,
        delay=2,
        dropout=0.1,
        activation='ReLU',
        aggregation_type='attention'
    )
    
    network_config = NetworkConfig(
        num_echelons=3,  # More realistic supply chain
        node_configs=[node_config] * 3,
        global_hidden_dim=64,
        noise_type='gaussian',
        topology_type='chain'
    )
    
    return InformationSharingNetwork(network_config)

def generate_training_data(env, isn, n_episodes=100, episode_length=52):
    """Generate training data using environment and ISN."""
    sequences = []
    targets = []
    
    for _ in range(n_episodes):
        state = env.reset()
        
        # Store states and demands for this episode
        episode_states = []
        episode_demands = []
        
        for t in range(episode_length):
            # Random action for data collection
            action = np.random.uniform(0, 2, size=env.num_echelons)
            
            # Step environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Get enhanced state from ISN
            with torch.no_grad():
                # Convert state into list of node states
                node_states = []
                for i in range(env.num_echelons):
                    # Extract inventory, backlog, and demand for each node
                    node_state = np.array([
                        next_state[i],  # inventory
                        next_state[i + env.num_echelons],  # backlog
                        next_state[-2]  # current demand (same for all nodes)
                    ])
                    node_states.append(torch.tensor(node_state, dtype=torch.float32).unsqueeze(0))
                
                enhanced_states, _ = isn(node_states)
                
                # Store first node's enhanced state
                episode_states.append(enhanced_states[0][0].numpy())
                episode_demands.append(next_state[-2])  # Store current demand
            
            if done or truncated:
                break
        
        # Create sequences and targets
        for i in range(len(episode_states) - 15):  # 10 for history, 5 for forecast
            seq = episode_states[i:i+10]  # 10 timesteps history
            target = episode_demands[i+10:i+15]  # 5 timesteps forecast
            
            sequences.append(np.stack(seq))
            targets.append(np.array(target))
    
    return (torch.tensor(np.stack(sequences), dtype=torch.float32),
            torch.tensor(np.stack(targets), dtype=torch.float32))

def main():
    # Create environment
    env = SupplyChainEnv(
        num_echelons=3,  # 3-echelon supply chain
        max_steps=52,  # 1 year of weekly data
        demand_mean=100,
        demand_std=20,
        lead_time_mean=2,
        lead_time_std=0.5,
        service_level_target=0.95,
        max_inventory=500.0
    )
    
    # Create ISN
    isn = create_isn()
    
    # Generate data
    print("Generating training data...")
    train_x, train_y = generate_training_data(env, isn, n_episodes=100)
    val_x, val_y = generate_training_data(env, isn, n_episodes=20)
    
    # Create config
    config = TransformerConfig(
        input_dim=64,  # ISN hidden size
        output_dim=1,  # Demand prediction
        forecast_horizon=5,
        history_length=10,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        batch_size=32,
        max_epochs=100,
        learning_rate=1e-3,
        uncertainty_type='probabilistic'
    )
    
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
    
    # Create trainer
    trainer = DemandPredictorTrainer(config, train_loader, val_loader)
    
    # Train model
    print("Training model...")
    metrics = trainer.train()
    
    print(f"Final metrics: {metrics}")
    
    # Save model
    torch.save(trainer.model.state_dict(), "transformer_model.pt")
    print("Model saved to transformer_model.pt")

if __name__ == "__main__":
    main()
