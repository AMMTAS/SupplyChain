"""
Supply Chain Management System with Intelligent Decision Support

This script integrates all components of the supply chain management system:
1. Environment - Simulates the supply chain dynamics
2. Information Sharing Network - Enhances state information across echelons
3. Transformer - Predicts future demand patterns
4. Fuzzy Controller - Provides rule-based recommendations
5. MOEA - Optimizes system parameters
6. Actor-Critic - Makes final decisions based on all inputs

The system aims to minimize costs while maintaining high service levels and
reducing the bullwhip effect through intelligent decision-making.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.environment.supply_chain_env import SupplyChainEnv
from src.models.information_sharing.network import InformationNode
from src.models.transformer.network import DemandPredictor
from src.models.fuzzy.controller import FuzzyController, FuzzyControllerConfig
from src.models.moea.optimizer import MOEAOptimizer, MOEAConfig
from src.models.moea.objectives import TotalCost, ServiceLevel, BullwhipEffect
from src.models.actor_critic.network import ActorCriticNetwork
from config.information_sharing_config import NodeConfig, NetworkConfig
from config.transformer_config import TransformerConfig

@dataclass
class SystemConfig:
    """Configuration for the supply chain system."""
    n_echelons: int = 4
    episode_length: int = 100
    state_size: int = 14  # 4 echelons * (inventory + backlog + on_order) + demand + avg_demand
    hidden_size: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    demand_history_len: int = 10
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
class SupplyChainSystem:
    """Integrated supply chain management system."""
    
    def __init__(self, config: SystemConfig):
        """Initialize all components."""
        self.config = config
        
        # Initialize environment
        self.env = SupplyChainEnv(
            num_echelons=config.n_echelons,
            max_steps=config.episode_length
        )
        
        # Initialize Information Sharing Network
        isn_config = NodeConfig(
            input_dim=config.state_size,  # Match environment state size
            hidden_dim=config.hidden_size,
            output_dim=config.hidden_size,
            delay=2,
            dropout=config.dropout,
            activation='ReLU'
        )
        self.isn = InformationNode(config=isn_config)
        
        # Initialize Demand Predictor (Transformer)
        transformer_config = TransformerConfig(
            input_dim=1,  # Single demand value per timestep
            d_model=config.hidden_size,
            nhead=8,
            dim_feedforward=config.hidden_size * 4,
            num_encoder_layers=config.n_layers,
            num_decoder_layers=config.n_layers,
            dropout=config.dropout,
            max_seq_length=100,
            output_dim=1,
            uncertainty_type='probabilistic',
            forecast_horizon=10,
            history_length=config.demand_history_len
        )
        self.transformer = DemandPredictor(config=transformer_config)
        
        # Initialize Fuzzy Controller
        fuzzy_config = FuzzyControllerConfig(
            input_dim=config.hidden_size,  # Match ISN output size
            n_membership_functions=3,  # Low, Medium, High
            universe_range=(-1.0, 1.0),  # Normalized range
            defuzz_method='centroid'  # Standard defuzzification method
        )
        self.fuzzy = FuzzyController(fuzzy_config)
        
        # Initialize MOEA Optimizer
        objectives = [
            TotalCost(holding_cost=1.0, backlog_cost=5.0),
            ServiceLevel(),
            BullwhipEffect(window_size=10)
        ]
        parameter_bounds = {
            'reorder_point': (-100, 100),
            'order_quantity': (0, 200),
            'safety_stock': (0, 100)
        }
        moea_config = MOEAConfig(
            population_size=100,
            num_neighbors=20,
            max_evaluations=10000,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        self.moea = MOEAOptimizer(
            objectives=objectives,
            parameter_bounds=parameter_bounds,
            config=moea_config
        )
        
        # Initialize Actor-Critic Network
        self.actor_critic = ActorCriticNetwork(
            demand_dim=2,  # Demand prediction with uncertainty
            fuzzy_dim=2,  # Order adjustment and risk level
            moea_dim=3,  # Reorder point, order quantity, safety stock
            action_dim=config.n_echelons,  # One action per echelon
            hidden_dim=config.hidden_size,
            n_hidden=config.n_layers,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_ratio=config.clip_ratio,
            target_kl=config.target_kl,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.isn.parameters()) +
            list(self.transformer.parameters()) +
            list(self.actor_critic.parameters()),
            lr=config.learning_rate
        )
        
        self.last_action_info = None
        
    def process_state(self, state: np.ndarray) -> Dict[str, torch.Tensor]:
        """Process raw state through ISN, Transformer, and Fuzzy Controller.
        
        Args:
            state: Current environment state
            
        Returns:
            Dictionary containing processed state information
        """
        # Convert state to tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, state_size]
        
        # Process through Information Sharing Network
        isn_state = self.isn(state_tensor)  # [1, hidden_size]
        
        # Extract demand history from state
        demand_history = state[:self.config.demand_history_len]  # [history_len]
        demand_history = demand_history.reshape(-1, 1)  # [history_len, 1]
        demand_history = torch.FloatTensor(demand_history).unsqueeze(0)  # [1, history_len, 1]
        
        # Get demand predictions from Transformer
        demand_pred, uncertainty = self.transformer(demand_history)  # [1, horizon, 1], [1, horizon, 1]
        demand_pred = demand_pred.squeeze(-1)  # [1, horizon]
        uncertainty = uncertainty.squeeze(-1)  # [1, horizon]
        
        # Get rule-based recommendations from Fuzzy Controller
        fuzzy_output = self.fuzzy.process_state(state)  # Process raw state
        fuzzy_rec = torch.FloatTensor([
            fuzzy_output['order_adjustment'],
            fuzzy_output['risk_level']
        ]).unsqueeze(0)  # [1, 2]
        
        # Reshape state for MOEA
        # State contains: inventory (3), backlog (3), pipeline (3), demand (1), avg_demand (1)
        n_echelons = self.config.n_echelons
        reshaped_state = np.zeros((1, n_echelons, 4))  # [time_steps=1, echelons, features]
        
        # Extract features
        inventory = state[:n_echelons]  # First n_echelons elements are inventory
        backlog = state[n_echelons:2*n_echelons]  # Next n_echelons elements are backlog
        demand = state[-2]  # Second to last element is current demand
        orders = state[2*n_echelons:3*n_echelons]  # Third n_echelons elements are pipeline/orders
        
        # Fill reshaped state
        reshaped_state[0, :, 0] = inventory  # Inventory levels
        reshaped_state[0, :, 1] = backlog    # Backlog levels
        reshaped_state[0, :, 2] = demand     # Demand (broadcast to all echelons)
        reshaped_state[0, :, 3] = orders     # Orders
        
        # Get optimized parameters from MOEA
        solutions = self.moea.optimize(reshaped_state)  # Run optimization with reshaped state
        best_solution = min(solutions, key=lambda x: sum(x[1]))[0]  # Get best solution
        moea_tensor = torch.FloatTensor([
            best_solution['reorder_point'],
            best_solution['order_quantity'],
            best_solution['safety_stock']
        ]).unsqueeze(0)  # [1, 3]
        
        return {
            'isn_state': isn_state,
            'demand_pred': demand_pred,
            'uncertainty': uncertainty,
            'fuzzy_rec': fuzzy_rec,
            'moea_params': moea_tensor
        }
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select action based on current state.
        
        Args:
            state: Current environment state
            
        Returns:
            Selected action
        """
        processed = self.process_state(state)
        
        # Get action from actor-critic
        action, info = self.actor_critic.select_action(
            torch.cat([processed['demand_pred'], processed['uncertainty']], dim=-1),  # [1, 2]
            processed['fuzzy_rec'],  # [1, 2]
            processed['moea_params']  # [1, 3]
        )
        
        # Store info for later use during training
        self.last_action_info = info
        
        # Action is already a numpy array
        return action
    
    def update(self, state: np.ndarray, action: np.ndarray,
              reward: float, next_state: np.ndarray, done: bool):
        """
        Update all learnable components.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Process states
        curr_processed = self.process_state(state)
        next_processed = self.process_state(next_state)
        
        # Combine states
        curr_combined = torch.cat([
            curr_processed['isn_state'],
            torch.cat([curr_processed['demand_pred'], curr_processed['uncertainty']], dim=-1),
            curr_processed['fuzzy_rec'],
            curr_processed['moea_params']
        ])
        next_combined = torch.cat([
            next_processed['isn_state'],
            torch.cat([next_processed['demand_pred'], next_processed['uncertainty']], dim=-1),
            next_processed['fuzzy_rec'],
            next_processed['moea_params']
        ])
        
        # Update actor-critic
        self.actor_critic.update(
            curr_combined, action, reward, next_combined, done
        )
        
        # Update ISN and transformer
        self.optimizer.zero_grad()
        
        # ISN loss (reconstruction)
        isn_loss = self.isn.compute_loss(
            torch.FloatTensor(state).unsqueeze(0)
        )
        
        # Transformer loss (prediction)
        demand_history = state[:self.config.demand_history_len]
        demand_history = demand_history.reshape(-1, 1)
        transformer_loss = self.transformer.compute_loss(
            torch.FloatTensor(demand_history).unsqueeze(0)
        )
        
        # Combined loss
        total_loss = isn_loss + transformer_loss
        total_loss.backward()
        self.optimizer.step()
    
    def train(self, n_episodes: int = 1000):
        """
        Train the system for a specified number of episodes.
        
        Args:
            n_episodes: Number of episodes to train for
        """
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(self.config.episode_length):
                # Select and take action
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                # Update all components
                self.update(state, action, reward, next_state, done or truncated)
                
                episode_reward += reward
                state = next_state
                
                if done or truncated:
                    break
            
            # Log progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    # Initialize and train system
    config = SystemConfig()
    system = SupplyChainSystem(config)
    system.train()
