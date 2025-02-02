"""Actor-Critic network implementation."""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .policy import OrderingPolicy
from .value import ValueNetwork


class ExperienceBuffer:
    """Buffer for storing experience transitions."""
    
    def __init__(self, capacity: int = 1000):
        """Initialize buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        log_prob: torch.Tensor,
        value: torch.Tensor
    ):
        """Add transition to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        if len(self.states) > self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.log_probs.pop(0)
            self.values.pop(0)
    
    def clear(self):
        """Clear buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.states) >= self.capacity
    
    def get_batch(self) -> Tuple[torch.Tensor, ...]:
        """Get batch of transitions."""
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        next_states = torch.stack(self.next_states)
        dones = torch.BoolTensor(self.dones)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)
        
        return states, actions, rewards, next_states, dones, log_probs, values


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for supply chain optimization.
    
    Takes input from:
    1. Demand predictions (from Transformer)
    2. Rule-based recommendations (from Fuzzy Controller)
    3. Optimized parameters (from MOEA)
    
    Outputs:
    1. Order quantity policies (Actor)
    2. Value function estimates (Critic)
    """
    
    def __init__(
        self,
        demand_dim: int,
        fuzzy_dim: int,
        moea_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        buffer_capacity: int = 1000
    ):
        """Initialize Actor-Critic network.
        
        Args:
            demand_dim: Dimension of demand predictions
            fuzzy_dim: Dimension of fuzzy controller output
            moea_dim: Dimension of MOEA parameters
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            n_hidden: Number of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clip ratio
            target_kl: Target KL divergence
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            buffer_capacity: Maximum size of experience buffer
        """
        super().__init__()
        
        self.demand_dim = demand_dim
        self.fuzzy_dim = fuzzy_dim
        self.moea_dim = moea_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Feature extraction for different inputs
        self.demand_encoder = nn.Sequential(
            nn.Linear(demand_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fuzzy_encoder = nn.Sequential(
            nn.Linear(fuzzy_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.moea_encoder = nn.Sequential(
            nn.Linear(moea_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combine all features
        self.feature_combiner = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) network
        self.actor = OrderingPolicy(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            min_std=0.1,
            max_std=0.3  # Conservative exploration
        )
        
        # Critic (value) network
        self.critic = ValueNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Initialize experience buffer
        self.buffer = ExperienceBuffer(capacity=buffer_capacity)
    
    def forward(
        self,
        demand_pred: torch.Tensor,
        fuzzy_rec: torch.Tensor,
        moea_params: torch.Tensor
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            demand_pred: Demand prediction tensor [batch_size, seq_len] or [batch_size, 1, seq_len]
            fuzzy_rec: Fuzzy controller recommendations [batch_size, fuzzy_dim]
            moea_params: MOEA parameters [batch_size, moea_dim]
            
        Returns:
            Tuple of (action distribution, value estimate)
        """
        # Ensure demand_pred is 2D [batch_size, seq_len]
        if len(demand_pred.shape) == 3:
            demand_pred = demand_pred.squeeze(1)
        
        # Encode different inputs
        demand_features = self.demand_encoder(demand_pred)  # [batch_size, hidden_dim]
        fuzzy_features = self.fuzzy_encoder(fuzzy_rec)     # [batch_size, hidden_dim]
        moea_features = self.moea_encoder(moea_params)     # [batch_size, hidden_dim]
        
        # Combine features
        combined = torch.cat([
            demand_features,
            fuzzy_features,
            moea_features
        ], dim=-1)  # [batch_size, 3*hidden_dim]
        features = self.feature_combiner(combined)  # [batch_size, hidden_dim]
        
        # Get policy distribution and value estimate
        action_dist = self.actor(features)  # Returns Normal distribution
        value = self.critic(features)  # Returns value estimate

        return action_dist, value
    
    def select_action(
        self,
        demand_pred: np.ndarray,
        fuzzy_rec: np.ndarray,
        moea_params: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """Select action based on current policy.
        
        Args:
            demand_pred: Demand predictions
            fuzzy_rec: Fuzzy controller recommendations
            moea_params: MOEA parameters
            deterministic: If True, use mean action instead of sampling
            
        Returns:
            Tuple of (selected action, info dictionary)
        """
        # Convert inputs to tensors
        demand_pred = torch.FloatTensor(demand_pred)
        fuzzy_rec = torch.FloatTensor(fuzzy_rec)
        moea_params = torch.FloatTensor(moea_params)
        
        with torch.no_grad():
            # Get action distribution and value
            action_dist, value = self.forward(demand_pred, fuzzy_rec, moea_params)
            
            # Select action
            if deterministic:
                action = action_dist.mean
            else:
                action = action_dist.sample()
            
            # Get log probability
            log_prob = action_dist.log_prob(action)
            
            # Get entropy
            entropy = action_dist.entropy().mean()
        
        # Convert to numpy and squeeze batch dimension
        action = action.squeeze(0).numpy()  # Remove batch dimension
        value = value.squeeze(0).numpy()  # Remove batch dimension
        log_prob = log_prob.squeeze(0).numpy()  # Remove batch dimension
        entropy = entropy.numpy()
        
        info = {
            'value': value,
            'log_prob': log_prob,
            'entropy': entropy
        }
        
        return action, info
    
    def update(
        self,
        demand_preds: torch.Tensor,
        fuzzy_recs: torch.Tensor,
        moea_params: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """Update policy and value networks.
        
        Returns:
            Dictionary containing loss metrics
        """
        # Get current policy distribution and value estimate
        action_dist, values = self.forward(demand_preds, fuzzy_recs, moea_params)
        
        # Calculate log probabilities and entropy
        log_probs = action_dist.log_prob(actions).sum(dim=-1)  # Sum over action dimensions
        entropy = action_dist.entropy().mean()
        
        # Calculate ratios and clipped ratios
        ratios = torch.exp(log_probs - old_log_probs.sum(dim=-1))  # Sum over action dimensions
        clipped_ratios = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        # Calculate policy loss components
        policy_loss_1 = -ratios * advantages
        policy_loss_2 = -clipped_ratios * advantages
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
        
        # Calculate value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Calculate total loss
        total_loss = (
            policy_loss +
            self.value_coef * value_loss -
            self.entropy_coef * entropy
        )
        
        # Calculate approximate KL divergence
        with torch.no_grad():
            kl = ((ratios - 1) - (log_probs - old_log_probs.sum(dim=-1))).mean().item()
            clip_fraction = (abs(ratios - 1) > self.clip_ratio).float().mean().item()
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl': kl,
            'clip_fraction': clip_fraction,
            'early_stop': kl > self.target_kl
        }
        
        return metrics

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_values: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO loss function.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            old_values: Value estimates from old policy
            old_log_probs: Log probabilities from old policy
            rewards: Batch of rewards received
            dones: Batch of done flags
            
        Returns:
            Dictionary containing loss components
        """
        # Compute returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        last_return = 0.0
        
        # Compute GAE and returns in reverse order
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = old_values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            # Compute TD error and GAE
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - old_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            
            # Compute returns
            last_return = rewards[t] + self.gamma * next_non_terminal * last_return
            returns[t] = last_return
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass through network
        # Split state into components (assuming state format matches training)
        demand_preds = states[:, :24]  # First 24 elements are demand predictions
        fuzzy_recs = states[:, 24:26]  # Next 2 elements are fuzzy recommendations
        moea_params = states[:, 26:29]  # Last 3 elements are MOEA parameters
        
        # Get current policy distribution and values
        action_dist, values = self(demand_preds, fuzzy_recs, moea_params)
        
        # Compute log probabilities and entropy
        log_probs = action_dist.log_prob(actions)  # Shape: [batch_size, action_dim]
        log_probs = log_probs.sum(dim=-1)  # Sum log probs across action dimensions
        entropy = action_dist.entropy().mean()
        
        # Compute policy loss (PPO clip objective)
        ratio = torch.exp(log_probs - old_log_probs)
        clip_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = -torch.min(
            ratio * advantages,
            clip_ratio * advantages
        ).mean()
        
        # Compute value loss (make sure shapes match)
        values = values.squeeze(-1)  # Remove last dimension to match returns shape
        value_loss = F.mse_loss(values, returns)
        
        # Compute total loss
        total_loss = (
            policy_loss +
            self.value_coef * value_loss -
            self.entropy_coef * entropy
        )
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy
        }

    def _check_early_stopping(self, metrics):
        """Check if training should stop early."""
        if len(self.loss_history) < self.patience:
            return False

        recent_losses = self.loss_history[-self.patience:]
        min_improvement = 0.001  # Minimum required improvement

        # Check if loss hasn't improved significantly
        return all(
            abs(metrics['policy_loss'] - prev_loss) < min_improvement
            for prev_loss in recent_losses
        )
