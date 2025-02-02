"""Training script for Actor-Critic network."""

import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Dict, List, Tuple

from src.models.actor_critic import ActorCriticNetwork
from src.models.transformer import DemandPredictor
from src.models.fuzzy import FuzzyController
from src.models.moea import MOEAOptimizer
from src.environment import SupplyChainEnv
from src.utils.logger import setup_logger


def collect_rollout(
    env: SupplyChainEnv,
    actor_critic: ActorCriticNetwork,
    transformer: DemandPredictor,
    fuzzy_controller: FuzzyController,
    moea: MOEAOptimizer,
    n_steps: int = 2048
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Collect rollout data using current policy.
    
    Args:
        env: Supply chain environment
        actor_critic: Actor-Critic network
        transformer: Demand predictor
        fuzzy_controller: Fuzzy logic controller
        moea: MOEA optimizer
        n_steps: Number of steps to collect
        
    Returns:
        Tuple of (rollout data, episode statistics)
    """
    # Initialize buffers
    demand_preds = []
    fuzzy_recs = []
    moea_params = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    
    # Initial state
    done = False
    
    for step in range(n_steps):
        # Get demand predictions
        demand_pred = transformer.predict(env.get_demand_history())
        
        # Get fuzzy recommendations
        fuzzy_rec = fuzzy_controller.evaluate(env.get_state())
        
        # Get MOEA parameters
        moea_params_step = moea.get_parameters()
        
        # Select action
        action, info = actor_critic.select_action(
            demand_pred,
            fuzzy_rec,
            moea_params_step
        )
        
        # Take step in environment
        next_state, reward, done, _ = env.step(action)
        
        # Store data
        demand_preds.append(demand_pred)
        fuzzy_recs.append(fuzzy_rec)
        moea_params.append(moea_params_step)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(info['log_prob'])
        values.append(info['value'])
        dones.append(done)
        
        # Update episode stats
        current_episode_reward += reward
        current_episode_length += 1
        
        if done:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            current_episode_reward = 0
            current_episode_length = 0
            env.reset()
    
    # Convert to numpy arrays
    rollout_data = {
        'demand_preds': np.array(demand_preds),
        'fuzzy_recs': np.array(fuzzy_recs),
        'moea_params': np.array(moea_params),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'log_probs': np.array(log_probs),
        'values': np.array(values),
        'dones': np.array(dones)
    }
    
    # Calculate statistics
    stats = {
        'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
        'mean_length': np.mean(episode_lengths) if episode_lengths else 0,
        'num_episodes': len(episode_rewards)
    }
    
    return rollout_data, stats


def compute_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute advantages using GAE.
    
    Args:
        rewards: Array of rewards
        values: Array of value estimates
        dones: Array of done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    last_gae = 0
    last_return = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
            next_done = 1
        else:
            next_value = values[t + 1]
            next_done = dones[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
        last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
        advantages[t] = last_gae
        
        last_return = rewards[t] + gamma * last_return * (1 - next_done)
        returns[t] = last_return
    
    return advantages, returns


def train(
    env: SupplyChainEnv,
    actor_critic: ActorCriticNetwork,
    transformer: DemandPredictor,
    fuzzy_controller: FuzzyController,
    moea: MOEAOptimizer,
    n_epochs: int = 1000,
    n_steps_per_epoch: int = 2048,
    n_updates_per_epoch: int = 10,
    batch_size: int = 64,
    save_dir: str = 'checkpoints',
    log_dir: str = 'logs'
):
    """Train Actor-Critic network.
    
    Args:
        env: Supply chain environment
        actor_critic: Actor-Critic network
        transformer: Demand predictor
        fuzzy_controller: Fuzzy logic controller
        moea: MOEA optimizer
        n_epochs: Number of epochs to train
        n_steps_per_epoch: Number of steps per epoch
        n_updates_per_epoch: Number of policy updates per epoch
        batch_size: Batch size for updates
        save_dir: Directory to save checkpoints
        log_dir: Directory to save logs
    """
    # Setup logging
    logger = setup_logger('actor_critic_training')
    writer = SummaryWriter(log_dir)
    
    # Create checkpoint directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    total_steps = 0
    
    for epoch in range(n_epochs):
        logger.info(f'Starting epoch {epoch}')
        
        # Collect rollout
        rollout_data, stats = collect_rollout(
            env,
            actor_critic,
            transformer,
            fuzzy_controller,
            moea,
            n_steps=n_steps_per_epoch
        )
        
        # Compute advantages
        advantages, returns = compute_advantages(
            rollout_data['rewards'],
            rollout_data['values'],
            rollout_data['dones'],
            gamma=actor_critic.gamma,
            gae_lambda=actor_critic.gae_lambda
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        demand_preds = torch.FloatTensor(rollout_data['demand_preds'])
        fuzzy_recs = torch.FloatTensor(rollout_data['fuzzy_recs'])
        moea_params = torch.FloatTensor(rollout_data['moea_params'])
        actions = torch.FloatTensor(rollout_data['actions'])
        old_log_probs = torch.FloatTensor(rollout_data['log_probs'])
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Policy updates
        for update in range(n_updates_per_epoch):
            # Generate random indices
            indices = np.random.permutation(n_steps_per_epoch)
            
            # Update in batches
            for start_idx in range(0, n_steps_per_epoch, batch_size):
                # Get batch indices
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Update policy
                metrics = actor_critic.update(
                    demand_preds=demand_preds[batch_indices],
                    fuzzy_recs=fuzzy_recs[batch_indices],
                    moea_params=moea_params[batch_indices],
                    actions=actions[batch_indices],
                    old_log_probs=old_log_probs[batch_indices],
                    advantages=advantages[batch_indices],
                    returns=returns[batch_indices]
                )
                
                # Early stopping
                if metrics['early_stop']:
                    break
            
            if metrics['early_stop']:
                break
        
        # Log metrics
        total_steps += n_steps_per_epoch
        writer.add_scalar('reward/mean', stats['mean_reward'], total_steps)
        writer.add_scalar('episode_length/mean', stats['mean_length'], total_steps)
        writer.add_scalar('policy/loss', metrics['policy_loss'], total_steps)
        writer.add_scalar('value/loss', metrics['value_loss'], total_steps)
        writer.add_scalar('policy/entropy', metrics['entropy'], total_steps)
        writer.add_scalar('policy/kl', metrics['approx_kl'], total_steps)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                save_dir,
                f'actor_critic_epoch_{epoch+1}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': actor_critic.state_dict(),
                'optimizer_state_dict': actor_critic.optimizer.state_dict(),
                'stats': stats,
                'metrics': metrics
            }, checkpoint_path)
            
            logger.info(f'Saved checkpoint to {checkpoint_path}')
        
        logger.info(
            f'Epoch {epoch} - '
            f'Mean Reward: {stats["mean_reward"]:.2f}, '
            f'Mean Length: {stats["mean_length"]:.2f}, '
            f'Policy Loss: {metrics["policy_loss"]:.4f}, '
            f'Value Loss: {metrics["value_loss"]:.4f}, '
            f'Entropy: {metrics["entropy"]:.4f}, '
            f'KL: {metrics["approx_kl"]:.4f}'
        )
    
    writer.close()
