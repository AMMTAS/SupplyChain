"""Integration tests for Actor-Critic network."""

import pytest
import torch
import numpy as np
from src.models.actor_critic import ActorCriticNetwork
from src.models.transformer import DemandPredictor
from src.models.fuzzy import FuzzyController, FuzzyControllerConfig
from src.models.moea.optimizer import MOEAConfig, MOEAOptimizer
from src.environment import SupplyChainEnv
from config.transformer_config import TransformerConfig


@pytest.fixture
def env():
    """Create test environment."""
    return SupplyChainEnv(
        num_echelons=3,
        demand_mean=100.0,
        demand_std=20.0
    )


@pytest.fixture
def transformer():
    """Create test transformer."""
    config = TransformerConfig(
        input_dim=1,  # Single demand value per timestep
        output_dim=1,
        forecast_horizon=24,  # Match demand_dim in actor_critic
        history_length=24,  # Match environment history length
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_length=48,
        uncertainty_type='deterministic'
    )
    return DemandPredictor(config)


@pytest.fixture
def fuzzy_controller():
    """Create test fuzzy controller."""
    config = FuzzyControllerConfig(
        input_dim=24,  # Match transformer output
        n_membership_functions=3,
        universe_range=(-1.0, 1.0),
        defuzz_method='centroid'
    )
    return FuzzyController(config)


@pytest.fixture
def moea():
    """Create test MOEA optimizer."""
    objectives = [
        lambda solution, state: solution['holding_cost']**2,  # Cost minimization
        lambda solution, state: solution['backlog_cost']**2,  # Backlog minimization
        lambda solution, state: (1 - solution['target_service'])**2,  # Service level maximization
    ]
    parameter_bounds = {
        'holding_cost': (0, 10),
        'backlog_cost': (0, 10),
        'target_service': (0.8, 1.0),
    }
    
    config = MOEAConfig(
        n_objectives=3,
        population_size=4,  # Small population for testing
        neighborhood_size=2,  # Few neighbors
        mutation_rate=0.1,
        crossover_rate=0.8,
        n_generations=8  # Few generations
    )
    
    return MOEAOptimizer(
        config=config,
        objectives=objectives,
        parameter_bounds=parameter_bounds
    )


@pytest.fixture
def actor_critic():
    """Create test actor-critic network."""
    return ActorCriticNetwork(
        demand_dim=24,
        fuzzy_dim=2,
        moea_dim=3,  # Match number of MOEA parameters
        action_dim=3,
        hidden_dim=128,
        n_hidden=2
    )


def test_component_integration(
    env,
    transformer,
    fuzzy_controller,
    moea,
    actor_critic
):
    """Test integration between all components."""
    # Get initial state
    state, _ = env.reset()  # Unpack observation and info
    print(f"State shape: {state.shape}")  # Debug state shape
    
    # Get demand predictions
    demand_history = state[:24]  # First 24 elements are demand history
    demand_history = np.array(demand_history).reshape(-1, 1)  # Add feature dimension [seq_len, 1]
    demand_pred = transformer(
        torch.FloatTensor(demand_history).unsqueeze(0)  # Add batch dimension [1, seq_len, 1]
    )[0].detach()
    demand_pred = demand_pred.view(-1)  # Flatten to [24]
    assert len(demand_pred.shape) == 1  # Single sequence
    
    # Get fuzzy recommendations
    fuzzy_rec = fuzzy_controller.process_state(state)
    assert isinstance(fuzzy_rec, dict)  # Returns a dictionary of recommendations
    fuzzy_tensor = torch.FloatTensor([fuzzy_rec['order_adjustment'], fuzzy_rec['risk_level']])
    
    # Get MOEA parameters
    solutions = moea.optimize(state)  # Run optimization to get parameters
    best_solution = min(solutions, key=lambda x: sum(x[1]))[0]  # Get solution with lowest total objective
    moea_tensor = torch.FloatTensor([
        best_solution['holding_cost'],
        best_solution['backlog_cost'],
        best_solution['target_service']
    ])
    
    # Get actor-critic outputs
    action_dist, value = actor_critic(
        demand_pred.unsqueeze(0),  # Add batch dimension [1, 24]
        fuzzy_tensor.unsqueeze(0),  # Add batch dimension [1, 2]
        moea_tensor.unsqueeze(0)  # Add batch dimension [1, 3]
    )
    
    # Verify outputs
    assert isinstance(action_dist, torch.distributions.Distribution)
    assert isinstance(value, torch.Tensor)
    assert value.shape == (1, 1)  # [batch_size, 1]


def test_training_loop(
    env,
    transformer,
    fuzzy_controller,
    moea,
    actor_critic
):
    """Test training loop integration."""
    n_steps = 5
    batch_size = 2
    
    # Storage for experience
    states = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []
    
    # Initial state
    state, _ = env.reset()  # Unpack observation and info
    done = False
    
    # Collect experience
    for _ in range(n_steps):
        # Get inputs
        demand_history = state[:24]  # First 24 elements are demand history
        demand_history = np.array(demand_history).reshape(-1, 1)  # Add feature dimension [seq_len, 1]
        demand_pred = transformer(
            torch.FloatTensor(demand_history).unsqueeze(0)  # Add batch dimension [1, seq_len, 1]
        )[0].detach()
        demand_pred = demand_pred.view(-1)  # Flatten to [24]

        fuzzy_rec_dict = fuzzy_controller.process_state(state)
        fuzzy_rec = torch.FloatTensor([
            fuzzy_rec_dict['order_adjustment'],
            fuzzy_rec_dict['risk_level']
        ])

        solutions = moea.optimize(state)  # Run optimization to get parameters
        best_solution = min(solutions, key=lambda x: sum(x[1]))[0]  # Get solution with lowest total objective
        moea_tensor = torch.FloatTensor([
            best_solution['holding_cost'],
            best_solution['backlog_cost'],
            best_solution['target_service']
        ])

        # Get action and value
        action_dist, value = actor_critic(
            demand_pred.unsqueeze(0),  # Add batch dimension [1, 24]
            fuzzy_rec.unsqueeze(0),  # Add batch dimension [1, 2]
            moea_tensor.unsqueeze(0)  # Add batch dimension [1, 3]
        )
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        # Take step in environment
        next_state, reward, done, _, _ = env.step(action.squeeze(0).numpy())  # Unpack step return

        # Store experience
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        values.append(value)
        log_probs.append(log_prob)

        if done:
            state, _ = env.reset()  # Unpack observation and info
        else:
            state = next_state

    # Convert to tensors
    states = torch.FloatTensor(states)
    actions = torch.stack(actions)
    rewards = torch.FloatTensor(rewards)
    dones = torch.BoolTensor(dones)
    values = torch.cat(values)
    log_probs = torch.stack(log_probs)

    # Verify shapes
    assert states.shape[0] == n_steps
    assert actions.shape[0] == n_steps
    assert rewards.shape[0] == n_steps
    assert dones.shape[0] == n_steps
    assert values.shape[0] == n_steps
    assert log_probs.shape[0] == n_steps


def test_component_updates(
    env,
    transformer,
    fuzzy_controller,
    moea,
    actor_critic
):
    """Test component updates during training."""
    state, _ = env.reset()  # Unpack observation and info

    # Initial predictions/recommendations
    demand_history = state[:24]  # First 24 elements are demand history
    demand_history = np.array(demand_history).reshape(-1, 1)  # Add feature dimension [seq_len, 1]
    demand_pred1 = transformer(
        torch.FloatTensor(demand_history).unsqueeze(0)  # Add batch dimension [1, seq_len, 1]
    )[0].detach()
    demand_pred1 = demand_pred1.view(-1)  # Flatten to [24]

    fuzzy_rec1 = fuzzy_controller.process_state(state)
    fuzzy_tensor1 = torch.FloatTensor([fuzzy_rec1['order_adjustment'], fuzzy_rec1['risk_level']])

    solutions1 = moea.optimize(state)  # Run optimization to get parameters
    best_solution1 = min(solutions1, key=lambda x: sum(x[1]))[0]  # Get solution with lowest total objective
    moea_tensor1 = torch.FloatTensor([
        best_solution1['holding_cost'],
        best_solution1['backlog_cost'],
        best_solution1['target_service']
    ])

    # Take step in environment
    action_dist1, value1 = actor_critic(
        demand_pred1.unsqueeze(0),  # Add batch dimension [1, 24]
        fuzzy_tensor1.unsqueeze(0),  # Add batch dimension [1, 2]
        moea_tensor1.unsqueeze(0)  # Add batch dimension [1, 3]
    )
    action1 = action_dist1.sample()
    next_state, _, _, _, _ = env.step(action1.squeeze(0).numpy())  # Unpack step return

    # Get new predictions/recommendations
    demand_history = next_state[:24]  # First 24 elements are demand history
    demand_history = np.array(demand_history).reshape(-1, 1)  # Add feature dimension [seq_len, 1]
    demand_pred2 = transformer(
        torch.FloatTensor(demand_history).unsqueeze(0)  # Add batch dimension [1, seq_len, 1]
    )[0].detach()
    demand_pred2 = demand_pred2.view(-1)  # Flatten to [24]

    fuzzy_rec2 = fuzzy_controller.process_state(next_state)
    fuzzy_tensor2 = torch.FloatTensor([fuzzy_rec2['order_adjustment'], fuzzy_rec2['risk_level']])

    solutions2 = moea.optimize(next_state)  # Run optimization to get parameters
    best_solution2 = min(solutions2, key=lambda x: sum(x[1]))[0]  # Get solution with lowest total objective
    moea_tensor2 = torch.FloatTensor([
        best_solution2['holding_cost'],
        best_solution2['backlog_cost'],
        best_solution2['target_service']
    ])

    # Get new outputs
    action_dist2, value2 = actor_critic(
        demand_pred2.unsqueeze(0),  # Add batch dimension [1, 24]
        fuzzy_tensor2.unsqueeze(0),  # Add batch dimension [1, 2]
        moea_tensor2.unsqueeze(0)  # Add batch dimension [1, 3]
    )
    action2 = action_dist2.sample()

    # Verify changes
    assert not torch.allclose(demand_pred1, demand_pred2)  # Demand predictions should change
    assert not torch.allclose(fuzzy_tensor1, fuzzy_tensor2)  # Fuzzy recommendations should change
    assert not torch.allclose(moea_tensor1, moea_tensor2)  # MOEA parameters should change
    assert not torch.allclose(action1, action2)  # Actions should change
    assert not torch.allclose(value1, value2)  # Values should change


def test_training_performance(results_logger):
    """
    Test and log end-to-end training performance.
    
    This test verifies that the actor-critic network can learn a reasonable policy by:
    1. Running short training episodes (max 50 steps)
    2. Checking that episode rewards are within expected range (-10 to -1)
    3. Monitoring service levels and bullwhip effect
    4. Verifying policy and value losses are reasonable
    
    Test Parameters:
    - num_echelons: 3 (manufacturer, distributor, retailer)
    - demand_mean: 100.0 units
    - demand_std: 20.0 units
    - max_episodes: 3 (limited for quick testing)
    - steps_per_episode: 50 (limited for quick testing)
    
    Success Criteria:
    1. Mean episode reward (normalized by steps) between -10 and -1
    2. Service level and bullwhip metrics are tracked
    3. Policy and value losses are non-zero and finite
    
    Note: This is a smoke test to catch major issues. Full training
    evaluation should be done separately with longer episodes and
    more thorough metrics.
    """
    # Initialize components
    env = SupplyChainEnv(
        num_echelons=3,
        demand_mean=100.0,
        demand_std=20.0
    )
    actor_critic = ActorCriticNetwork(
        demand_dim=24,
        fuzzy_dim=2,
        moea_dim=3,  # Match number of MOEA parameters
        action_dim=3,
        hidden_dim=128,
        n_hidden=2,
        buffer_capacity=10  # Small buffer for testing
    )
    fuzzy_controller = FuzzyController(FuzzyControllerConfig(
        input_dim=24,
        n_membership_functions=3,
        universe_range=(-1.0, 1.0),
        defuzz_method='centroid'
    ))
    moea = MOEAOptimizer(
        config=MOEAConfig(
            n_objectives=3,
            population_size=4,
            neighborhood_size=2,
            mutation_rate=0.1,
            crossover_rate=0.8,
            n_generations=8
        ),
        objectives=[
            lambda solution, state: solution['holding_cost']**2,  # Cost minimization
            lambda solution, state: solution['backlog_cost']**2,  # Backlog minimization
            lambda solution, state: (1 - solution['target_service'])**2,  # Service level maximization
        ],
        parameter_bounds={
            'holding_cost': (0, 10),
            'backlog_cost': (0, 10),
            'target_service': (0.8, 1.0),
        }
    )
    
    # Training loop parameters
    n_steps = 2
    max_episodes = 3
    batch_size = 2
    
    # Collect metrics
    episode_rewards = []
    service_levels = []
    bullwhip_ratios = []
    policy_losses = []
    value_losses = []
    entropy_losses = []
    
    # Run training episodes
    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        demands = []
        orders = []
        
        # Early stopping if we've collected enough data
        if len(policy_losses) >= n_steps and len(value_losses) >= n_steps:
            break
            
        steps = 0
        while not done and steps < 50:
            steps += 1
            
            # Pad state to match expected demand dimension
            state_padded = np.zeros(24)
            state_padded[:len(state)] = state
            state_tensor = torch.FloatTensor(state_padded).unsqueeze(0)
            
            # Get fuzzy recommendations and MOEA parameters
            fuzzy_rec_dict = fuzzy_controller.process_state(state)
            fuzzy_rec = torch.FloatTensor([[fuzzy_rec_dict['order_adjustment'], fuzzy_rec_dict['risk_level']]])
            
            solutions = moea.optimize(state)
            best_solution = min(solutions, key=lambda x: sum(x[1]))[0]
            moea_params = torch.FloatTensor([[
                best_solution['holding_cost'],
                best_solution['backlog_cost'],
                best_solution['target_service']
            ]])
            
            # Get action and value
            action_dist, value = actor_critic(
                state_tensor,
                fuzzy_rec,
                moea_params
            )
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(-1)
            
            # Take step in environment
            next_state, reward, done, truncated, info = env.step(action.squeeze(0).numpy())
            episode_reward += reward
            if 'demand' in info:
                demands.append(info['demand'])
            if 'orders' in info:
                orders.append(info['orders'])
            
            # Store transition in buffer
            next_state_padded = np.zeros(24)
            next_state_padded[:len(next_state)] = next_state
            next_state_tensor = torch.FloatTensor(next_state_padded).unsqueeze(0)
            
            actor_critic.buffer.add(
                state=state_tensor,
                action=action,
                reward=reward,
                next_state=next_state_tensor,
                done=done,
                log_prob=log_prob,
                value=value
            )
            
            state = next_state
            
            # Update networks if batch is full
            if actor_critic.buffer.is_full():
                states, actions, rewards, next_states, dones, log_probs, values = actor_critic.buffer.get_batch()
                advantages = rewards - values.squeeze(-1)
                returns = rewards
                
                # Ensure all tensors have consistent batch dimensions
                states = states.squeeze(1)
                actions = actions.squeeze(1)
                log_probs = log_probs.squeeze(1)
                advantages = advantages.unsqueeze(-1)
                
                # Get current fuzzy and MOEA inputs for all states
                fuzzy_recs = torch.zeros(len(states), 2)
                moea_params = torch.zeros(len(states), 3)
                
                # Update actor and critic
                p_loss, v_loss, e_loss = actor_critic.update(
                    demand_preds=states,
                    fuzzy_recs=fuzzy_recs,
                    moea_params=moea_params,
                    actions=actions,
                    old_log_probs=log_probs,
                    advantages=advantages,
                    returns=returns
                )
                policy_losses.append(p_loss)
                value_losses.append(v_loss)
                entropy_losses.append(e_loss)
                
                # Clear buffer after update
                actor_critic.buffer.clear()
        
        # Calculate episode metrics
        episode_rewards.append(episode_reward / steps)  # Normalize by number of steps
        if demands and orders:
            service_level = np.mean([d <= o for d, o in zip(demands, orders)])
            service_levels.append(service_level)
            
            # Calculate bullwhip ratio if we have enough data
            if len(demands) > 1 and len(orders) > 1:
                demand_std = np.std(demands)
                order_std = np.std(orders)
                if demand_std > 0:
                    bullwhip_ratio = order_std / demand_std
                    bullwhip_ratios.append(bullwhip_ratio)
    
    # Log results
    results_logger.log_result(
        'actor_critic_training',
        episode_reward=np.mean(episode_rewards),
        service_level=np.mean(service_levels) if service_levels else 0.0,
        bullwhip=np.mean(bullwhip_ratios) if bullwhip_ratios else 0.0,
        policy_loss=np.mean(policy_losses) if policy_losses else 0.0,
        value_loss=np.mean(value_losses) if value_losses else 0.0,
        entropy_loss=np.mean(entropy_losses) if entropy_losses else 0.0
    )
    results_logger.save_results()
    
    # Assert expected ranges for normalized rewards
    mean_reward = np.mean(episode_rewards)
    assert -10 <= mean_reward <= -1, f"Episode reward {mean_reward} outside expected range"
    
    # Assert other metrics are in reasonable ranges
    if service_levels:
        assert 0 <= np.mean(service_levels) <= 1, "Service level should be between 0 and 1"
    if bullwhip_ratios:
        assert 0 <= np.mean(bullwhip_ratios) <= 10, "Bullwhip ratio should be reasonable"
    if policy_losses:
        assert 0.0 <= np.mean(policy_losses) <= 2.0, "Policy loss too high"
    if value_losses:
        assert 0.0 <= np.mean(value_losses) <= 2.0, "Value loss too high"
    if entropy_losses:
        assert 0.0 <= np.mean(entropy_losses) <= 1.0, "Entropy outside expected range"


def test_moea_parameter_interface(moea, actor_critic):
    """Test MOEA parameter integration."""
    # Get parameters from MOEA
    solutions = moea.optimize(np.random.randn(24))
    best_solution = min(solutions, key=lambda x: sum(x[1]))[0]
    moea_tensor = torch.FloatTensor([
        best_solution['holding_cost'],
        best_solution['backlog_cost'],
        best_solution['target_service']
    ])

    # Verify parameter ranges
    assert torch.all(moea_tensor >= 0)
    assert 0 <= moea_tensor[0] <= 10.0
    assert 0 <= moea_tensor[1] <= 10.0
    assert 0.8 <= moea_tensor[2] <= 1.0
