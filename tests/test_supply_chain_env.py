"""Tests for the supply chain environment."""

import numpy as np
import pytest
from src.environment.supply_chain_env import SupplyChainEnv


@pytest.fixture
def env():
    """Create a test environment."""
    return SupplyChainEnv(
        num_echelons=3,
        max_steps=50,
        demand_mean=100,
        demand_std=10,
        seed=42
    )


def test_env_initialization(env):
    """Test environment initialization."""
    assert env.num_echelons == 3
    assert env.max_steps == 50
    assert env.demand_mean == 100
    assert env.demand_std == 10
    
    # Check spaces
    assert env.action_space.shape == (3,)
    assert env.observation_space.shape == (11,)  # 3 (inventory) + 3 (backlog) + 3 (pipeline) + 1 (demand) + 1 (avg demand)


def test_reset(env):
    """Test environment reset."""
    obs, _ = env.reset(seed=42)
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert env.current_step == 0
    assert env.total_demand == 0
    assert env.fulfilled_demand == 0
    assert np.all(env.inventory_levels == 0)
    assert np.all(env.backlog_levels == 0)


def test_step(env):
    """Test environment step."""
    env.reset(seed=42)
    
    # Take a step with zero orders
    action = np.zeros(env.num_echelons)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check return types
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # Check shapes
    assert obs.shape == env.observation_space.shape
    assert action.shape == env.action_space.shape
    
    # Check info dict contains required keys
    required_keys = {
        'inventory_levels',
        'backlog_levels',
        'pipeline_inventory',
        'service_level',
        'holding_costs',
        'backlog_costs',
        'transportation_costs',
        'service_level_reward',
        'total_reward'
    }
    assert all(key in info for key in required_keys)


def test_demand_generation(env):
    """Test demand generation."""
    demands = []
    for _ in range(1000):
        demand = env._generate_demand()
        demands.append(demand)
        
        # Demand should never be negative
        assert demand >= 0
    
    # Check demand statistics
    mean_demand = np.mean(demands)
    std_demand = np.std(demands)
    
    # Allow for some statistical variation
    assert abs(mean_demand - env.demand_mean) < env.demand_std
    assert abs(std_demand - env.demand_std) < env.demand_std / 2


def test_service_level(env):
    """Test service level calculation."""
    env.reset(seed=42)
    
    # Initially should be 1.0 (perfect service level)
    assert env._calculate_service_level() == 1.0
    
    # After some demand but no fulfillment
    env.total_demand = 100
    env.fulfilled_demand = 0
    assert env._calculate_service_level() == 0.0
    
    # After partial fulfillment
    env.fulfilled_demand = 50
    assert env._calculate_service_level() == 0.5


def test_reward_calculation(env):
    """Test reward calculation."""
    # Test with some inventory and backlog
    inventory = np.array([10, 20, 30])
    backlog = np.array([5, 10, 15])
    orders = np.array([50, 100, 150])
    
    reward, breakdown = env._calculate_rewards(inventory, backlog, orders)
    
    # Check reward components
    assert 'holding_costs' in breakdown
    assert 'backlog_costs' in breakdown
    assert 'transportation_costs' in breakdown
    assert 'service_level_reward' in breakdown
    assert 'total_reward' in breakdown
    
    # Rewards should be negative (costs)
    assert breakdown['holding_costs'] <= 0
    assert breakdown['backlog_costs'] <= 0
    assert breakdown['transportation_costs'] <= 0


def test_episode_length(env):
    """Test that episodes terminate after max_steps."""
    env.reset(seed=42)
    
    for _ in range(env.max_steps - 1):
        action = np.zeros(env.num_echelons)
        _, _, terminated, truncated, _ = env.step(action)
        assert not terminated
        assert not truncated
    
    # Last step should truncate
    _, _, terminated, truncated, _ = env.step(np.zeros(env.num_echelons))
    assert not terminated
    assert truncated
