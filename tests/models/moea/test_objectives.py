"""Tests for MOEA objective functions."""

import numpy as np
import pytest
from src.models.moea.objectives import TotalCost, ServiceLevel, BullwhipEffect
from tests.test_results import TestResults
import time


@pytest.fixture
def mock_state():
    """Create mock state data for testing."""
    # Create state with shape (time_steps, echelons, features)
    state = np.zeros((10, 3, 4))  # 10 time steps, 3 echelons, 4 features
    
    # Set some test values
    state[..., 0] = 1.0  # Inventory
    state[..., 1] = 0.5  # Backlog
    state[..., 2] = 2.0  # Demand
    state[..., 3] = 2.5  # Orders
    
    return state


def test_total_cost(mock_state):
    """Test total cost objective calculation."""
    cost_obj = TotalCost(holding_cost=1.0, backlog_cost=5.0)
    solution = {}  # Not used for cost calculation
    
    cost = cost_obj(solution, mock_state)
    
    assert isinstance(cost, float)
    assert cost > 0
    
    # Test with different cost parameters
    cost_obj2 = TotalCost(holding_cost=2.0, backlog_cost=10.0)
    cost2 = cost_obj2(solution, mock_state)
    
    assert cost2 > cost  # Higher costs should result in higher objective


def test_service_level(mock_state):
    """Test service level objective calculation."""
    service_obj = ServiceLevel()
    solution = {}  # Not used for service calculation
    
    service = service_obj(solution, mock_state)
    
    assert isinstance(service, float)
    assert service <= 0  # Service level is negated for minimization
    
    # Test with full service
    full_service_state = mock_state.copy()
    full_service_state[..., 0] = 10.0  # High inventory
    full_service_state[..., 2] = 1.0   # Low demand
    
    full_service = service_obj(solution, full_service_state)
    assert full_service == -1.0  # Perfect service level


def test_bullwhip_effect(mock_state):
    """Test bullwhip effect objective calculation."""
    bullwhip_obj = BullwhipEffect(window_size=5)
    solution = {}  # Not used for bullwhip calculation
    
    bullwhip = bullwhip_obj(solution, mock_state)
    
    assert isinstance(bullwhip, float)
    assert bullwhip >= 0  # Variance ratio is non-negative
    
    # Test with no bullwhip effect
    no_bullwhip_state = mock_state.copy()
    no_bullwhip_state[..., 2] = no_bullwhip_state[..., 3]  # Orders match demand
    
    no_bullwhip = bullwhip_obj(solution, no_bullwhip_state)
    assert abs(no_bullwhip - 1.0) < 1e-6  # Should be close to 1.0


def test_zero_demand_handling(mock_state):
    """Test handling of zero demand in objectives."""
    zero_demand_state = mock_state.copy()
    zero_demand_state[..., 2] = 0.0  # Zero demand
    
    # Service level should handle zero demand
    service_obj = ServiceLevel()
    service = service_obj({}, zero_demand_state)
    assert np.isfinite(service)
    
    # Bullwhip effect should handle zero demand variance
    bullwhip_obj = BullwhipEffect()
    bullwhip = bullwhip_obj({}, zero_demand_state)
    assert np.isfinite(bullwhip)


def test_objective_performance():
    """Test and log objective function performance."""
    # Create test data
    n_samples = 1000
    states = np.random.uniform(-1, 1, (n_samples, 11))
    
    # Initialize objectives
    total_cost = TotalCost()
    service_level = ServiceLevel()
    bullwhip = BullwhipEffect()
    
    # Evaluate objectives
    start_time = time.time()
    cost_values = []
    service_values = []
    bullwhip_values = []
    
    for state in states:
        # Reshape state to include history dimension
        state_history = state.reshape(1, -1)
        cost_values.append(total_cost(None, state_history))
        
        # Calculate service level (ratio of fulfilled demand)
        demand = state[2]  # Assuming demand is at index 2
        backlog = state[1]  # Assuming backlog is at index 1
        
        if demand > 0:
            service = max(0, min(1, 1 - backlog/demand))
        else:
            service = 1.0  # Perfect service when no demand
            
        service_values.append(service)
        
        bullwhip_values.append(bullwhip(None, state_history))
    
    # Calculate metrics
    avg_cost = np.mean(cost_values)
    avg_service = np.mean(service_values)
    avg_bullwhip = np.mean(bullwhip_values)
    eval_time = (time.time() - start_time) / n_samples
    
    # Log results
    results = TestResults()
    results.log_objective_performance(avg_cost, avg_service, avg_bullwhip, eval_time)
    results.save_results()
    
    # Assert expected ranges
    assert 0 <= avg_cost <= 1000, f"Average cost {avg_cost} outside expected range"
    assert 0 <= avg_service <= 1, f"Average service level {avg_service} outside expected range"
    assert 1 <= avg_bullwhip <= 5, f"Average bullwhip {avg_bullwhip} outside expected range"
    assert eval_time <= 0.001, f"Evaluation time {eval_time}s too high"
