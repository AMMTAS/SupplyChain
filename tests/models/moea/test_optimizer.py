"""Tests for MOEA optimizer."""

import pytest
import numpy as np
import torch
import time
from src.models.moea.optimizer import MOEAConfig, MOEAOptimizer
from src.models.moea.objectives import TotalCost, ServiceLevel, BullwhipEffect


@pytest.fixture
def test_optimizer():
    """Create test MOEA optimizer."""
    objectives = [
        TotalCost(),
        ServiceLevel(),
        BullwhipEffect()
    ]
    
    parameter_bounds = {
        'holding_cost': (0.1, 5.0),
        'backlog_cost': (1.0, 10.0),
        'target_service': (0.9, 0.99)
    }
    
    config = MOEAConfig(
        n_objectives=3,
        population_size=20,
        neighborhood_size=5,
        mutation_rate=0.1,
        crossover_rate=0.9,
        n_generations=100
    )
    
    return MOEAOptimizer(
        config=config,
        objectives=objectives,
        parameter_bounds=parameter_bounds
    )


def test_initialization(test_optimizer):
    """Test optimizer initialization."""
    assert len(test_optimizer.population) == test_optimizer.config.population_size
    assert len(test_optimizer.weights) == test_optimizer.config.population_size
    assert len(test_optimizer.neighbors) == test_optimizer.config.population_size
    assert len(test_optimizer.neighbors[0]) == test_optimizer.config.neighborhood_size


def test_weight_vector_generation(test_optimizer):
    """Test weight vector generation."""
    weights = test_optimizer.weights
    assert weights.shape == (test_optimizer.config.population_size, test_optimizer.config.n_objectives)
    assert np.allclose(weights.sum(axis=1), 1.0)  # Sum to 1
    assert np.all(weights >= 0)  # Non-negative


def test_neighbor_computation(test_optimizer):
    """Test neighbor computation."""
    neighbors = test_optimizer.neighbors
    for i in range(test_optimizer.config.population_size):
        assert len(neighbors[i]) == test_optimizer.config.neighborhood_size
        assert i in neighbors[i]  # Include self
        assert len(set(neighbors[i])) == len(neighbors[i])  # No duplicates


def test_optimization(test_optimizer):
    """Test optimization process."""
    # Create state with proper shape (time_steps, features)
    time_steps = 10
    features = 4  # inventory, backlog, demand, orders
    state = np.random.uniform(-1, 1, (time_steps, features))
    
    solutions = test_optimizer.optimize(state)
    
    assert len(solutions) == test_optimizer.config.population_size
    
    # Check solution bounds
    for solution_tuple in solutions:
        solution = solution_tuple[0]  # Extract solution dict from tuple
        objectives = solution_tuple[1]  # Extract objective values
        
        # Check solution bounds
        for param, (min_val, max_val) in test_optimizer.parameter_bounds.items():
            assert min_val <= solution[param] <= max_val
            
        # Check objectives are reasonable
        assert len(objectives) == test_optimizer.config.n_objectives
        assert all(isinstance(obj, (int, float)) for obj in objectives)


def test_solution_bounds(test_optimizer):
    """Test solution bounds are respected."""
    for solution in test_optimizer.population:
        for param, (min_val, max_val) in test_optimizer.parameter_bounds.items():
            assert min_val <= solution[param] <= max_val


def test_crossover_operator(test_optimizer):
    """Test crossover operator."""
    parent1 = test_optimizer.population[0]
    parent2 = test_optimizer.population[1]
    
    child = test_optimizer._crossover(parent1, parent2)
    
    # Check child bounds
    for param, (min_val, max_val) in test_optimizer.parameter_bounds.items():
        assert min_val <= child[param] <= max_val


def test_mutation_operator(test_optimizer):
    """Test mutation operator."""
    solution = test_optimizer.population[0]
    mutated = test_optimizer._mutate(solution)
    
    # Check mutated bounds
    for param, (min_val, max_val) in test_optimizer.parameter_bounds.items():
        assert min_val <= mutated[param] <= max_val


def test_optimizer_performance():
    """Test and log optimizer performance."""
    # Create test environment state with proper shape
    time_steps = 10
    features = 4  # inventory, backlog, demand, orders
    state = np.random.uniform(-1, 1, (time_steps, features))
    
    # Initialize optimizer
    config = MOEAConfig(
        n_objectives=3,
        population_size=100,
        neighborhood_size=20,
        mutation_rate=0.1,
        crossover_rate=0.9,
        n_generations=50
    )
    
    objectives = [
        TotalCost(),
        ServiceLevel(),
        BullwhipEffect()
    ]
    
    parameter_bounds = {
        'holding_cost': (0.1, 5.0),
        'backlog_cost': (1.0, 10.0),
        'target_service': (0.9, 0.99)
    }
    
    optimizer = MOEAOptimizer(
        config=config,
        objectives=objectives,
        parameter_bounds=parameter_bounds
    )
    
    # Run optimization
    start_time = time.time()
    solutions = optimizer.optimize(state)
    convergence_time = time.time() - start_time
    
    # Evaluate final solutions
    objective_values = np.array([
        optimizer._evaluate_solution(sol, state)
        for sol in solutions
    ])
    
    # Calculate metrics
    pareto_size = len(solutions)
    
    # Calculate hypervolume
    # Reference point slightly worse than worst observed values
    nadir_point = np.max(objective_values, axis=0) * 1.1  # Add 10% margin
    ideal_point = np.min(objective_values, axis=0) * 0.9  # Subtract 10% margin
    
    # Normalize objectives to [0,1] range
    normalized_objectives = (objective_values - ideal_point) / (nadir_point - ideal_point)
    
    # Calculate hypervolume as percentage of total volume
    hypervolume = 1.0 - np.mean(np.prod(normalized_objectives, axis=1))
    
    # Log results
    from tests.test_results import TestResults
    results = TestResults()
    results.log_moea_results(
        convergence_time=convergence_time,
        pareto_size=pareto_size,
        hypervolume=hypervolume
    )
    results.save_results()
    
    # Assert reasonable ranges
    assert 0.1 <= convergence_time <= 5.0, "Convergence time outside expected range"
    assert 10 <= pareto_size <= 100, "Pareto front size outside expected range"
    assert 0.3 <= hypervolume <= 0.9, "Hypervolume outside expected range"
