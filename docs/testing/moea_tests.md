# MOEA Optimizer Test Documentation

## Overview
This document outlines the test suite for the Multi-Objective Evolutionary Algorithm (MOEA) optimizer, including recent improvements and test configurations.

## Test Structure

### 1. Test Fixtures
- `test_optimizer`: Creates a test MOEA optimizer with:
  - 3 objectives (TotalCost, ServiceLevel, BullwhipEffect)
  - Parameter bounds for holding cost, backlog cost, and target service level
  - Configuration for population size, neighborhood size, mutation/crossover rates

### 2. Core Tests
- `test_initialization`: Verifies proper initialization of population, weights, and neighbors
- `test_weight_vector_generation`: Ensures weight vectors sum to 1 and are non-negative
- `test_neighbor_computation`: Validates neighbor computation and uniqueness
- `test_optimization`: Tests the main optimization process with proper state dimensions
- `test_solution_bounds`: Verifies solutions respect parameter bounds
- `test_crossover_operator`: Tests crossover operation maintains valid solutions
- `test_mutation_operator`: Ensures mutation produces valid solutions

### 3. Performance Test
The `test_optimizer_performance` test evaluates and logs optimizer performance metrics:

#### State Configuration
```python
time_steps = 10
features = 4  # inventory, backlog, demand, orders
state = np.random.uniform(-1, 1, (time_steps, features))
```

#### Hypervolume Calculation
```python
# Reference points from actual values
nadir_point = np.max(objective_values, axis=0) * 1.1  # Add 10% margin
ideal_point = np.min(objective_values, axis=0) * 0.9  # Subtract 10% margin

# Normalize to [0,1] range
normalized_objectives = (objective_values - ideal_point) / (nadir_point - ideal_point)
hypervolume = 1.0 - np.mean(np.prod(normalized_objectives, axis=1))
```

#### Performance Metrics
- Convergence Time: 0.1-2.0 seconds
- Pareto Front Size: 10-100 solutions
- Hypervolume: 0.3-0.9 (normalized)

## Recent Improvements

1. **State Representation**
   - Updated state shape to (time_steps, features)
   - Added proper feature dimensions for inventory, backlog, demand, and orders

2. **Hypervolume Calculation**
   - Dynamic reference points based on actual objective values
   - Added safety margins to avoid numerical issues
   - Improved normalization for consistent measurement

3. **Performance Metrics**
   - Adjusted convergence time expectations based on empirical results
   - Added comprehensive logging via TestResults class
   - Implemented robust assertions for all performance metrics

## Usage Notes

1. **Running Tests**
   ```bash
   python -m pytest tests/models/moea/test_optimizer.py -v
   ```

2. **Test Results**
   - Results are logged to `logs/test_results.log`
   - Includes convergence time, pareto front size, and hypervolume metrics
   - Timestamps are added for tracking performance over time

3. **Maintenance**
   - Review performance thresholds periodically
   - Update test state dimensions if supply chain model changes
   - Monitor hypervolume calculation for numerical stability
