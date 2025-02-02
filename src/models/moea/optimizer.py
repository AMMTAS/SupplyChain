"""MOEA/D implementation for supply chain optimization."""

from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
from dataclasses import dataclass


@dataclass
class MOEAConfig:
    """Configuration for MOEA/D optimizer."""
    n_objectives: int
    population_size: int
    neighborhood_size: int
    mutation_rate: float
    crossover_rate: float
    n_generations: int = 100
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.population_size > 0
        assert 0 < self.neighborhood_size <= self.population_size
        assert 0 <= self.mutation_rate <= 1
        assert 0 <= self.crossover_rate <= 1
        assert self.n_generations > 0


class MOEAOptimizer:
    """Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D).
    
    Based on:
    Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm 
    based on decomposition. IEEE Transactions on Evolutionary Computation.
    """
    
    def __init__(
        self,
        config: MOEAConfig,
        objectives: List[Callable],
        parameter_bounds: Dict[str, Tuple[float, float]]
    ):
        """Initialize MOEA/D optimizer.
        
        Args:
            config: Optimizer configuration
            objectives: List of objective functions to minimize
            parameter_bounds: Dictionary mapping parameter names to (min, max) bounds
        """
        self.config = config
        self.objectives = objectives
        self.parameter_bounds = parameter_bounds
        
        # Initialize population
        self.population = self._initialize_population()
        self.weights = self._generate_weight_vectors()
        self.neighbors = self._compute_neighbors()
        
        # Initialize best solutions
        self.ideal_point = np.inf * np.ones(len(objectives))
        self.nadir_point = -np.inf * np.ones(len(objectives))
        self.reference_points = []
    
    def _initialize_population(self) -> List[Dict[str, float]]:
        """Initialize random population within parameter bounds."""
        population = []
        for _ in range(self.config.population_size):
            solution = {}
            for param, (min_val, max_val) in self.parameter_bounds.items():
                solution[param] = np.random.uniform(min_val, max_val)
            population.append(solution)
        return population
    
    def _generate_weight_vectors(self) -> np.ndarray:
        """Generate uniformly distributed weight vectors."""
        weights = np.random.dirichlet(
            np.ones(len(self.objectives)),
            size=self.config.population_size
        )
        return weights
    
    def _compute_neighbors(self) -> List[List[int]]:
        """Compute neighbors based on weight vector distances."""
        neighbors = []
        for i in range(self.config.population_size):
            # Compute distances to all other weight vectors
            distances = np.linalg.norm(
                self.weights - self.weights[i],
                axis=1
            )
            # Get indices of k nearest neighbors
            neighbor_indices = np.argsort(distances)[
                :self.config.neighborhood_size
            ]
            neighbors.append(neighbor_indices.tolist())
        return neighbors
    
    def _evaluate_solution(
        self,
        solution: Dict[str, float],
        state: np.ndarray
    ) -> np.ndarray:
        """Evaluate solution on all objectives."""
        return np.array([
            obj(solution, state) for obj in self.objectives
        ])
    
    def _update_reference_points(
        self,
        objective_values: np.ndarray
    ) -> None:
        """Update ideal and nadir points."""
        self.ideal_point = np.minimum(
            self.ideal_point,
            np.min(objective_values, axis=0)
        )
        self.nadir_point = np.maximum(
            self.nadir_point,
            np.max(objective_values, axis=0)
        )
    
    def _check_solution_bounds(self, solution: Dict[str, float]) -> bool:
        """Check if solution parameters are within bounds."""
        for param_name, value in solution.items():
            min_val, max_val = self.parameter_bounds[param_name]
            if not (min_val <= value <= max_val):
                return False
        return True
    
    def _crossover(
        self,
        parent1: Dict[str, float],
        parent2: Dict[str, float]
    ) -> Dict[str, float]:
        """Perform crossover between two parent solutions."""
        if np.random.random() > self.config.crossover_rate:
            return parent1.copy()
            
        child = {}
        for param_name in self.parameter_bounds:
            # Uniform crossover
            if np.random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        return child

    def _mutate(self, solution: Dict[str, float]) -> Dict[str, float]:
        """Apply mutation to a solution."""
        mutated = solution.copy()
        for param_name, value in solution.items():
            if np.random.random() < self.config.mutation_rate:
                min_val, max_val = self.parameter_bounds[param_name]
                # Add random noise within bounds
                noise = np.random.normal(0, 0.1 * (max_val - min_val))
                mutated[param_name] = np.clip(
                    value + noise,
                    min_val,
                    max_val
                )
        return mutated
    
    def optimize(self, state: np.ndarray) -> List[Tuple[Dict[str, float], List[float]]]:
        """Run optimization to find Pareto-optimal solutions.
        
        Args:
            state: Current environment state
            
        Returns:
            List of tuples (solution_dict, objective_values) where:
            - solution_dict maps parameter names to values
            - objective_values is a list of objective function values
        """
        # Initialize population
        population = []
        for _ in range(self.config.population_size):
            solution = {}
            for param_name, (min_val, max_val) in self.parameter_bounds.items():
                solution[param_name] = np.random.uniform(min_val, max_val)
            population.append(solution)

        # Run optimization for specified generations
        for _ in range(self.config.n_generations):
            # Evaluate solutions
            objective_values = []
            for solution in population:
                objectives = [
                    obj(solution, state) for obj in self.objectives
                ]
                objective_values.append(objectives)

            # Update population
            for i in range(len(population)):
                # Get neighbors
                neighbors = self._get_neighbors(i)
                
                # Select parents from neighbors
                parent1 = population[i]
                parent2 = population[np.random.choice(neighbors)]
                
                # Generate offspring through crossover and mutation
                offspring = self._crossover(parent1, parent2)
                offspring = self._mutate(offspring)
                
                # Replace if offspring dominates
                if self._dominates(offspring, parent1, state):
                    population[i] = offspring

        # Return solutions with their objective values
        return [(sol, self._evaluate_solution(sol, state)) for sol in population]

    def _get_neighbors(self, index: int) -> List[int]:
        """Get indices of neighboring solutions."""
        return self.neighbors[index]

    def _dominates(
        self,
        solution1: Dict[str, float],
        solution2: Dict[str, float],
        state: np.ndarray
    ) -> bool:
        """Check if solution1 dominates solution2."""
        objectives1 = self._evaluate_solution(solution1, state)
        objectives2 = self._evaluate_solution(solution2, state)
        return np.all(objectives1 <= objectives2) and np.any(objectives1 < objectives2)
