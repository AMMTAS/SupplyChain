"""Objective functions for the MOEA optimizer."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np


class ObjectiveFunction(ABC):
    """Base class for objective functions."""
    
    @abstractmethod
    def __call__(self, solution: Dict[str, Any], state: np.ndarray) -> float:
        """Calculate objective value for a given solution.
        
        Args:
            solution: Dictionary containing parameter values
            state: Enhanced state from Information Sharing Network
            
        Returns:
            Objective value (to be minimized)
        """
        pass


class TotalCost(ObjectiveFunction):
    """Total cost objective including inventory, backlog, and ordering costs."""
    
    def __init__(self, holding_cost: float = 1.0, backlog_cost: float = 5.0):
        """Initialize cost parameters.
        
        Args:
            holding_cost: Cost per unit of inventory per time step
            backlog_cost: Cost per unit of backlog per time step
        """
        self.holding_cost = holding_cost
        self.backlog_cost = backlog_cost
    
    def __call__(self, solution: Dict[str, Any], state: np.ndarray) -> float:
        """Calculate total cost from state information.
        
        Args:
            solution: Parameter values (not used for cost calculation)
            state: Enhanced state containing inventory and backlog information
            
        Returns:
            Total cost value
        """
        # Extract inventory and backlog from state
        inventory = state[..., 0]  # Assuming first dimension is inventory
        backlog = state[..., 1]    # Assuming second dimension is backlog
        
        # Calculate costs
        holding_costs = self.holding_cost * np.maximum(inventory, 0)
        backlog_costs = self.backlog_cost * np.maximum(-inventory, 0)
        
        return float(np.sum(holding_costs + backlog_costs))


class ServiceLevel(ObjectiveFunction):
    """Service level objective (to be maximized, so we return negative)."""
    
    def __call__(self, solution: Dict[str, Any], state: np.ndarray) -> float:
        """Calculate service level from state information.
        
        Args:
            solution: Parameter values (not used for service calculation)
            state: Enhanced state containing inventory and demand information
            
        Returns:
            Negative service level (for minimization)
        """
        # Extract inventory and demand from state
        inventory = state[..., 0]  # Assuming first dimension is inventory
        demand = state[..., 2]     # Assuming third dimension is demand
        
        # Calculate fulfilled demand ratio
        fulfilled = np.minimum(inventory, demand)
        service_level = np.mean(fulfilled / np.maximum(demand, 1e-6))
        
        return -float(service_level)  # Negative for minimization


class BullwhipEffect(ObjectiveFunction):
    """Bullwhip effect objective measuring order variance amplification."""
    
    def __init__(self, window_size: int = 10):
        """Initialize bullwhip calculation parameters.
        
        Args:
            window_size: Number of time steps to consider for variance calculation
        """
        self.window_size = window_size
    
    def __call__(self, solution: Dict[str, Any], state: np.ndarray) -> float:
        """Calculate bullwhip effect from state information.
        
        Args:
            solution: Parameter values (not used for bullwhip calculation)
            state: Enhanced state containing order and demand information
            
        Returns:
            Bullwhip effect measure
        """
        # Extract order and demand histories
        orders = state[..., 3]  # Assuming fourth dimension is orders
        demand = state[..., 2]  # Assuming third dimension is demand
        
        # Use recent window for calculation
        recent_orders = orders[-self.window_size:]
        recent_demand = demand[-self.window_size:]
        
        # Calculate variances
        order_var = np.var(recent_orders) if len(recent_orders) > 1 else 0.0
        demand_var = np.var(recent_demand) if len(recent_demand) > 1 else 0.0
        
        # Special case: If both variances are 0 (perfect match), return 1.0
        if order_var == 0 and demand_var == 0:
            return 1.0
            
        # Avoid division by zero
        if demand_var == 0:
            return order_var + 1.0  # Penalize order variance when demand is stable
            
        return float(order_var / demand_var)
