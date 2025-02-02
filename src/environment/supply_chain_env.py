"""
Multi-Echelon Supply Chain Environment

This module implements a realistic supply chain simulation environment following the
Gymnasium interface. The implementation is based on established supply chain theory
and recent research in multi-echelon inventory management.

Key Features:
1. Multi-echelon structure with variable number of echelons
   Based on: Lee, H. L., Padmanabhan, V., & Whang, S. (1997). Information 
   distortion in a supply chain: The bullwhip effect. Management Science, 43(4).

2. Inventory and cost modeling
   Based on: Graves, S. C., & Willems, S. P. (2003). Supply chain design: Safety 
   stock placement and supply chain configuration. Handbooks in OR & MS, 11.

3. Lead time modeling
   Based on: Simchi-Levi, D., & Zhao, Y. (2011). Performance evaluation of 
   stochastic multi-echelon inventory systems: A survey. Advances in OR.

4. Service level metrics
   Based on: Gunasekaran, A., Patel, C., & McGaughey, R. E. (2004). A framework 
   for supply chain performance measurement. Int. Journal of Production Economics.

5. Information flow modeling
   Based on: Cachon, G. P., & Fisher, M. (2000). Supply chain inventory management 
   and the value of shared information. Management Science, 46(8).

The environment implements the standard Gymnasium interface for compatibility with
modern reinforcement learning algorithms while maintaining realistic supply chain
dynamics based on established academic literature.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SupplyChainEnv(gym.Env):
    """
    Multi-echelon supply chain environment.
    
    State Space:
        - Inventory levels for each echelon
        - Backlog levels for each echelon
        - On-order inventory (pipeline inventory) for each echelon
        - Demand history
        - Lead time status
    
    Action Space:
        - Order quantities for each echelon
    
    Rewards:
        Combination of:
        - Holding costs (penalty for excess inventory)
        - Backlog costs (penalty for unfulfilled demand)
        - Transportation costs
        - Service level rewards
    """
    
    def __init__(
        self,
        num_echelons: int = 4,
        max_steps: int = 100,
        demand_mean: float = 100.0,
        demand_std: float = 20.0,
        lead_time_mean: int = 2,
        lead_time_std: float = 0.5,
        holding_cost: Union[float, List[float]] = 0.5,
        backlog_cost: Union[float, List[float]] = 2.0,
        transportation_cost: Union[float, List[float]] = 0.3,
        service_level_target: float = 0.95,
        max_inventory: float = 500.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the supply chain environment.
        
        Args:
            num_echelons: Number of echelons in the supply chain
            max_steps: Maximum number of steps per episode
            demand_mean: Mean customer demand
            demand_std: Standard deviation of customer demand
            lead_time_mean: Mean lead time between echelons
            lead_time_std: Standard deviation of lead time
            holding_cost: Cost per unit of inventory held
            backlog_cost: Cost per unit of backlogged demand
            transportation_cost: Cost per unit transported
            service_level_target: Target service level (fill rate)
            max_inventory: Maximum inventory capacity per echelon
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Environment parameters
        self.num_echelons = num_echelons
        self.max_steps = max_steps
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        self.lead_time_mean = lead_time_mean
        self.lead_time_std = lead_time_std
        self.max_inventory = max_inventory
        self.service_level_target = service_level_target
        
        # Convert costs to arrays if necessary
        self.holding_cost = self._ensure_cost_array(holding_cost)
        self.backlog_cost = self._ensure_cost_array(backlog_cost)
        self.transportation_cost = self._ensure_cost_array(transportation_cost)
        
        # Initialize random number generator
        self.np_random = None
        self.seed(seed)
        
        # Define action space: order quantities for each echelon
        self.action_space = spaces.Box(
            low=0,
            high=max_inventory,
            shape=(num_echelons,),
            dtype=np.float32
        )
        
        # Define observation space
        obs_dim = (
            num_echelons +  # Inventory levels
            num_echelons +  # Backlog levels
            num_echelons +  # On-order inventory
            1 +             # Current demand
            1              # Average demand (as reference)
        )
        self.observation_space = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
    
    def _ensure_cost_array(self, cost: Union[float, List[float]]) -> np.ndarray:
        """Convert cost to array if it's a single value."""
        if isinstance(cost, (int, float)):
            return np.array([cost] * self.num_echelons)
        return np.array(cost)
    
    def _generate_lead_times(self) -> np.ndarray:
        """Generate random lead times for each echelon."""
        lead_times = self.np_random.normal(
            self.lead_time_mean,
            self.lead_time_std,
            size=self.num_echelons
        )
        return np.maximum(1, np.round(lead_times)).astype(int)
    
    def _generate_demand(self) -> float:
        """Generate customer demand for the current timestep."""
        demand = self.np_random.normal(self.demand_mean, self.demand_std)
        return max(0, demand)  # Demand cannot be negative
    
    def _calculate_rewards(
        self,
        inventory_levels: np.ndarray,
        backlog_levels: np.ndarray,
        orders: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate rewards and costs.
        
        The reward function follows standard supply chain cost modeling as described in:
        Simchi-Levi, D., et al. (2008). Designing and managing the supply chain.
        
        Reward Calculation Details:
        1. Cost Normalization:
           - All costs are normalized by (demand_mean * max_steps) to make them independent
             of the problem scale and episode length
           - This ensures rewards are comparable across different demand scenarios
        
        2. Individual Components:
           - Holding costs: Cost for storing excess inventory
           - Backlog costs: Penalty for unfulfilled demand (2x weight of holding)
           - Transportation costs: Cost for moving inventory
           - Service level reward: Bonus for meeting target service level
        
        3. Reward Scaling:
           - Components are weighted to prioritize backlog reduction (0.2)
           - Other components have equal weights (0.1)
           - Final reward is clipped to [-10, -1] for stable learning
        
        Args:
            inventory_levels: Current inventory at each echelon
            backlog_levels: Current backlog at each echelon
            orders: Order quantities placed by each echelon
        
        Returns:
            tuple: (total_reward, cost_breakdown)
        """
        # Calculate individual costs (normalized by demand mean and episode length)
        normalization_factor = self.demand_mean * self.max_steps
        holding_costs = -np.sum(self.holding_cost * np.maximum(0, inventory_levels)) / normalization_factor
        backlog_costs = -np.sum(self.backlog_cost * backlog_levels) / normalization_factor
        transportation_costs = -np.sum(self.transportation_cost * orders) / normalization_factor
        
        # Calculate service level reward (normalized)
        service_level = self._calculate_service_level()
        service_level_reward = (
            0.1 if service_level >= self.service_level_target
            else -0.2 * (self.service_level_target - service_level)
        )
        
        # Scale rewards to be in a reasonable range
        total_reward = (
            0.1 * holding_costs +      # Lower weight to holding cost
            0.2 * backlog_costs +      # Higher weight to backlog (prioritize customer satisfaction)
            0.1 * transportation_costs +  # Lower weight to transportation
            0.1 * service_level_reward   # Lower weight to service level bonus
        )
        
        # Clip reward to reasonable range for stable learning
        total_reward = np.clip(total_reward, -10.0, -1.0)
        
        # Create cost breakdown
        cost_breakdown = {
            'holding_costs': holding_costs,
            'backlog_costs': backlog_costs,
            'transportation_costs': transportation_costs,
            'service_level_reward': service_level_reward,
            'total_reward': total_reward
        }
        
        return total_reward, cost_breakdown
    
    def _calculate_service_level(self) -> float:
        """Calculate the current service level (fill rate)."""
        if self.total_demand == 0:
            return 1.0
        return self.fulfilled_demand / self.total_demand
    
    def _update_state(self, orders: np.ndarray):
        """Update the environment state based on orders."""
        # Generate new demand
        current_demand = self._generate_demand()
        self.total_demand += current_demand
        
        # Update pipeline inventory (orders in transit)
        self.pipeline_inventory = np.roll(self.pipeline_inventory, -1, axis=0)
        self.pipeline_inventory[-1] = orders
        
        # Receive inventory from pipeline
        received_inventory = self.pipeline_inventory[0]
        
        # Update inventory levels
        self.inventory_levels += received_inventory
        
        # Fulfill demand and update backlog
        fulfilled = np.minimum(current_demand, self.inventory_levels[0])
        self.fulfilled_demand += fulfilled
        
        unfulfilled = current_demand - fulfilled
        self.backlog_levels[0] += unfulfilled
        self.inventory_levels[0] -= fulfilled
        
        # Propagate orders and update other echelons
        for i in range(1, self.num_echelons):
            order = orders[i-1]
            fulfilled = np.minimum(order, self.inventory_levels[i])
            self.inventory_levels[i] -= fulfilled
            unfulfilled = order - fulfilled
            self.backlog_levels[i] += unfulfilled
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed."""
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed(seed)
        
        # Reset state variables
        self.inventory_levels = np.zeros(self.num_echelons)
        self.backlog_levels = np.zeros(self.num_echelons)
        self.pipeline_inventory = np.zeros((self.lead_time_mean, self.num_echelons))
        self.current_step = 0
        
        # Reset metrics
        self.total_demand = 0
        self.fulfilled_demand = 0
        
        # Get initial observation
        observation = np.concatenate([
            self.inventory_levels,
            self.backlog_levels,
            self.pipeline_inventory.sum(axis=0),
            [self._generate_demand()],
            [self.demand_mean]
        ])
        
        return observation, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Array of order quantities for each echelon
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Ensure action is valid
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Update state based on action
        self._update_state(action)
        
        # Calculate rewards
        reward, cost_breakdown = self._calculate_rewards(
            self.inventory_levels,
            self.backlog_levels,
            action
        )
        
        # Check if episode is done
        self.current_step += 1
        terminated = False  # Episode can only end by truncation
        truncated = self.current_step >= self.max_steps
        
        # Get observation
        observation = np.concatenate([
            self.inventory_levels,
            self.backlog_levels,
            self.pipeline_inventory.sum(axis=0),
            [self._generate_demand()],
            [self.demand_mean]
        ])
        
        # Prepare info dict
        info = {
            'inventory_levels': self.inventory_levels,
            'backlog_levels': self.backlog_levels,
            'pipeline_inventory': self.pipeline_inventory,
            'service_level': self._calculate_service_level(),
            **cost_breakdown
        }
        
        return observation, reward, terminated, truncated, info
