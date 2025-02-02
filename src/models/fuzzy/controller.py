"""
Fuzzy Logic Controller for Supply Chain Management.

This module implements a fuzzy logic controller that takes enhanced state information
from the Information Sharing Network (ISN) and provides rule-based recommendations
to the Actor-Critic network.

Based on:
1. Petrovic, D., Roy, R., & Petrovic, R. (1999). "Supply chain modelling using fuzzy sets."
   International Journal of Production Economics, 59(1-3), 443-453.
2. Zadeh, L. A. (1996). "Fuzzy logic = computing with words."
   IEEE Transactions on Fuzzy Systems, 4(2), 103-111.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class FuzzyControllerConfig:
    """Configuration for the Fuzzy Logic Controller."""
    input_dim: int = 64  # Matches ISN output dimension
    n_membership_functions: int = 3  # Low, Medium, High
    universe_range: Tuple[float, float] = (-1.0, 1.0)
    defuzz_method: str = 'centroid'

class FuzzyController:
    """
    Fuzzy Logic Controller for supply chain decision support.
    
    Takes enhanced state information from ISN and provides rule-based
    recommendations to the Actor-Critic network.
    """
    
    def __init__(self, config: FuzzyControllerConfig):
        """Initialize the fuzzy controller."""
        self.config = config
        self.universe = np.linspace(*self.config.universe_range, 100)
        self.defuzz_method = self.config.defuzz_method
        
    def process_state(self, state: np.ndarray) -> Dict[str, float]:
        """Process state through fuzzy system.
        
        Args:
            state: State vector containing inventory, demand trend, and service level
            
        Returns:
            Dictionary with order_adjustment and risk_level recommendations
        """
        # Normalize state components
        inventory = state[0]  # Already normalized
        demand_trend = state[1]  # Already normalized
        service_level = state[2]  # Already normalized
        
        # Create antecedent membership functions
        inventory_level = {}
        inventory_level['low'] = fuzz.trimf(self.universe, [-1.0, -1.0, 0.0])
        inventory_level['medium'] = fuzz.trimf(self.universe, [-0.5, 0.0, 0.5])
        inventory_level['high'] = fuzz.trimf(self.universe, [0.0, 1.0, 1.0])

        demand_trend_level = {}
        demand_trend_level['decreasing'] = fuzz.trimf(self.universe, [-1.0, -1.0, 0.0])
        demand_trend_level['stable'] = fuzz.trimf(self.universe, [-0.2, 0.0, 0.2])
        demand_trend_level['increasing'] = fuzz.trimf(self.universe, [0.0, 1.0, 1.0])

        service_level_status = {}
        service_level_status['poor'] = fuzz.trimf(self.universe, [-1.0, -1.0, -0.3])
        service_level_status['acceptable'] = fuzz.trimf(self.universe, [-0.5, 0.0, 0.5])
        service_level_status['good'] = fuzz.trimf(self.universe, [0.3, 1.0, 1.0])
        
        # Create consequent membership functions for order adjustment
        order_adjustment = {}
        order_adjustment['decrease_lot'] = fuzz.trimf(self.universe, [-1.0, -1.0, -0.5])
        order_adjustment['decrease'] = fuzz.trimf(self.universe, [-0.8, -0.4, 0.0])
        order_adjustment['maintain'] = fuzz.trimf(self.universe, [-0.2, 0.0, 0.2])
        order_adjustment['increase'] = fuzz.trimf(self.universe, [0.0, 0.4, 0.8])
        order_adjustment['increase_lot'] = fuzz.trimf(self.universe, [0.5, 1.0, 1.0])

        # Create consequent membership functions for risk level
        risk_level = {}
        risk_level['low'] = fuzz.trimf(self.universe, [-1.0, -1.0, -0.3])
        risk_level['medium'] = fuzz.trimf(self.universe, [-0.5, 0.0, 0.5])
        risk_level['high'] = fuzz.trimf(self.universe, [0.3, 1.0, 1.0])
        
        # Calculate membership degrees
        inventory_low = fuzz.interp_membership(self.universe, inventory_level['low'], inventory)
        inventory_med = fuzz.interp_membership(self.universe, inventory_level['medium'], inventory)
        inventory_high = fuzz.interp_membership(self.universe, inventory_level['high'], inventory)
        
        demand_decreasing = fuzz.interp_membership(self.universe, demand_trend_level['decreasing'], demand_trend)
        demand_stable = fuzz.interp_membership(self.universe, demand_trend_level['stable'], demand_trend)
        demand_increasing = fuzz.interp_membership(self.universe, demand_trend_level['increasing'], demand_trend)
        
        service_poor = fuzz.interp_membership(self.universe, service_level_status['poor'], service_level)
        service_acceptable = fuzz.interp_membership(self.universe, service_level_status['acceptable'], service_level)
        service_good = fuzz.interp_membership(self.universe, service_level_status['good'], service_level)
        
        # Apply fuzzy rules for order adjustment
        # Rule 1: If inventory is low OR demand is increasing OR service is poor, increase orders
        rule1 = np.fmax(np.fmax(inventory_low, demand_increasing), service_poor)
        
        # Rule 2: If inventory is high OR demand is decreasing, decrease orders
        rule2 = np.fmax(inventory_high, demand_decreasing)
        
        # Rule 3: If everything is medium/stable/acceptable, maintain orders
        rule3 = np.fmin(np.fmin(inventory_med, demand_stable), service_acceptable)
        
        # Combine rules
        aggregated = np.zeros_like(self.universe)
        aggregated = np.fmax(aggregated, np.fmin(rule1, order_adjustment['increase_lot']))
        aggregated = np.fmax(aggregated, np.fmin(rule2, order_adjustment['decrease_lot']))
        aggregated = np.fmax(aggregated, np.fmin(rule3, order_adjustment['maintain']))
        
        # Defuzzify order adjustment
        order_adj = fuzz.defuzz(self.universe, aggregated, self.defuzz_method)
        if order_adj is None:  # Handle edge case
            order_adj = 0.0
            
        # Apply fuzzy rules for risk level
        # Rule 1: If inventory is low OR demand is increasing OR service is poor, high risk
        risk_rule1 = np.fmax(np.fmax(inventory_low, demand_increasing), service_poor)
        
        # Rule 2: If inventory and service are good, low risk
        risk_rule2 = np.fmin(inventory_high, service_good)
        
        # Rule 3: Medium risk otherwise
        risk_rule3 = np.fmin(inventory_med, service_acceptable)
        
        # Combine risk rules
        risk_aggregated = np.zeros_like(self.universe)
        risk_aggregated = np.fmax(risk_aggregated, np.fmin(risk_rule1, risk_level['high']))
        risk_aggregated = np.fmax(risk_aggregated, np.fmin(risk_rule2, risk_level['low']))
        risk_aggregated = np.fmax(risk_aggregated, np.fmin(risk_rule3, risk_level['medium']))
        
        # Defuzzify risk level
        risk = fuzz.defuzz(self.universe, risk_aggregated, self.defuzz_method)
        if risk is None:  # Handle edge case
            risk = 0.5  # Default to medium risk
            
        return {
            'order_adjustment': float(order_adj),
            'risk_level': float(risk)
        }
