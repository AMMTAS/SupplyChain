"""Performance metrics for Information Sharing Network.

This module provides comprehensive metrics for evaluating:
1. Information quality
2. Supply chain performance
3. Network efficiency
"""

from typing import List, Dict, Optional
import torch
import numpy as np
from scipy.stats import entropy
from config.information_sharing_config import NetworkConfig


def calculate_bullwhip_effect(
    demand: torch.Tensor,
    orders: torch.Tensor
) -> float:
    """Calculate the bullwhip effect ratio.
    
    Args:
        demand: Demand time series
        orders: Order time series
    
    Returns:
        Bullwhip effect ratio (variance amplification)
    """
    demand_var = torch.var(demand)
    orders_var = torch.var(orders)
    return (orders_var / demand_var).item()


def calculate_information_loss(
    original: torch.Tensor,
    processed: torch.Tensor,
    reduction: str = 'mean'
) -> float:
    """Calculate information loss using MSE.
    
    Args:
        original: Original signal
        processed: Processed signal
        reduction: How to reduce the loss ('mean' or 'sum')
    
    Returns:
        Information loss value
    """
    if reduction == 'mean':
        return torch.mean((original - processed) ** 2).item()
    return torch.sum((original - processed) ** 2).item()


def calculate_mutual_information(
    x: torch.Tensor,
    y: torch.Tensor,
    n_bins: int = 20
) -> float:
    """Calculate mutual information between two signals.
    
    Args:
        x: First signal
        y: Second signal
        n_bins: Number of bins for histogram
    
    Returns:
        Mutual information value
    """
    # Convert to numpy for histogram calculation
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    # Calculate joint histogram
    hist_2d, _, _ = np.histogram2d(x_np.flatten(), y_np.flatten(), n_bins)
    
    # Calculate marginal histograms
    hist_x = hist_2d.sum(axis=1)
    hist_y = hist_2d.sum(axis=0)
    
    # Calculate entropies
    h_x = entropy(hist_x)
    h_y = entropy(hist_y)
    h_xy = entropy(hist_2d.flatten())
    
    # Mutual information = H(X) + H(Y) - H(X,Y)
    return h_x + h_y - h_xy


def calculate_delay_impact(
    original_states: List[torch.Tensor],
    delayed_states: List[torch.Tensor],
    config: NetworkConfig
) -> Dict[int, float]:
    """Calculate impact of delays on each echelon.
    
    Args:
        original_states: Original state time series
        delayed_states: Delayed state time series
        config: Network configuration
    
    Returns:
        Dictionary mapping echelon index to delay impact
    """
    delay_impacts = {}
    for i in range(config.num_echelons):
        delay = config.node_configs[i].delay
        if delay > 0:
            # Calculate MSE between original and delayed signals
            impact = calculate_information_loss(
                original_states[i],
                delayed_states[i]
            )
            delay_impacts[i] = impact
    return delay_impacts


def calculate_forecast_accuracy(
    predictions: torch.Tensor,
    actuals: torch.Tensor,
    metric: str = 'mape'
) -> float:
    """Calculate forecast accuracy using various metrics.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        metric: Metric to use ('mape', 'rmse', 'mae')
    
    Returns:
        Forecast accuracy metric value
    """
    if metric == 'mape':
        return torch.mean(torch.abs((actuals - predictions) / actuals)).item() * 100
    elif metric == 'rmse':
        return torch.sqrt(torch.mean((actuals - predictions) ** 2)).item()
    elif metric == 'mae':
        return torch.mean(torch.abs(actuals - predictions)).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


def calculate_network_efficiency(
    network_outputs: List[torch.Tensor],
    computation_time: float,
    memory_usage: float
) -> Dict[str, float]:
    """Calculate network efficiency metrics.
    
    Args:
        network_outputs: List of network outputs
        computation_time: Time taken for processing
        memory_usage: Memory used during processing
    
    Returns:
        Dictionary of efficiency metrics
    """
    return {
        'throughput': len(network_outputs) / computation_time,
        'memory_per_sample': memory_usage / len(network_outputs),
        'time_per_sample': computation_time / len(network_outputs)
    }


def calculate_information_propagation_speed(
    states_history: List[List[torch.Tensor]],
    config: NetworkConfig
) -> Dict[int, float]:
    """Calculate how quickly information propagates through network.
    
    Args:
        states_history: History of states
        config: Network configuration
    
    Returns:
        Dictionary mapping echelon index to propagation speed
    """
    speeds = {}
    for i in range(1, config.num_echelons):
        # Calculate cross-correlation between adjacent echelons
        signal1 = torch.stack([s[i-1] for s in states_history])
        signal2 = torch.stack([s[i] for s in states_history])
        
        xcorr = torch.tensor([
            torch.sum(signal1[:-k] * signal2[k:]).item()
            for k in range(1, min(20, len(states_history)))
        ])
        
        # Find lag with maximum correlation
        max_lag = torch.argmax(xcorr).item() + 1
        speeds[i] = 1.0 / max_lag  # Speed = 1/lag
    
    return speeds


def calculate_comprehensive_metrics(
    network_config: NetworkConfig,
    states_history: List[List[torch.Tensor]],
    enhanced_states_history: List[List[torch.Tensor]],
    computation_metrics: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """Calculate comprehensive set of performance metrics.
    
    Args:
        network_config: Network configuration
        states_history: History of original states
        enhanced_states_history: History of enhanced states
        computation_metrics: Optional computational performance metrics
    
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # 1. Information Quality Metrics
    for i in range(network_config.num_echelons):
        orig_states = torch.stack([s[i] for s in states_history])
        enh_states = torch.stack([s[i] for s in enhanced_states_history])
        
        metrics[f'info_loss_echelon_{i}'] = calculate_information_loss(
            orig_states, enh_states
        )
        metrics[f'mutual_info_echelon_{i}'] = calculate_mutual_information(
            orig_states, enh_states
        )
    
    # 2. Bullwhip Effect Metrics
    for i in range(1, network_config.num_echelons):
        metrics[f'bullwhip_ratio_{i}'] = calculate_bullwhip_effect(
            torch.stack([s[i-1] for s in states_history]),
            torch.stack([s[i] for s in states_history])
        )
    
    # 3. Delay Impact Metrics
    delay_impacts = calculate_delay_impact(
        [s[0] for s in states_history],
        [s[0] for s in enhanced_states_history],
        network_config
    )
    metrics.update({f'delay_impact_{k}': v for k, v in delay_impacts.items()})
    
    # 4. Information Propagation Metrics
    prop_speeds = calculate_information_propagation_speed(
        states_history, network_config
    )
    metrics.update({f'prop_speed_{k}': v for k, v in prop_speeds.items()})
    
    # 5. Computational Efficiency Metrics
    if computation_metrics:
        metrics.update(computation_metrics)
    
    return metrics
