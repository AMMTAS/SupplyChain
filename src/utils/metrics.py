"""
Performance Metrics for Time Series Forecasting

This module implements various metrics for evaluating forecasting performance,
following standard practices in the forecasting literature.

References:
- Hyndman & Koehler (2006) - "Another look at measures of forecast accuracy"
- Makridakis et al. (2020) - "M4 Competition: 100,000 time series and 61 forecasting methods"
"""

import numpy as np
from typing import Dict, Union, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_intervals: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive forecast accuracy metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_pred_intervals: Prediction intervals (if available)
        sample_weights: Optional weights for weighted metrics
    
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    # Scale-dependent errors
    metrics['mse'] = mean_squared_error(
        y_true, y_pred, sample_weight=sample_weights
    )
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(
        y_true, y_pred, sample_weight=sample_weights
    )
    
    # Percentage errors
    metrics['mape'] = calculate_mape(y_true, y_pred, sample_weights)
    metrics['smape'] = calculate_smape(y_true, y_pred, sample_weights)
    
    # Scale-free errors
    metrics['mase'] = calculate_mase(y_true, y_pred, sample_weights)
    
    # If prediction intervals are provided
    if y_pred_intervals is not None:
        coverage = calculate_prediction_interval_coverage(
            y_true, y_pred_intervals
        )
        metrics.update(coverage)
    
    return metrics


def calculate_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: Optional[np.ndarray] = None
) -> float:
    """Calculate Mean Absolute Percentage Error."""
    if np.any(y_true == 0):
        # Handle zero values
        mask = y_true != 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if sample_weights is not None:
            sample_weights = sample_weights[mask]
    
    percentage_errors = np.abs((y_true - y_pred) / y_true)
    if sample_weights is not None:
        return np.average(percentage_errors, weights=sample_weights)
    return np.mean(percentage_errors)


def calculate_smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: Optional[np.ndarray] = None
) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.abs(y_pred - y_true) / denominator
    if sample_weights is not None:
        return np.average(smape, weights=sample_weights)
    return np.mean(smape)


def calculate_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    seasonality: int = 1
) -> float:
    """
    Calculate Mean Absolute Scaled Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        sample_weights: Optional sample weights
        seasonality: Seasonal period (1 for non-seasonal data)
    """
    # Calculate errors
    errors = np.abs(y_true - y_pred)
    
    # Calculate scaling factor (mean absolute error of naive forecast)
    naive_errors = np.abs(y_true[seasonality:] - y_true[:-seasonality])
    scale = np.mean(naive_errors)
    
    # Handle zero scale
    if scale == 0:
        return np.inf if np.any(errors != 0) else 0
    
    scaled_errors = errors / scale
    if sample_weights is not None:
        return np.average(scaled_errors, weights=sample_weights)
    return np.mean(scaled_errors)


def calculate_prediction_interval_coverage(
    y_true: np.ndarray,
    y_pred_intervals: np.ndarray,
    confidence_levels: Optional[list] = None
) -> Dict[str, float]:
    """
    Calculate prediction interval coverage and sharpness.
    
    Args:
        y_true: Actual values
        y_pred_intervals: Prediction intervals
        confidence_levels: List of confidence levels
    
    Returns:
        Dictionary of coverage metrics
    """
    if confidence_levels is None:
        confidence_levels = [0.5, 0.8, 0.9, 0.95]
    
    coverage_metrics = {}
    n_intervals = y_pred_intervals.shape[1] // 2
    
    for i, level in enumerate(confidence_levels):
        if i < n_intervals:
            lower = y_pred_intervals[:, i]
            upper = y_pred_intervals[:, -(i+1)]
            
            # Calculate coverage
            coverage = np.mean(
                (y_true >= lower) & (y_true <= upper)
            )
            coverage_metrics[f'coverage_{level}'] = coverage
            
            # Calculate interval width (sharpness)
            width = np.mean(upper - lower)
            coverage_metrics[f'interval_width_{level}'] = width
    
    return coverage_metrics
