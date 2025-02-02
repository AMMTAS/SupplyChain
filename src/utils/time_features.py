"""
Time Feature Engineering Utilities

This module provides common time-based feature engineering functions used across
different components of the supply chain system.

References:
- Lai et al. (2018) - "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks"
- Oreshkin et al. (2020) - "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
"""

import numpy as np
from typing import List, Optional
from datetime import datetime


def create_seasonal_features(
    timestamps: List[datetime],
    freq: str = 'H'
) -> np.ndarray:
    """
    Create seasonal features from timestamps.
    
    Args:
        timestamps: List of datetime objects
        freq: Frequency of the time series ('H' for hourly, 'D' for daily)
    
    Returns:
        Array of seasonal features
    """
    features = []
    for ts in timestamps:
        if freq == 'H':
            # Hour of day (normalized)
            hour_sin = np.sin(2 * np.pi * ts.hour / 24)
            hour_cos = np.cos(2 * np.pi * ts.hour / 24)
            
            # Day of week (normalized)
            dow_sin = np.sin(2 * np.pi * ts.weekday() / 7)
            dow_cos = np.cos(2 * np.pi * ts.weekday() / 7)
            
            features.append([hour_sin, hour_cos, dow_sin, dow_cos])
        
        elif freq == 'D':
            # Day of week (normalized)
            dow_sin = np.sin(2 * np.pi * ts.weekday() / 7)
            dow_cos = np.cos(2 * np.pi * ts.weekday() / 7)
            
            # Month of year (normalized)
            month_sin = np.sin(2 * np.pi * ts.month / 12)
            month_cos = np.cos(2 * np.pi * ts.month / 12)
            
            features.append([dow_sin, dow_cos, month_sin, month_cos])
    
    return np.array(features)


def add_time_features(
    data: np.ndarray,
    timestamps: List[datetime],
    freq: str = 'H'
) -> np.ndarray:
    """
    Add time-based features to data array.
    
    Args:
        data: Input data array (samples, features)
        timestamps: List of datetime objects
        freq: Time frequency
    
    Returns:
        Data array with time features added
    """
    # Create seasonal features
    seasonal_features = create_seasonal_features(timestamps, freq)
    
    # Concatenate with original data
    return np.concatenate([data, seasonal_features], axis=1)


def create_lags(
    data: np.ndarray,
    lag_list: Optional[List[int]] = None
) -> np.ndarray:
    """
    Create lagged features from time series data.
    
    Args:
        data: Input time series data
        lag_list: List of lag values to create
    
    Returns:
        Array with lagged features
    """
    if lag_list is None:
        # Default lags based on common patterns
        lag_list = [1, 2, 3, 24, 48, 168]  # hour, day, week for hourly data
    
    n_samples = len(data)
    n_lags = len(lag_list)
    lagged_data = np.zeros((n_samples, n_lags))
    
    for i, lag in enumerate(lag_list):
        lagged_data[lag:, i] = data[:-lag]
    
    return lagged_data
