"""
MSWARM library for missing value imputation

This library provides functions for imputing missing values in time series data using the
MSWARM (Modified Standard Weighted Average Robust Method). This method combines local
information with global trends, adapting to local data characteristics and accounting for
data asymmetry.

Main functions:
- find_optimal_neighbors: Determines the optimal number of neighbors for imputation
- impute_with_specified_neighbors: Performs imputation with a specified number of neighbors
- MissingValuesWithFlag: Main class for applying imputation methods

Usage:
Import this library and use the MissingValuesWithFlag class, specifying the
data frame, value column, and trend column.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

def find_optimal_neighbors(df, value_col, trend_col, seasonal_period=30, max_neighbors=10):
    df_copy = df.copy()
    
    valid_mask = ~df_copy[value_col].isna()
    
    if valid_mask.sum() < 10:
        return 3
    
    if valid_mask.sum() < max_neighbors * 2:
        return min(3, valid_mask.sum() // 2)
    
    test_indices = df_copy[valid_mask].sample(min(10, valid_mask.sum() // 3)).index
    
    mse_values = []
    
    for n in range(2, max_neighbors + 1):
        total_mse = 0
        
        for idx in test_indices:
            test_df = df_copy.copy()
            test_df.loc[idx, value_col] = np.nan
            
            imputed = impute_with_specified_neighbors(test_df, value_col, trend_col, n)
            
            true_value = df_copy.loc[idx, value_col]
            imputed_value = imputed[idx]
            
            if not np.isnan(imputed_value):
                total_mse += (true_value - imputed_value) ** 2
        
        avg_mse = total_mse / len(test_indices) if test_indices.size > 0 else float('inf')
        mse_values.append(avg_mse)
    
    if not mse_values or all(np.isnan(mse_values)):
        return 3
    
    best_n = np.nanargmin(mse_values) + 2
    
    return best_n

def impute_with_specified_neighbors(df, value_col, trend_col, n_neighbors=3):
    df_copy = df.copy()
    result_series = df_copy[value_col].copy()
    
    valid_mask = ~df_copy[value_col].isna()
    
    if valid_mask.sum() == 0:
        return result_series
    
    missing_indices = df_copy[~valid_mask].index
    
    if len(missing_indices) == 0:
        return result_series
    
    valid_indices = df_copy[valid_mask].index
    
    first_valid_idx = valid_indices.min()
    last_valid_idx = valid_indices.max()
    
    trend_values = df_copy[trend_col].fillna(method='ffill').fillna(method='bfill')
    
    prev_imputed_values = {}
    
    for missing_idx in missing_indices:
        if missing_idx < first_valid_idx or missing_idx > last_valid_idx:
            continue
            
        left_neighbors = []
        right_neighbors = []
        
        left_idx = missing_idx - 1
        right_idx = missing_idx + 1
        
        left_count = 0
        right_count = 0
        
        while left_count < n_neighbors and left_idx >= first_valid_idx:
            if not pd.isna(result_series[left_idx]):
                left_neighbors.append((left_idx, result_series[left_idx]))
                left_count += 1
            left_idx -= 1
            
        while right_count < n_neighbors and right_idx <= last_valid_idx:
            if not pd.isna(result_series[right_idx]):
                right_neighbors.append((right_idx, result_series[right_idx]))
                right_count += 1
            right_idx += 1
            
        if len(left_neighbors) == 0 and len(right_neighbors) == 0:
            continue
            
        left_weights = []
        left_values = []
        
        for idx, val in left_neighbors:
            distance = missing_idx - idx
            weight = np.exp(-0.15 * distance)
            left_weights.append(weight)
            left_values.append(val)
            
        right_weights = []
        right_values = []
        
        for idx, val in right_neighbors:
            distance = idx - missing_idx
            weight = np.exp(-0.15 * distance)
            right_weights.append(weight)
            right_values.append(val)
            
        all_weights = left_weights + right_weights
        all_values = left_values + right_values
        
        if len(all_values) == 0:
            continue
            
        total_weight = sum(all_weights)
        
        if total_weight == 0:
            weighted_avg = np.mean(all_values)
        else:
            weighted_avg = sum(w * v for w, v in zip(all_weights, all_values)) / total_weight
            
        skew_factor = 0.01
        
        if len(left_values) > 0 and len(right_values) > 0:
            left_avg = np.mean(left_values)
            right_avg = np.mean(right_values)
            skew = (right_avg - left_avg) * skew_factor
            weighted_avg += skew
            
        trend_value = trend_values[missing_idx]
        
        max_gap = max(missing_idx - first_valid_idx, last_valid_idx - missing_idx)
        gap_factor = min(1.0, (missing_idx - first_valid_idx) / max_gap if missing_idx - first_valid_idx > 0 else 0)
        
        trend_weight = 0.3 + 0.4 * gap_factor
        local_weight = 1.0 - trend_weight
        
        if missing_idx - 1 in prev_imputed_values:
            prev_value = prev_imputed_values[missing_idx - 1]
            curr_value = local_weight * weighted_avg + trend_weight * trend_value
            
            if abs(curr_value - prev_value) > 0.5 * abs(trend_value - prev_value):
                oscillation_factor = 0.5
                final_value = (1 - oscillation_factor) * curr_value + oscillation_factor * prev_value
            else:
                final_value = 0.7 * curr_value + 0.3 * prev_value
        else:
            final_value = local_weight * weighted_avg + trend_weight * trend_value
            
        result_series[missing_idx] = final_value
        prev_imputed_values[missing_idx] = final_value
        
    return result_series

def dynamic_weighted_local_and_trend_imputation(df, value_col, trend_col):
    return impute_with_specified_neighbors(df, value_col, trend_col, n_neighbors=3)

def plot_imputed_data(df, col_Y, col_X, seasonal_period, deviation_threshold=None):
    """
    Compute the interpolated trend, perform dynamic imputation, and plot the results.
    """
    df_copy = df.copy()
    
    if deviation_threshold is None:
        deviation_threshold = df_copy[col_Y].std() * 3
        
    interpolated_data = df_copy[col_Y].interpolate(method='linear')
    decomposition = seasonal_decompose(interpolated_data, period=seasonal_period, model='additive')
    trend = decomposition.trend
    pchip_trend = trend.interpolate(method='pchip')
    
    if deviation_threshold:
        deviations = (pchip_trend - df_copy[col_Y]).abs()
        exceed_threshold = deviations > deviation_threshold
        if exceed_threshold.any():
            cut_point = exceed_threshold.idxmax()
            pchip_trend[cut_point:] = np.nan
    
    df_copy['Pchip_Trend'] = pchip_trend
    
    mvi = MissingValuesWithFlag()
    imputed_series, optimal_n = mvi.Calculate(df_copy, col_Y, 'Pchip_Trend')
    df_copy['Dynamic_Imputed_Temperature_v2'] = imputed_series
    
    missing_count = df_copy[col_Y].isna().sum()
    total_count = len(df_copy)
    missing_proportion = (missing_count / total_count) * 100
    
    plt.figure(figsize=(20, 10))
    
    plt.plot(df_copy[col_X], df_copy['Pchip_Trend'], label='Pchip Interpolated Trend', color='green', linestyle='dotted', lw=2.5)
    plt.plot(df_copy[col_X], df_copy['Dynamic_Imputed_Temperature_v2'], label='Dynamic Imputed Data', color='red', alpha=0.25, lw=5)
    plt.plot(df_copy[col_X], df_copy[col_Y], label='Observed Data', color='blue', lw=1)
    
    y_label = df_copy[col_Y].name
    plt.title(f"Missing value imputation for {y_label} based on {optimal_n} neighbor \nMissing: {missing_count} ({missing_proportion:.2f}%)")
    plt.ylabel(y_label)
    plt.xlabel(col_X)
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return imputed_series

class MissingValuesWithFlag:
    def Calculate(self, df, value_col, trend_col):
        df_copy = df.copy()
        
        missing_mask = df_copy[value_col].isna()
        
        if missing_mask.sum() == 0:
            return df_copy[value_col], missing_mask
            
        n_neighbors = find_optimal_neighbors(df_copy, value_col, trend_col)
        
        imputed_series = impute_with_specified_neighbors(df_copy, value_col, trend_col, n_neighbors)
        
        return imputed_series, n_neighbors
        
    def CalcAndPlot(self, df, col_Y, col_X, seasonal_period, deviation_threshold=None, DoPlot=1):
        df = df.copy()
        df[col_Y] = pd.to_numeric(df[col_Y], errors='coerce')
        
        original_values = df[col_Y].copy()
        
        initial_interpolation = df[col_Y].interpolate(method='linear', limit_direction='both')
        decomposition = seasonal_decompose(initial_interpolation, period=seasonal_period, model='additive')
        trend = decomposition.trend
        pchip_trend = trend.interpolate(method='pchip')

        if deviation_threshold:
            deviations = (pchip_trend - df[col_Y]).abs()
            exceed_threshold = deviations > deviation_threshold
            if exceed_threshold.any():
                cut_point = exceed_threshold.idxmax()
                pchip_trend[cut_point:] = np.nan

        df['Pchip_Trend'] = pchip_trend

        imputed_data, optimal_neighbors = self.Calculate(df, col_Y, 'Pchip_Trend')
        
        imputed_data[original_values.notna()] = original_values[original_values.notna()]

        df['Dynamic_Imputed_Data'] = imputed_data

        if DoPlot:
            plot_imputed_data(df, col_Y, col_X, seasonal_period, deviation_threshold)

        return df, imputed_data
