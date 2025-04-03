#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Outliers library for detecting and removing outliers in time series data

This library provides functions for detecting and removing outliers in time series data using
multiple advanced techniques. The methods adapt to data characteristics and can preserve
specified data ranges when needed.

Main functions:
- detect_outliers: Identifies outliers using multiple detection approaches
- clean_outliers: Removes outliers and replaces them with trend-based values
- calculate_trend: Calculates robust trend lines for data
- OutliersV2: Class implementation for easier usage

Usage:
Import this library and use either the functional approach with detect_outliers and
clean_outliers or the object-oriented approach with the OutliersV2 class.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from scipy.signal import savgol_filter
from Rolling_window_v1 import rolling_window_sd
from Winsorization_v1 import winsorize_data
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Tuple, Optional, Union
import os

def calculate_trend(data, window_length=31):
    valid_data = data.dropna()
    min_val = valid_data.min()
    max_val = valid_data.max()
    
    valid_indices = ~data.isna()
    trend = np.full_like(data.values, np.nan, dtype=float)
    
    if len(valid_data) > window_length:
        rolling_med = valid_data.rolling(window=window_length, center=True, min_periods=3).median()
        
        rolling_med = rolling_med.bfill().ffill()
        
        trend_smooth = savgol_filter(
            rolling_med.values, 
            window_length=min(window_length*2+1, len(rolling_med)//2*2+1), 
            polyorder=3
        )
        
        trend[valid_indices] = trend_smooth
        
        trend[trend < min_val] = min_val
        trend[trend > max_val] = max_val
    else:
        trend[valid_indices] = valid_data.median()
    
    return pd.Series(trend, index=data.index)

def detect_extreme_outliers(data, column_name, percentiles=(2.5, 97.5), keep_ranges=None):
    series = data[column_name].copy()
    
    if keep_ranges is not None:
        mask = np.zeros(len(series), dtype=bool)
        for start, end in keep_ranges:
            mask[start:end] = True
        
        working_series = series[~mask].copy()
        
        lower_bound = np.nanpercentile(working_series, percentiles[0])
        upper_bound = np.nanpercentile(working_series, percentiles[1])
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_mask[mask] = False
    else:
        lower_bound = np.nanpercentile(series, percentiles[0])
        upper_bound = np.nanpercentile(series, percentiles[1])
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
    
    winsorized = series.copy()
    winsorized[outlier_mask & (series < lower_bound)] = lower_bound
    winsorized[outlier_mask & (series > upper_bound)] = upper_bound
    
    return outlier_mask, winsorized

def detect_local_outliers_multi_window(data, window_sizes=[7, 15, 31], z_threshold=1.5, use_mad=True):
    outlier_mask = np.zeros(len(data), dtype=bool)
    detection_counts = np.zeros(len(data))
    
    for window_size in window_sizes:
        if len(data) < window_size:
            continue
            
        half_window = window_size // 2
        
        for i in range(len(data)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(data), i + half_window + 1)
            
            window_data = data.iloc[start_idx:end_idx].copy()
            
            if window_data.isna().all():
                continue
                
            window_data = window_data.dropna()
            
            if len(window_data) < 3:
                continue
                
            current_value = data.iloc[i]
            
            if pd.isna(current_value):
                continue
                
            if use_mad:
                center = window_data.median()
                mad = np.median(np.abs(window_data - center))
                
                if mad == 0:
                    std_dev = window_data.std()
                    if std_dev == 0:
                        continue
                    z_score = abs(current_value - center) / std_dev
                else:
                    z_score = abs(current_value - center) / (mad * 1.4826)
            else:
                mean = window_data.mean()
                std_dev = window_data.std()
                
                if std_dev == 0:
                    continue
                    
                z_score = abs(current_value - mean) / std_dev
            
            if z_score > z_threshold:
                outlier_mask[i] = True
                detection_counts[i] += 1
                
        trend = calculate_trend(data, window_size)
        detrended = data - trend
        
        if use_mad:
            center = detrended.median()
            mad = np.median(np.abs(detrended.dropna() - center))
            scale = mad * 1.4826
        else:
            center = detrended.mean()
            scale = detrended.std()
            
        if scale > 0:
            z_scores = np.abs(detrended - center) / scale
            trend_outliers = z_scores > z_threshold * 1.5
            
            outlier_mask = outlier_mask | trend_outliers
            detection_counts[trend_outliers] += 0.5
    
    return outlier_mask, detection_counts

def detect_outliers(data, column_name, window_sizes=None, z_threshold=None, keep_ranges=None, use_robust_stats=True):
    series = data[column_name].copy()
    
    if window_sizes is None or z_threshold is None:
        window_sizes, z_threshold = _get_adaptive_params(series, window_sizes, z_threshold)
    
    extreme_mask, _ = detect_extreme_outliers(
        data, 
        column_name, 
        percentiles=(1.0, 99.0), 
        keep_ranges=keep_ranges
    )
    
    local_mask, detection_counts = detect_local_outliers_multi_window(
        series,
        window_sizes=window_sizes,
        z_threshold=z_threshold,
        use_mad=use_robust_stats
    )
    
    combined_mask = extreme_mask | local_mask
    
    if keep_ranges is not None:
        for start, end in keep_ranges:
            combined_mask[start:end] = False
    
    outliers = pd.Series(False, index=series.index)
    outliers[combined_mask] = True
    
    return outliers

def clean_outliers(data, column_name, window_sizes=None, z_threshold=None, keep_ranges=None, use_robust_stats=True):
    df = data.copy()
    series = df[column_name]
    
    outliers = detect_outliers(
        df, 
        column_name, 
        window_sizes=window_sizes,
        z_threshold=z_threshold,
        keep_ranges=keep_ranges,
        use_robust_stats=use_robust_stats
    )
    
    if not outliers.any():
        return df, outliers
    
    cleaned_series = series.copy()
    
    if window_sizes is None:
        window_sizes, _ = _get_adaptive_params(series, window_sizes, z_threshold)
    
    trend = calculate_trend(series, window_length=max(window_sizes))
    
    for i, is_outlier in enumerate(outliers):
        if is_outlier:
            cleaned_series.iloc[i] = trend.iloc[i]
    
    df[column_name] = cleaned_series
    
    return df, outliers

def _get_adaptive_params(data, base_window=None, z_threshold=None):
    series_length = len(data)
    
    if base_window is None:
        if series_length < 30:
            base_window = 5
        elif series_length < 100:
            base_window = 7
        elif series_length < 365:
            base_window = 15
        else:
            base_window = 31
    
    window_sizes = [
        max(3, int(base_window * 0.5)),
        base_window,
        min(series_length // 2, int(base_window * 2))
    ]
    
    if z_threshold is None:
        if series_length < 30:
            z_threshold = 2.0
        elif series_length < 100:
            z_threshold = 2.5
        else:
            z_threshold = 3.0
    
    return window_sizes, z_threshold

def plot_results(data, cleaned_data, outliers, title=None):
    plt.figure(figsize=(12, 6))
    
    if isinstance(data, pd.DataFrame):
        x = data.index
        y_orig = data.iloc[:, 0]
        y_clean = cleaned_data.iloc[:, 0]
    else:
        x = data.index
        y_orig = data
        y_clean = cleaned_data
    
    plt.plot(x, y_orig, 'b-', label='Original Data', alpha=0.5)
    plt.plot(x, y_clean, 'g-', label='Cleaned Data')
    
    if outliers.any():
        plt.scatter(x[outliers], y_orig[outliers], color='r', label='Outliers', s=50)
    
    plt.title(title or 'Outlier Detection and Cleaning Results')
    plt.legend()
    plt.tight_layout()
    plt.show()

class OutliersV2:
    def __init__(self, window_sizes=[7, 15, 31], z_threshold=3.0, use_robust_stats=True, keep_ranges=None):
        self.window_sizes = window_sizes
        self.z_threshold = z_threshold
        self.use_robust_stats = use_robust_stats
        self.keep_ranges = keep_ranges
        self.outliers = None
        self.cleaned_data = None
    
    def detect(self, data):
        if isinstance(data, pd.DataFrame):
            series = data.iloc[:, 0]
        else:
            series = data
            
        df = pd.DataFrame({'value': series})
        self.outliers = detect_outliers(
            df, 
            'value', 
            window_sizes=self.window_sizes,
            z_threshold=self.z_threshold,
            keep_ranges=self.keep_ranges,
            use_robust_stats=self.use_robust_stats
        )
        
        return self.outliers
    
    def clean(self, data):
        if isinstance(data, pd.DataFrame):
            series = data.iloc[:, 0]
            df = data.copy()
            column_name = df.columns[0]
        else:
            series = data
            df = pd.DataFrame({'value': series})
            column_name = 'value'
        
        cleaned_df, self.outliers = clean_outliers(
            df,
            column_name,
            window_sizes=self.window_sizes,
            z_threshold=self.z_threshold,
            keep_ranges=self.keep_ranges,
            use_robust_stats=self.use_robust_stats
        )
        
        if isinstance(data, pd.DataFrame):
            self.cleaned_data = cleaned_df
        else:
            self.cleaned_data = cleaned_df[column_name]
            
        return self.cleaned_data
    
    def plot(self, data=None, title=None):
        if data is None and self.cleaned_data is None:
            raise ValueError("No data to plot. Call clean() first or provide data.")
            
        if data is not None and self.cleaned_data is None:
            self.clean(data)
            
        plot_results(
            data if data is not None else self.cleaned_data, 
            self.cleaned_data, 
            self.outliers, 
            title
        )

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    
    data = pd.read_excel(os.path.join(os.path.dirname(__file__), "..", "Data", "Test_dati_sprosti_2cikls.xlsx"))
    
    if 'Temperature Indoor' in data.columns:
        target_column = 'Temperature Indoor'
    else:
        temp_columns = [col for col in data.columns if 'Temperature' in col]
        data['Average Temperature'] = data[temp_columns].mean(axis=1)
        target_column = 'Average Temperature'
    
    detector = OutliersV2()
    cleaned_data = detector.clean(data[target_column])
    detector.plot(data[target_column], "Outlier Detection Results")
    
    print(f"Detected {detector.outliers.sum()} outliers")
