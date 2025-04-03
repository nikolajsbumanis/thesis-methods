#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rolling Window library for outlier detection and removal

This library provides functions for detecting and removing outliers in time series data using
a rolling window approach with standard deviation thresholds. The method identifies values that
deviate significantly from local trends and replaces them with more appropriate values.

Main functions:
- rolling_window_sd: Detects and replaces outliers using rolling window statistics
- plot_rolling_window: Visualizes original data, processed data, and identified outliers

Usage:
Import this library and use the rolling_window_sd function to process time series data,
or use plot_rolling_window for both processing and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def rolling_window_sd(data, column_name, window_size=7, threshold=2):
    series = data[column_name].astype(float).copy()
    rolling_mean = series.rolling(window=window_size).mean()
    rolling_std = series.rolling(window=window_size).std()
    outlier_mask = np.abs(series - rolling_mean) > threshold * rolling_std
    series[outlier_mask] = rolling_mean[outlier_mask]
    return series

def plot_rolling_window(data, column_name, date_column='Date'):
    series = data[column_name].copy()
    original = series.copy()
    processed = rolling_window_sd(data, column_name)
    plt.figure(figsize=(12, 6))
    plt.plot(data[date_column], original, 'b-', label='Original')
    plt.plot(data[date_column], processed, 'r-', label='Processed')
    outlier_mask = original != processed
    plt.scatter(data.loc[outlier_mask, date_column], original[outlier_mask], color='green', s=50, label='Outliers')
    plt.title(f'Rolling Window Method - {column_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return processed

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    
    data = pd.read_excel(os.path.join(os.path.dirname(__file__), "..", "Data", "Test_dati_sprosti_2cikls.xlsx"))
    
    if 'Temperature Indoor' in data.columns:
        target_column = 'Temperature Indoor'
    else:
        temp_columns = [col for col in data.columns if 'Temperature' in col]
        data['Average Temperature'] = data[temp_columns].mean(axis=1)
        target_column = 'Average Temperature'
    
    processed_data = plot_rolling_window(data, target_column, 'Date')
    print(f"Detected and processed {sum(data[target_column] != processed_data)} outliers")
