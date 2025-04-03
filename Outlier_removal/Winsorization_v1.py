#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Winsorization library for outlier detection and removal

This library provides functions for detecting and handling outliers in time series data using
the winsorization technique. Winsorization replaces extreme values with less extreme values
at specified percentile boundaries, preserving the data distribution while reducing the impact
of outliers.

Main functions:
- winsorize_data: Applies winsorization at three different levels (1-99%, 5-95%, 10-90%)
- plot_winsorized_data: Visualizes original data alongside winsorized versions

Usage:
Import this library and use the winsorize_data function to process time series data,
or use plot_winsorized_data for both processing and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
import os

def winsorize_data(data, column_name):
    raw_data = data[column_name].values
    winsorized_1_99 = winsorize(raw_data, limits=(0.01, 0.01))
    winsorized_5_95 = winsorize(raw_data, limits=(0.05, 0.05))
    winsorized_10_90 = winsorize(raw_data, limits=(0.10, 0.10))
    
    return winsorized_1_99, winsorized_5_95, winsorized_10_90

def plot_winsorized_data(data, column_name, date_column='Date'):
    plt.figure(figsize=(12, 6))
    
    plt.plot(data[date_column], data[column_name], 'b-', label='Original Data', alpha=0.7)
    
    raw_data = data[column_name].values
    winsorized_1_99 = winsorize(raw_data, limits=(0.01, 0.01))
    winsorized_5_95 = winsorize(raw_data, limits=(0.05, 0.05))
    winsorized_10_90 = winsorize(raw_data, limits=(0.10, 0.10))
    
    plt.plot(data[date_column], winsorized_1_99, 'g-', label='Winsorized (1-99%)', linewidth=1.5)
    plt.plot(data[date_column], winsorized_5_95, 'r-', label='Winsorized (5-95%)', linewidth=1.5)
    plt.plot(data[date_column], winsorized_10_90, 'y-', label='Winsorized (10-90%)', linewidth=1.5)
    
    plt.title(f'Winsorization Results - {column_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return winsorized_1_99, winsorized_5_95, winsorized_10_90

if __name__ == "__main__":
    data = pd.read_excel(os.path.join(os.path.dirname(__file__), "..", "Data", "Test_dati_sprosti_2cikls.xlsx"))
    
    if 'Temperature Indoor' in data.columns:
        target_column = 'Temperature Indoor'
    else:
        temp_columns = [col for col in data.columns if 'Temperature' in col]
        data['Average Temperature'] = data[temp_columns].mean(axis=1)
        target_column = 'Average Temperature'
    
    wins_1_99, wins_5_95, wins_10_90 = plot_winsorized_data(data, target_column, 'Date')
    
    print(f"Original data range: [{data[target_column].min():.2f}, {data[target_column].max():.2f}]")
    print(f"Winsorized (1-99%) range: [{min(wins_1_99):.2f}, {max(wins_1_99):.2f}]")
    print(f"Winsorized (5-95%) range: [{min(wins_5_95):.2f}, {max(wins_5_95):.2f}]")
    print(f"Winsorized (10-90%) range: [{min(wins_10_90):.2f}, {max(wins_10_90):.2f}]")
