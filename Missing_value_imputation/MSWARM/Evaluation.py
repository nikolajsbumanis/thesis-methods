#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation of Missing Value Imputation Methods

This module provides comparison and evaluation of various missing value imputation methods
using real data with artificially inserted gaps. It allows assessing the accuracy and
suitability of different methods for specific data types.

Main functions:
- find_missing_regions: Finds regions of missing values in a data series
- introduce_missing_values_randomly: Introduces random gaps in a data series
- introduce_missing_values_in_chunks: Introduces gaps in blocks in a data series
- calculate_performance_metrics: Calculates imputation accuracy metrics

Usage:
Run this script to compare different imputation methods using temperature
measurements from chicken egg production data.
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from MSWARM_v1 import MissingValuesWithFlag
from ARIMA_v1 import impute_missing_values_with_arima
from datetime import datetime

def find_missing_regions(data):
    missing_mask = data.isna()
    regions = []
    start_idx = None
    
    for idx, is_missing in enumerate(missing_mask):
        if is_missing and start_idx is None:
            start_idx = idx
        elif not is_missing and start_idx is not None:
            regions.append((start_idx, idx))
            start_idx = None
    
    if start_idx is not None:
        regions.append((start_idx, len(missing_mask)))
    
    return regions

def introduce_missing_values_randomly(df, column, proportion):
    df_copy = df.copy()
    n_values = len(df_copy)
    n_missing = int(n_values * proportion)
    
    missing_indices = random.sample(range(n_values), n_missing)
    df_copy.loc[missing_indices, column] = np.nan
    
    missing_regions = find_missing_regions(df_copy[column])
    
    return df_copy

def introduce_missing_values_in_chunks(df, column, chunk_size, num_chunks):
    df_copy = df.copy()
    n_values = len(df_copy)
    
    for i in range(num_chunks):
        if n_values <= chunk_size:
            start_idx = 0
        else:
            start_idx = random.randint(0, n_values - chunk_size)
        
        end_idx = min(start_idx + chunk_size, n_values)
        df_copy.iloc[start_idx:end_idx, df_copy.columns.get_loc(column)] = np.nan
    
    missing_count = df_copy[column].isna().sum()
    missing_regions = find_missing_regions(df_copy[column])
    
    return df_copy

file_path = '../../Data/Test_dati_sprosti_2cikls.xlsx'
df_full = pd.read_excel(file_path)
df_full = df_full[(df_full['Date'] >= '2021-03-23') & (df_full['Date'] <= '2021-10-09')]

target_column = 'Average Temperature Indoor'
df_full[target_column] = df_full[['Temperature 1 floor', 'Temperature 8 floor', 'Temperature computer']].mean(axis=1)

original_values = df_full[target_column].copy()

random.seed(42)
df_with_gaps = introduce_missing_values_in_chunks(df_full.copy(), target_column, chunk_size=5, num_chunks=10)

plt.figure(figsize=(15, 6))
plt.plot(df_full['Date'], original_values, 'b-', label='Original data')
plt.plot(df_with_gaps['Date'], df_with_gaps[target_column], 'r-', label='Data with gaps')
plt.title('Original vs Data with Gaps')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

df_poly_interpolate = df_with_gaps.copy()
df_poly_interpolate[target_column] = df_poly_interpolate[target_column].interpolate(method='polynomial', order=3)

mvi = MissingValuesWithFlag()
_, imputed_mswarm = mvi.CalcAndPlot(df_with_gaps.copy(), target_column, 'Date', seasonal_period=30, DoPlot=0)

df_arima = df_with_gaps.copy()
df_arima = impute_missing_values_with_arima(df_arima, target_column, 'Date')

def calculate_performance_metrics(true_values, imputed_values):
    mask = df_with_gaps[target_column].isna()
    
    mae = mean_absolute_error(true_values[mask], imputed_values[mask])
    mse = mean_squared_error(true_values[mask], imputed_values[mask])
    rmse = np.sqrt(mse)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

performance_metrics = {}

for method_name, df_imputed in [
    ("Polynomial Interpolation", df_poly_interpolate[target_column]),
    ("Modified Standard Weighted Average Robust Method", imputed_mswarm),
    ("ARIMA Method", df_arima[target_column])
]:
    metrics = calculate_performance_metrics(original_values, df_imputed)
    performance_metrics[method_name] = metrics

def main():
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(df_full['Date'], original_values, 'b-', label='Original')
    plt.plot(df_full['Date'], df_poly_interpolate[target_column], 'r-', label='Polynomial')
    plt.title('Polynomial Interpolation')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(df_full['Date'], original_values, 'b-', label='Original')
    plt.plot(df_full['Date'], imputed_mswarm, 'g-', label='Modified Standard Weighted Average Robust Method')
    plt.title('Modified Standard Weighted Average Robust Method')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(df_full['Date'], original_values, 'b-', label='Original')
    plt.plot(df_full['Date'], df_arima[target_column], 'y-', label='ARIMA')
    plt.title('ARIMA Method')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    methods = list(performance_metrics.keys())
    mae_values = [performance_metrics[method]['MAE'] for method in methods]
    rmse_values = [performance_metrics[method]['RMSE'] for method in methods]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(methods, mae_values)
    plt.title('Mean Absolute Error (MAE)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    plt.subplot(1, 2, 2)
    plt.bar(methods, rmse_values)
    plt.title('Root Mean Squared Error (RMSE)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
