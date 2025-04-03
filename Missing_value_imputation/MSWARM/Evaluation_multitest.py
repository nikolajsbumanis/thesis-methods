#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from datetime import datetime
from MSWARM_v1 import MissingValuesWithFlag
from ARIMA_v1 import impute_missing_values_with_arima

class SimpleMissingValuesWithFlag:
    @staticmethod
    def CalcAndPlot(df, target_column, date_column, seasonal_period=None, deviation_threshold=None, DoPlot=0):
        df_result = df.copy()
        df_result[target_column] = df_result[target_column].interpolate(method='linear')
        return df_result, df_result[target_column], None

def introduce_missing_values_in_chunks(df, target_column, chunk_size, num_chunks):
    df_copy = df.copy()
    n_values = len(df_copy)
    
    valid_starts = list(range(0, n_values - chunk_size + 1))
    
    if not valid_starts:
        return df_copy  
        
    if len(valid_starts) < num_chunks:
        num_chunks = len(valid_starts)  
        
    chunk_starts = np.random.choice(valid_starts, size=num_chunks, replace=False)
    
    for start in chunk_starts:
        chunk_indices = df_copy.index[start:start + chunk_size]
        df_copy.loc[chunk_indices, target_column] = np.nan
    
    return df_copy

def introduce_missing_values_randomly(df, target_column, proportion):
    df_copy = df.copy()
    n_values = len(df_copy)
    n_missing = int(n_values * proportion)
    
    valid_indices = df_copy[~df_copy[target_column].isna()].index.tolist()
    
    if n_missing > len(valid_indices):
        n_missing = len(valid_indices)  
        
    if valid_indices:  
        missing_indices = np.random.choice(valid_indices, size=n_missing, replace=False)
        df_copy.loc[missing_indices, target_column] = np.nan
    
    return df_copy

def impute_missing_values_with_arima(df, column, date_column):
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=date_column)
    
    if not df[column].isna().any():
        return df
    
    values = df[column].copy()
    
    values = values.interpolate(method='linear')
    
    try:
        model = ARIMA(values, order=(2,1,2))
        results = model.fit()
        df[column] = results.fittedvalues
    except Exception as e:
        df[column] = df[column].interpolate(method='linear')
    
    return df

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

def determine_chunk_position(start_idx, total_rows, chunk_size):
    third = total_rows / 3
    if start_idx < third:
        return "start"
    elif start_idx > (2 * third):
        return "end"
    else:
        return "middle"

def calculate_metrics(true_values, imputed_values):
    mask = ~(pd.isna(true_values) | pd.isna(imputed_values))
    true_vals = true_values[mask]
    imp_vals = imputed_values[mask]
    
    if len(true_vals) == 0:
        return np.nan, np.nan, np.nan
    
    mse = mean_squared_error(true_vals, imp_vals)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_vals, imp_vals)
    
    return mse, rmse, mae

def calculate_all_method_metrics(df, df_modified, target_column, date_column):
    df_arima = impute_missing_values_with_arima(df_modified, target_column, date_column)
    
    _, dynamic_values, _ = MissingValuesWithFlag.CalcAndPlot(df_modified, target_column, date_column, 30, deviation_threshold=5, DoPlot=0)
    df_dynamic = pd.DataFrame({
        date_column: df_modified[date_column],
        target_column: dynamic_values
    })
    
    df_poly = df_modified.copy()
    df_poly[target_column] = df_poly[target_column].interpolate(method='polynomial', order=2)
    
    true_values = df[target_column]
    methods = {
        'ARIMA': df_arima[target_column],
        'Dynamic': df_dynamic[target_column],
        'Polynomial': df_poly[target_column]
    }
    
    metrics = {}
    for method_name, imputed_values in methods.items():
        mask = df_modified[target_column].isna()
        true_vals = true_values[mask]
        imp_vals = imputed_values[mask]
        metrics[method_name] = calculate_metrics(true_vals, imp_vals)
    
    return metrics

def perform_comprehensive_tests(df, target_column, date_column, num_instances=3):
    chunk_sizes = [2,4,6,8,10,12]
    chunk_numbers = [1, 2, 3, 4, 5]
    proportions = np.arange(0, 0.22, 0.02)  
    
    results = []
    test_counter = 0
    
    total_combinations = len(chunk_sizes) * len(chunk_numbers) * len(proportions) * num_instances
    
    for chunk_size in chunk_sizes:
        for num_chunks in chunk_numbers:
            for prop in proportions:
                for instance in range(num_instances):
                    test_counter += 1
                    
                    df_modified = df.copy()
                    
                    df_modified = introduce_missing_values_in_chunks(df_modified, target_column, chunk_size, num_chunks)
                    
                    non_missing_mask = ~df_modified[target_column].isna()
                    remaining_data = df_modified[non_missing_mask].copy()
                    if len(remaining_data) > 0:
                        remaining_modified = introduce_missing_values_randomly(remaining_data, target_column, prop)
                        df_modified.loc[remaining_modified.index, target_column] = remaining_modified[target_column]
                    
                    total_missing_pct = (df_modified[target_column].isna().sum() / len(df_modified)) * 100
                    
                    metrics = calculate_all_method_metrics(df, df_modified, target_column, date_column)
                    
                    for method_name, (mse, rmse, mae) in metrics.items():
                        results.append({
                            'test_id': test_counter,
                            'instance': instance + 1,
                            'method': method_name,
                            'chunk_size': chunk_size,
                            'num_chunks': num_chunks,
                            'base_proportion': prop,
                            'total_missing_pct': total_missing_pct,
                            'mse': mse,
                            'rmse': rmse,
                            'mae': mae
                        })
    
    results_df = pd.DataFrame(results)
    
    deterioration_analysis = analyze_performance_deterioration(results_df)
    
    return results_df, deterioration_analysis

def analyze_performance_deterioration(results_df):
    analysis = {}
    
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        
        missing_analysis = method_data.groupby('total_missing_pct')[['mse', 'rmse', 'mae']].mean()
        missing_analysis = missing_analysis.sort_index()
        
        for metric in ['mse', 'rmse', 'mae']:
            values = missing_analysis[metric].values
            deterioration_point = find_deterioration_point(values)
            if deterioration_point is not None:
                threshold_pct = missing_analysis.index[deterioration_point]
                analysis.setdefault(method, {})[f'{metric}_threshold'] = threshold_pct
    
    return analysis

def find_deterioration_point(values, threshold_factor=1.5):
    if len(values) < 2:
        return None
    
    baseline = values[0]
    for i in range(1, len(values)):
        if values[i] > baseline * threshold_factor:
            return i
    return None

def save_test_results(results_df, deterioration_analysis, output_dir='Tests'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    detailed_file = os.path.join(output_dir, f'detailed_results_{timestamp}.xlsx')
    results_df.to_excel(detailed_file, index=False)
    
    mean_file = os.path.join(output_dir, f'mean_results_{timestamp}.xlsx')
    with pd.ExcelWriter(mean_file) as writer:
        config_means = results_df.groupby(
            ['method', 'chunk_size', 'num_chunks', 'base_proportion']
        )[['mse', 'rmse', 'mae']].mean()
        config_means.to_excel(writer, sheet_name='Configuration_Means')
        
        method_means = results_df.groupby('method')[['mse', 'rmse', 'mae']].mean()
        method_means.to_excel(writer, sheet_name='Method_Means')
        
        deterioration_df = pd.DataFrame(deterioration_analysis).T
        deterioration_df.to_excel(writer, sheet_name='Deterioration_Analysis')
    
    return detailed_file, mean_file

def main():
    file_path = '../../Data/Test_dati_sprosti_2cikls.xlsx'
    df = pd.read_excel(file_path)
    
    if 'Temperature Indoor' in df.columns:
        target_column = 'Temperature Indoor'
    elif 'Average Temperature Indoor' in df.columns:
        target_column = 'Average Temperature Indoor'
    else:
        temp_columns = [col for col in df.columns if 'Temperature' in col]
        df['Average Temperature Indoor'] = df[temp_columns].mean(axis=1)
        target_column = 'Average Temperature Indoor'
    
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    
    results_df, deterioration_analysis = perform_comprehensive_tests(df, target_column, 'Date')
    
    detailed_file, mean_file = save_test_results(results_df, deterioration_analysis)
    
    return results_df, deterioration_analysis

if __name__ == "__main__":
    main()
