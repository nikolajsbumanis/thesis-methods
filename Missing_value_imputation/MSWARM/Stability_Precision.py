#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stability and Precision Analysis of MSWARM (Modified Standard Weighted Average Robust Method) Imputation Method

This module provides statistical analysis and precision evaluation for missing value
imputation methods. It allows assessing the stability and precision of the MSWARM 
(Modified Standard Weighted Average Robust Method) in various scenarios using different 
statistical indicators and visualizations.

Main functions:
- calculate_basic_stats: Calculates basic statistical indicators
- create_missing_values: Creates missing values randomly or in blocks
- analyze_scenario_multiple: Analyzes a scenario multiple times and calculates average results
- plot_combined_analysis: Creates combined charts for all scenarios

Usage:
Customize the configuration and run the script to perform a complete analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime
import os
from MSWARM_v1 import MissingValuesWithFlag

CONFIG = {
    'data': {
        'file_path': "../../Data/Test_dati_sprosti_2cikls.xlsx",
        'target_column': "Average Temperature Indoor",
        'date_column': "Date"
    },
    
    'analysis': {
        'n_iterations': 10,
        'max_lag': 24,
        'scenarios': [
            {
                'name': 'random_5pct',
                'description': 'Random 5% missing',
                'type': 'random',
                'proportion': 0.05
            },
            {
                'name': 'random_10pct',
                'description': 'Random 10% missing',
                'type': 'random',
                'proportion': 0.10
            },
            {
                'name': 'random_20pct',
                'description': 'Random 20% missing',
                'type': 'random',
                'proportion': 0.20
            },
            {
                'name': 'chunks_small',
                'description': 'Small chunks (3 points, 5 chunks)',
                'type': 'chunks',
                'chunk_size': 3,
                'num_chunks': 5
            },
            {
                'name': 'chunks_medium',
                'description': 'Medium chunks (5 points, 5 chunks)',
                'type': 'chunks',
                'chunk_size': 5,
                'num_chunks': 5
            },
            {
                'name': 'chunks_large',
                'description': 'Large chunks (10 points, 3 chunks)',
                'type': 'chunks',
                'chunk_size': 10,
                'num_chunks': 3
            }
        ]
    },
    
    'output': {
        'base_dir': 'results',
        'language': 'lv'
    }
}

LABELS = {
    'lv': {
        'original_data': 'Oriģinālie dati',
        'imputed_data': 'Imputētie dati',
        'values': 'Vērtības',
        'date': 'Datums',
        'temperature': 'Temperatūra (°C)',
        'scenario': 'Scenārijs',
        'autocorrelation': 'Autokorelācija',
        'lag': 'Nobīde',
        'distribution': 'Sadalījums',
        'density': 'Blīvums',
        'qq_plot': 'Q-Q grafiks',
        'theoretical_quantiles': 'Teorētiskie kvantili',
        'sample_quantiles': 'Parauga kvantili',
        'histogram': 'Histogramma',
        'frequency': 'Biežums',
        'residuals': 'Atlikumi',
        'residual_plot': 'Atlikumu grafiks',
        'predicted_values': 'Prognozētās vērtības',
        'mean': 'Vidējā vērtība',
        'median': 'Mediāna',
        'std_dev': 'Standartnovirze',
        'min': 'Minimums',
        'max': 'Maksimums',
        'skewness': 'Asimetrija',
        'kurtosis': 'Ekscesa koeficients',
        'mae': 'MAE',
        'mse': 'MSE',
        'rmse': 'RMSE',
        'mape': 'MAPE (%)',
        'r_squared': 'R²',
        'stability_analysis': 'Stabilitātes analīze',
        'precision_analysis': 'Precizitātes analīze',
        'missing_value_imputation': 'Trūkstošo vērtību imputācija',
        'analysis_results': 'Analīzes rezultāti',
        'metrics': 'Metriku salīdzinājums',
        'metric': 'Metrika',
        'value': 'Vērtība',
        'autocorr_diff': 'Autokorelācijas starpība',
        'dist_diff': 'Sadalījuma starpība',
        'outlier': 'novirze'
    },
    'en': {
        'original_data': 'Original data',
        'imputed_data': 'Imputed data',
        'values': 'Values',
        'date': 'Date',
        'temperature': 'Temperature (°C)',
        'scenario': 'Scenario',
        'autocorrelation': 'Autocorrelation',
        'lag': 'Lag',
        'distribution': 'Distribution',
        'density': 'Density',
        'qq_plot': 'Q-Q Plot',
        'theoretical_quantiles': 'Theoretical Quantiles',
        'sample_quantiles': 'Sample Quantiles',
        'histogram': 'Histogram',
        'frequency': 'Frequency',
        'residuals': 'Residuals',
        'residual_plot': 'Residual Plot',
        'predicted_values': 'Predicted Values',
        'mean': 'Mean',
        'median': 'Median',
        'std_dev': 'Std. Deviation',
        'min': 'Minimum',
        'max': 'Maximum',
        'skewness': 'Skewness',
        'kurtosis': 'Kurtosis',
        'mae': 'MAE',
        'mse': 'MSE',
        'rmse': 'RMSE',
        'mape': 'MAPE (%)',
        'r_squared': 'R²',
        'stability_analysis': 'Stability Analysis',
        'precision_analysis': 'Precision Analysis',
        'missing_value_imputation': 'Missing Value Imputation',
        'analysis_results': 'Analysis Results',
        'metrics': 'Metrics Comparison',
        'metric': 'Metric',
        'value': 'Value',
        'autocorr_diff': 'Autocorrelation Difference',
        'dist_diff': 'Distribution Difference',
        'outlier': 'Outlier'
    }
}

def setup_output_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CONFIG['output']['base_dir'], f"analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def calculate_basic_stats(original_series, imputed_series):
    stats = {}
    
    mask = original_series.isna()
    
    if mask.sum() == 0:
        return None
    
    stats['mae'] = np.mean(np.abs(original_series[~mask] - imputed_series[~mask]))
    stats['mse'] = np.mean((original_series[~mask] - imputed_series[~mask])**2)
    stats['rmse'] = np.sqrt(stats['mse'])
    stats['mape'] = np.mean(np.abs((original_series[~mask] - imputed_series[~mask]) / original_series[~mask])) * 100
    
    return stats

def create_missing_values(df, column, proportion=None, chunk_size=None, num_chunks=None):
    df_copy = df.copy()
    
    if proportion is not None:
        n_values = len(df_copy)
        n_missing = int(n_values * proportion)
        missing_indices = np.random.choice(df_copy.index, size=n_missing, replace=False)
        df_copy.loc[missing_indices, column] = np.nan
    
    elif chunk_size is not None and num_chunks is not None:
        n_values = len(df_copy)
        
        for _ in range(num_chunks):
            start_idx = np.random.randint(0, n_values - chunk_size + 1)
            end_idx = start_idx + chunk_size
            df_copy.iloc[start_idx:end_idx, df_copy.columns.get_loc(column)] = np.nan
    
    return df_copy

def plot_combined_analysis(scenarios_data, output_dir, plot_type):
    lang = CONFIG['output']['language']
    labels = LABELS[lang]
    
    if plot_type == 'metrics':
        metrics = ['mae', 'rmse', 'mape']
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        
        for i, metric in enumerate(metrics):
            values = [data['stats'][metric] for data in scenarios_data]
            scenarios = [data['description'] for data in scenarios_data]
            
            axes[i].bar(scenarios, values)
            axes[i].set_title(f"{labels[metric]}")
            axes[i].set_ylabel(labels['value'])
            axes[i].set_xlabel(labels['scenario'])
            axes[i].grid(axis='y')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_metrics.png'), dpi=300)
        plt.close()
    
    elif plot_type == 'autocorr':
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for data in scenarios_data:
            ax.plot(data['autocorr_diff'], label=data['description'])
        
        ax.set_title(labels['autocorr_diff'])
        ax.set_xlabel(labels['lag'])
        ax.set_ylabel(labels['value'])
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_autocorr.png'), dpi=300)
        plt.close()

def analyze_scenario_multiple(file_path, target_column, scenario_desc, output_dir, n_iterations=5, proportion=None, chunk_size=None, num_chunks=None):
    df_full = pd.read_excel(file_path)
    df_full = df_full[(df_full['Date'] >= '2021-03-23') & (df_full['Date'] <= '2021-10-09')]
    
    if 'Average Temperature Indoor' not in df_full.columns:
        df_full['Average Temperature Indoor'] = df_full[['Temperature 1 floor', 'Temperature 8 floor', 'Temperature computer']].mean(axis=1)
    
    target_column = 'Average Temperature Indoor'
    
    stats_list = []
    autocorr_diffs_list = []
    
    for i in range(n_iterations):
        df_with_gaps = create_missing_values(df_full, target_column, proportion, chunk_size, num_chunks)
        
        original_series = df_full[target_column]
        
        mvi = MissingValuesWithFlag()
        _, imputed_series = mvi.CalcAndPlot(df_with_gaps, target_column, 'Date', seasonal_period=30, DoPlot=0)
        
        stats = calculate_basic_stats(original_series, imputed_series)
        if stats:
            stats_list.append(stats)
        
        autocorr_diff = calculate_autocorr_diffs(original_series, imputed_series)
        autocorr_diffs_list.append(autocorr_diff)
    
    avg_stats = {k: np.mean([s[k] for s in stats_list]) for k in stats_list[0].keys()}
    avg_autocorr_diff = np.mean(autocorr_diffs_list, axis=0)
    
    return {
        'description': scenario_desc,
        'stats': avg_stats,
        'autocorr_diff': avg_autocorr_diff
    }

def calculate_autocorr_diffs(original, imputed):
    max_lag = CONFIG['analysis']['max_lag']
    orig_autocorr = sm.tsa.acf(original.dropna(), nlags=max_lag)
    imp_autocorr = sm.tsa.acf(imputed.dropna(), nlags=max_lag)
    return np.abs(orig_autocorr - imp_autocorr)

def save_results_to_excel(scenarios_data, output_dir):
    lang = CONFIG['output']['language']
    labels = LABELS[lang]
    
    writer = pd.ExcelWriter(os.path.join(output_dir, 'analysis_results.xlsx'), engine='xlsxwriter')
    
    summary_data = []
    for scenario in scenarios_data:
        row = {'scenario': scenario['description']}
        row.update(scenario['stats'])
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    autocorr_data = {}
    for scenario in scenarios_data:
        autocorr_data[scenario['description']] = scenario['autocorr_diff']
    
    autocorr_df = pd.DataFrame(autocorr_data)
    autocorr_df.index.name = labels['lag']
    
    autocorr_df.to_excel(writer, sheet_name='Autocorrelation')
    
    field_desc = get_field_descriptions()
    desc_df = pd.DataFrame(list(field_desc.items()), columns=['Field', 'Description'])
    desc_df.to_excel(writer, sheet_name='Field Descriptions', index=False)
    
    workbook = writer.book
    
    format_header = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC'})
    format_percent = workbook.add_format({'num_format': '0.00%'})
    
    for sheet_name in writer.sheets:
        worksheet = writer.sheets[sheet_name]
        for col_num, value in enumerate(worksheet.table[0]):
            worksheet.write(0, col_num, value, format_header)
    
    writer.close()

def get_field_descriptions():
    return {
        'scenario': 'Description of the missing value scenario',
        'mae': 'Mean Absolute Error - average of absolute differences between original and imputed values',
        'mse': 'Mean Squared Error - average of squared differences between original and imputed values',
        'rmse': 'Root Mean Squared Error - square root of MSE',
        'mape': 'Mean Absolute Percentage Error - average of absolute percentage differences',
        'r_squared': 'Coefficient of determination - proportion of variance explained by the model',
        'skewness': 'Measure of asymmetry of the probability distribution',
        'kurtosis': 'Measure of "tailedness" of the probability distribution',
        'autocorr_diff': 'Absolute difference between autocorrelation functions of original and imputed data',
        'outlier': 'Data points that significantly deviate from the normal pattern of the dataset'
    }

def print_analysis_results(scenarios_data):
    results = {}
    
    for scenario in scenarios_data:
        scenario_results = {}
        
        for metric, value in scenario['stats'].items():
            scenario_results[metric] = value
        
        scenario_results['autocorr_diff_avg'] = np.mean(scenario['autocorr_diff'])
        results[scenario['description']] = scenario_results
    
    return results

def run_analysis():
    output_dir = setup_output_directory()
    
    file_path = CONFIG['data']['file_path']
    target_column = CONFIG['data']['target_column']
    n_iterations = CONFIG['analysis']['n_iterations']
    
    scenarios_data = []
    
    for scenario in CONFIG['analysis']['scenarios']:
        if scenario['type'] == 'random':
            result = analyze_scenario_multiple(
                file_path, 
                target_column,
                scenario['description'],
                output_dir,
                n_iterations=n_iterations,
                proportion=scenario['proportion']
            )
        elif scenario['type'] == 'chunks':
            result = analyze_scenario_multiple(
                file_path, 
                target_column,
                scenario['description'],
                output_dir,
                n_iterations=n_iterations,
                chunk_size=scenario['chunk_size'],
                num_chunks=scenario['num_chunks']
            )
        
        scenarios_data.append(result)
    
    analysis_results = print_analysis_results(scenarios_data)
    
    plot_combined_analysis(scenarios_data, output_dir, 'metrics')
    plot_combined_analysis(scenarios_data, output_dir, 'autocorr')
    
    save_results_to_excel(scenarios_data, output_dir)
    
    return output_dir, analysis_results

if __name__ == "__main__":
    run_analysis()
