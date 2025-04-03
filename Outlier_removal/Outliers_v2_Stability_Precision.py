#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Outliers Stability and Precision Evaluation

This module provides statistical analysis and precision evaluation for outlier detection methods.
It systematically tests different parameter configurations to assess the stability, precision,
and overall performance of the outlier detection algorithms across various scenarios.

Main functions:
- test_outlier_scenarios: Tests different outlier scenarios with varying parameters
- execute_parameter_analysis: Performs comprehensive parameter sensitivity analysis
- calculate_detection_metrics: Evaluates precision, recall, and F1 score for detection
- plot_scenario_heatmap: Visualizes results as heatmaps for easy interpretation

Usage:
Run this script directly to perform comprehensive testing of outlier detection methods
using temperature measurements from chicken egg production data.
"""

import os
import sys
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import colorsys
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import warnings

from Outliers_v2 import detect_outliers, clean_outliers, plot_results, OutliersV2

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams['font.family'] = 'DejaVu Sans'

def calculate_detection_metrics(true_outliers, detected_outliers):
    mask = ~(true_outliers.isna() | detected_outliers.isna())
    true_outliers = true_outliers[mask]
    detected_outliers = detected_outliers[mask]
    
    if len(true_outliers) == 0 or len(detected_outliers) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    precision = precision_score(true_outliers, detected_outliers)
    recall = recall_score(true_outliers, detected_outliers)
    f1 = f1_score(true_outliers, detected_outliers)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def calculate_cleaning_metrics(original_series, cleaned_series):
    mask = ~(original_series.isna() | cleaned_series.isna())
    original_series = original_series[mask]
    cleaned_series = cleaned_series[mask]
    
    if len(original_series) == 0 or len(cleaned_series) == 0:
        return {'mse': 0.0, 'rmse': 0.0, 'mae': 0.0}
    
    mse = mean_squared_error(original_series, cleaned_series)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(original_series, cleaned_series)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }

def calculate_basic_stats(original_series, modified_series):
    original = pd.Series(original_series).dropna()
    modified = pd.Series(modified_series).dropna()
    
    orig_stats = {
        'mean': np.mean(original),
        'median': np.median(original),
        'std': np.std(original),
        'var': np.var(original),
        'skew': stats.skew(original),
        'kurtosis': stats.kurtosis(original)
    }
    
    mod_stats = {
        'mean': np.mean(modified),
        'median': np.median(modified),
        'std': np.std(modified),
        'var': np.var(modified),
        'skew': stats.skew(modified),
        'kurtosis': stats.kurtosis(modified)
    }
    
    diff_metrics = {
        key: abs(mod_stats[key] - orig_stats[key]) / abs(orig_stats[key]) * 100 
        for key in orig_stats.keys()
    }
    
    return orig_stats, mod_stats, diff_metrics

def introduce_random_outliers(df, column, num_outliers=5, std_multiplier=2.0):
    df_modified = df.copy()
    
    data = df[column]
    std = data.std()
    mean = data.mean()
    
    outlier_mask = pd.Series(False, index=df.index)
    
    valid_indices = df.index[10:-10]  
    outlier_indices = np.random.choice(valid_indices, size=num_outliers, replace=False)
    
    outlier_types = ['spike', 'shift', 'trend']
    
    for idx in outlier_indices:
        outlier_type = np.random.choice(outlier_types)
        
        if outlier_type == 'spike':
            multiplier = np.random.uniform(3.0, 5.0) * std_multiplier
            sign = np.random.choice([-1, 1])
            df_modified.loc[idx, column] = mean + (sign * std * multiplier)
            outlier_mask[idx] = True
            
        elif outlier_type == 'shift':
            shift_length = np.random.randint(2, 4)
            shift_indices = range(idx, min(idx + shift_length, len(df)))
            multiplier = np.random.uniform(2.0, 3.0) * std_multiplier
            sign = np.random.choice([-1, 1])
            
            for shift_idx in shift_indices:
                df_modified.loc[shift_idx, column] = mean + (sign * std * multiplier)
                outlier_mask[shift_idx] = True
                
        else:  
            trend_length = np.random.randint(2, 4)
            trend_indices = range(idx, min(idx + trend_length, len(df)))
            base_multiplier = np.random.uniform(2.0, 3.0) * std_multiplier
            sign = np.random.choice([-1, 1])
            
            for i, trend_idx in enumerate(trend_indices):
                progressive_multiplier = base_multiplier * (1 + i * 0.5)
                df_modified.loc[trend_idx, column] = mean + (sign * std * progressive_multiplier)
                outlier_mask[trend_idx] = True
    
    return df_modified, outlier_mask

def test_window_sizes(df, column, window_sizes_list, num_outliers=5):
    results = {}
    
    for window_sizes in window_sizes_list:
        df_modified, true_outliers = introduce_random_outliers(df, column, num_outliers)
        
        detected_df = detect_outliers(df_modified, column, window_sizes=window_sizes)
        detected_outliers = pd.Series(False, index=df_modified.index)
        detected_outliers[detected_df.index] = True
        
        cleaned_df = clean_outliers(df_modified, column, window_sizes=window_sizes)
        
        detection_metrics = calculate_detection_metrics(true_outliers, detected_outliers)
        cleaning_metrics = calculate_cleaning_metrics(df[column], cleaned_df[column])
        
        results[str(window_sizes)] = {**detection_metrics, **cleaning_metrics}
    
    return results

def test_z_thresholds(df, column, z_thresholds, num_outliers=5):
    results = {}
    
    for threshold in z_thresholds:
        df_modified, true_outliers = introduce_random_outliers(df, column, num_outliers)
        
        detected_df = detect_outliers(df_modified, column, z_threshold=threshold)
        detected_outliers = pd.Series(False, index=df_modified.index)
        detected_outliers[detected_df.index] = True
        
        cleaned_df = clean_outliers(df_modified, column, z_threshold=threshold)
        
        detection_metrics = calculate_detection_metrics(true_outliers, detected_outliers)
        cleaning_metrics = calculate_cleaning_metrics(df[column], cleaned_df[column])
        
        results[str(threshold)] = {**detection_metrics, **cleaning_metrics}
    
    return results

def test_robust_stats(df, column, num_outliers=5):
    results = {}
    
    for use_robust in [True, False]:
        df_modified, true_outliers = introduce_random_outliers(df, column, num_outliers)
        
        detected_df = detect_outliers(df_modified, column, use_robust_stats=use_robust)
        detected_outliers = pd.Series(False, index=df_modified.index)
        detected_outliers[detected_df.index] = True
        
        cleaned_df = clean_outliers(df_modified, column, use_robust_stats=use_robust)
        
        detection_metrics = calculate_detection_metrics(true_outliers, detected_outliers)
        cleaning_metrics = calculate_cleaning_metrics(df[column], cleaned_df[column])
        
        results[str(use_robust)] = {**detection_metrics, **cleaning_metrics}
    
    return results

def plot_parameter_sensitivity(results, parameter_name, parameter_values, metric_names):
    plt.figure(figsize=(12, 6))
    
    for metric in metric_names:
        if parameter_name == 'Window Sizes':
            values = [results[str(param)][metric] for param in parameter_values]
            plt.plot(range(len(parameter_values)), values, marker='o', label=metric)
            plt.xticks(range(len(parameter_values)), [str(w) for w in parameter_values], rotation=45)
        else:
            values = [results[str(param)][metric] for param in parameter_values]
            plt.plot(parameter_values, values, marker='o', label=metric)
    
    plt.xlabel(parameter_name)
    plt.ylabel('Metric Value')
    plt.title(f'Parameter Sensitivity Analysis: {parameter_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_statistical_metrics(original, modified, title):
    orig_stats, mod_stats, diff_metrics = calculate_basic_stats(original, modified)
    
    metrics = list(orig_stats.keys())
    orig_values = [orig_stats[m] for m in metrics]
    mod_values = [mod_stats[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, orig_values, width, label='Original')
    rects2 = ax.bar(x + width/2, mod_values, width, label='Modified')
    
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return diff_metrics

def plot_distribution_comparison(original, modified, title):
    plt.figure(figsize=(12, 6))
    
    sns.kdeplot(original, label='Original', color='blue')
    sns.kdeplot(modified, label='Modified', color='red')
    
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_combined_distributions(scenario_results, original_data, folder_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (scenario_name, results) in enumerate(list(scenario_results.items())[:4]):
        ax = axes[i]
        
        sns.kdeplot(original_data, ax=ax, label='Original', color='blue')
        sns.kdeplot(results['cleaned_data'], ax=ax, label='Cleaned', color='green')
        sns.kdeplot(results['modified_data'], ax=ax, label='With Outliers', color='red')
        
        ax.set_title(f"Scenario: {scenario_name}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save in English
    plt.suptitle("Distribution Comparison Across Scenarios", y=1.02)
    plt.savefig(os.path.join(folder_path, "combined_distributions_en.png"), 
                dpi=300, bbox_inches='tight')
    
    # Save in Latvian
    plt.suptitle("Sadalījumu salīdzinājums dažādos scenārijos", y=1.02)
    plt.savefig(os.path.join(folder_path, "combined_distributions_lv.png"), 
                dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_combined_metrics(scenario_results, original_data, folder_path):
    metrics = {
        'precision': {'en': 'Precision', 'lv': 'Precizitāte'},
        'recall': {'en': 'Recall', 'lv': 'Pilnīgums'},
        'f1_score': {'en': 'F1 Score', 'lv': 'F1 vērtējums'},
        'rmse': {'en': 'RMSE', 'lv': 'RMSE'},
        'mae': {'en': 'MAE', 'lv': 'MAE'}
    }
    
    for metric_key, metric_names in metrics.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (param_type, param_values) in enumerate([
            ('Window Sizes', ['[3, 7, 11]', '[5, 11, 21]', '[7, 15, 31]', '[9, 19, 39]']),
            ('Z-Threshold', [1.5, 2.0, 2.5, 3.0]),
            ('Num Outliers', [3, 5, 7, 10]),
            ('Deviation %', [5, 10, 15, 20, 25])
        ]):
            ax = axes[i]
            
            # Collect data for this parameter
            param_data = {}
            
            for scenario_name, results in scenario_results.items():
                if param_type == 'Window Sizes':
                    param_value = str(results['window_sizes'])
                elif param_type == 'Z-Threshold':
                    param_value = str(results['z_threshold'])
                elif param_type == 'Num Outliers':
                    param_value = str(results['num_outliers'])
                else:  # Deviation %
                    param_value = str(results['deviation_pct'])
                
                if param_value not in param_data:
                    param_data[param_value] = []
                
                if metric_key in ['precision', 'recall', 'f1_score']:
                    value = results['detection_metrics'][metric_key]
                else:
                    value = results['cleaning_metrics'][metric_key]
                
                param_data[param_value].append(value)
            
            # Plot boxplots
            box_data = [param_data.get(str(val), []) for val in param_values]
            box_data = [d for d in box_data if d]  # Remove empty lists
            
            if box_data:
                ax.boxplot(box_data, labels=[str(val) for val in param_values if param_data.get(str(val))])
                
                if param_type == 'Window Sizes':
                    ax.set_xticklabels([str(val) for val in param_values if param_data.get(str(val))], rotation=45)
                
                ax.set_title(f"{param_type} vs {metric_names['en']}")
                ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save in English
        plt.suptitle(f"Impact of Parameters on {metric_names['en']}", y=1.02)
        plt.savefig(os.path.join(folder_path, f"combined_metrics_{metric_key}_en.png"), 
                    dpi=300, bbox_inches='tight')
        
        # Save in Latvian
        plt.suptitle(f"Parametru ietekme uz {metric_names['lv']}", y=1.02)
        plt.savefig(os.path.join(folder_path, f"combined_metrics_{metric_key}_lv.png"), 
                    dpi=300, bbox_inches='tight')
        
        plt.close()

def plot_scenario_heatmap(scenario_results, metric, folder_path):
    # Define translations
    metric_translations = {
        'precision': 'Precizitāte',
        'recall': 'Pilnīgums',
        'f1_score': 'F1 vērtējums',
        'rmse': 'RMSE',
        'mae': 'MAE'
    }
    
    # Extract unique parameter values
    window_sizes_list = []
    z_thresholds = []
    num_outliers_list = []
    deviation_percentages = []
    
    for scenario_name, results in scenario_results.items():
        if str(results['window_sizes']) not in window_sizes_list:
            window_sizes_list.append(str(results['window_sizes']))
        
        if results['z_threshold'] not in z_thresholds:
            z_thresholds.append(results['z_threshold'])
        
        if results['num_outliers'] not in num_outliers_list:
            num_outliers_list.append(results['num_outliers'])
        
        if results['deviation_pct'] not in deviation_percentages:
            deviation_percentages.append(results['deviation_pct'])
    
    # Sort parameter values
    window_sizes_list = sorted(window_sizes_list, key=lambda x: eval(x)[0])
    z_thresholds = sorted(z_thresholds)
    num_outliers_list = sorted(num_outliers_list)
    deviation_percentages = sorted(deviation_percentages)
    
    # Create matrices for heatmaps
    window_z_matrix = np.zeros((len(window_sizes_list), len(z_thresholds)))
    outlier_deviation_matrix = np.zeros((len(num_outliers_list), len(deviation_percentages)))
    
    # Fill matrices with metric values
    for scenario_name, results in scenario_results.items():
        w_idx = window_sizes_list.index(str(results['window_sizes']))
        z_idx = z_thresholds.index(results['z_threshold'])
        o_idx = num_outliers_list.index(results['num_outliers'])
        d_idx = deviation_percentages.index(results['deviation_pct'])
        
        if metric in ['precision', 'recall', 'f1_score']:
            value = results['detection_metrics'][metric]
        else:
            value = results['cleaning_metrics'][metric]
        
        # Update matrices
        window_z_matrix[w_idx, z_idx] += value
        outlier_deviation_matrix[o_idx, d_idx] += value
    
    # Calculate averages
    count_matrix = np.zeros((len(window_sizes_list), len(z_thresholds)))
    for scenario_name, results in scenario_results.items():
        w_idx = window_sizes_list.index(str(results['window_sizes']))
        z_idx = z_thresholds.index(results['z_threshold'])
        count_matrix[w_idx, z_idx] += 1
    
    window_z_matrix = np.divide(window_z_matrix, count_matrix, where=count_matrix!=0)
    
    count_matrix = np.zeros((len(num_outliers_list), len(deviation_percentages)))
    for scenario_name, results in scenario_results.items():
        o_idx = num_outliers_list.index(results['num_outliers'])
        d_idx = deviation_percentages.index(results['deviation_pct'])
        count_matrix[o_idx, d_idx] += 1
    
    outlier_deviation_matrix = np.divide(outlier_deviation_matrix, count_matrix, where=count_matrix!=0)
    
    # Calculate median matrices
    median_window_z = np.zeros((len(window_sizes_list), len(z_thresholds)))
    median_outlier_deviation = np.zeros((len(num_outliers_list), len(deviation_percentages)))
    
    for w_idx, window_size in enumerate(window_sizes_list):
        for z_idx, z_threshold in enumerate(z_thresholds):
            values = []
            for scenario_name, results in scenario_results.items():
                if str(results['window_sizes']) == window_size and results['z_threshold'] == z_threshold:
                    if metric in ['precision', 'recall', 'f1_score']:
                        values.append(results['detection_metrics'][metric])
                    else:
                        values.append(results['cleaning_metrics'][metric])
            
            if values:
                median_window_z[w_idx, z_idx] = np.median(values)
    
    for o_idx, num_outliers in enumerate(num_outliers_list):
        for d_idx, deviation in enumerate(deviation_percentages):
            values = []
            for scenario_name, results in scenario_results.items():
                if results['num_outliers'] == num_outliers and results['deviation_pct'] == deviation:
                    if metric in ['precision', 'recall', 'f1_score']:
                        values.append(results['detection_metrics'][metric])
                    else:
                        values.append(results['cleaning_metrics'][metric])
            
            if values:
                median_outlier_deviation[o_idx, d_idx] = np.median(values)
    
    # Function to plot heatmap
    def plot_heatmap(data_matrix, title_en, title_lv, matrix_type):
        if metric in ['rmse', 'mae'] and matrix_type != 'median':
            # For error metrics, lower is better
            vmin, vmax = np.min(data_matrix), np.max(data_matrix)
            cmap = 'RdYlGn_r'  # Reversed colormap
        else:
            # For precision, recall, f1, higher is better
            vmin, vmax = np.min(data_matrix), np.max(data_matrix)
            cmap = 'RdYlGn'
        
        # English version
        plt.figure(figsize=(12, 8))
        
        if matrix_type == 'window_z':
            ax = sns.heatmap(data_matrix, annot=True, fmt=".3f", cmap=cmap,
                            xticklabels=z_thresholds, yticklabels=window_sizes_list,
                            vmin=vmin, vmax=vmax)
            plt.xlabel('Z-Threshold')
            plt.ylabel('Window Sizes')
        else:  # outlier_deviation or median
            if matrix_type == 'outlier_deviation':
                ax = sns.heatmap(data_matrix, annot=True, fmt=".3f", cmap=cmap,
                                xticklabels=deviation_percentages, yticklabels=num_outliers_list,
                                vmin=vmin, vmax=vmax)
                plt.xlabel('Deviation Percentage')
                plt.ylabel('Number of Outliers')
            else:  # median
                ax = sns.heatmap(median_window_z, annot=True, fmt=".3f", cmap=cmap,
                                xticklabels=z_thresholds, yticklabels=window_sizes_list,
                                vmin=vmin, vmax=vmax)
                plt.xlabel('Z-Threshold')
                plt.ylabel('Window Sizes')
        
        plt.title(title_en)
        plt.tight_layout()
        
        if matrix_type == 'window_z':
            plt.savefig(os.path.join(folder_path, f"{metric}_window_z_heatmap_en.png"), 
                        dpi=300, bbox_inches='tight')
        elif matrix_type == 'outlier_deviation':
            plt.savefig(os.path.join(folder_path, f"{metric}_outlier_deviation_heatmap_en.png"), 
                        dpi=300, bbox_inches='tight')
        else:  # median
            plt.savefig(os.path.join(folder_path, f"{metric}_median_heatmap_en.png"), 
                        dpi=300, bbox_inches='tight')
        
        # Latvian version
        plt.title(title_lv)
        
        if matrix_type == 'window_z':
            plt.savefig(os.path.join(folder_path, f"{metric}_window_z_heatmap_lv.png"), 
                        dpi=300, bbox_inches='tight')
        elif matrix_type == 'outlier_deviation':
            plt.savefig(os.path.join(folder_path, f"{metric}_outlier_deviation_heatmap_lv.png"), 
                        dpi=300, bbox_inches='tight')
        else:  # median
            plt.savefig(os.path.join(folder_path, f"{metric}_median_heatmap_lv.png"), 
                        dpi=300, bbox_inches='tight')
        
        plt.close()
    
    # Plot heatmaps
    plot_heatmap(window_z_matrix, 
                 f'{metric.title()} by Window Size and Z-Threshold',
                 f'{metric_translations.get(metric, metric.title())} pēc loga izmēra un Z-sliekšņa',
                 'window_z')
    
    plot_heatmap(outlier_deviation_matrix, 
                 f'{metric.title()} by Number of Outliers and Deviation Percentage',
                 f'{metric_translations.get(metric, metric.title())} pēc noviržu skaita un novirzes procentiem',
                 'outlier_deviation')
    
    plot_heatmap(median_window_z, 
                 f'{metric.title()} - Median values across all parameters',
                 f'{metric_translations.get(metric, metric.title())} - Mediānas vērtības visiem parametriem',
                 'median')

def get_next_test_number():
    test_dir = "../Tests/Outliers"
    existing_folders = glob.glob(os.path.join(test_dir, "Stability_Precision_*"))
    if not existing_folders:
        return "001"
    
    numbers = [int(folder[-3:]) for folder in existing_folders]
    next_num = max(numbers) + 1
    return f"{next_num:03d}"

def create_test_folder():
    base_dir = "../Tests/Outliers"
    test_num = get_next_test_number()
    folder_name = f"Stability_Precision_{test_num}"
    folder_path = os.path.join(base_dir, folder_name)
    
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def save_plot_bilingual(fig, base_name, folder_path, title_en, title_lv):
    plt.suptitle(title_en)
    fig.savefig(os.path.join(folder_path, f"{base_name}_en.png"), 
                bbox_inches='tight', dpi=300)
    
    plt.suptitle(title_lv)
    fig.savefig(os.path.join(folder_path, f"{base_name}_lv.png"), 
                bbox_inches='tight', dpi=300)

def save_results_to_excel(scenario_results, folder_path, test_params):
    rows = []
    for scenario_name, results in scenario_results.items():
        row = {
            'Test_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Scenario': scenario_name,
            'Num_Outliers': results['num_outliers'],
            'Deviation_Pct': results['deviation_pct'],
            'Window_Sizes': results['window_sizes'],
            'Z_Threshold': results['z_threshold'],
            'Precision': results['detection_metrics']['precision'],
            'Recall': results['detection_metrics']['recall'],
            'F1_Score': results['detection_metrics']['f1_score'],
            'MSE': results['cleaning_metrics']['mse'],
            'RMSE': results['cleaning_metrics']['rmse'],
            'MAE': results['cleaning_metrics']['mae']
        }
        
        for metric, value in results['statistical_metrics'].items():
            row[f'Stat_{metric}'] = value
        
        row.update(test_params)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    excel_path = os.path.join(folder_path, "test_results.xlsx")
    df.to_excel(excel_path, index=False)
    
    return excel_path

def test_outlier_scenarios(df, column, num_outliers_list=[3, 5, 7, 10], deviation_percentages=[5, 10, 15, 20, 25], window_sizes_list=[[5, 11, 21], [3, 7, 11], [7, 15, 31], [9, 19, 39]], z_thresholds=[1.5, 2.0, 2.5, 3.0], folder_path=None):
    scenario_results = {}
    
    for num_outliers in num_outliers_list:
        for deviation_pct in deviation_percentages:
            for window_sizes in window_sizes_list:
                for z_threshold in z_thresholds:
                    std_multiplier = deviation_pct / 100
                    
                    test_df = df.copy()
                    
                    modified_df, true_outliers = introduce_random_outliers(
                        test_df, column, num_outliers=num_outliers, std_multiplier=std_multiplier
                    )
                    
                    outliers_v2 = OutliersV2(
                        window_sizes=window_sizes,
                        z_threshold=z_threshold
                    )
                    cleaned_df = modified_df.copy()
                    outliers_v2.detect_and_clean(cleaned_df, column)
                    
                    detected_outliers = outliers_v2.get_outlier_mask()
                    
                    detection_metrics = calculate_detection_metrics(true_outliers, detected_outliers)
                    cleaning_metrics = calculate_cleaning_metrics(df[column], cleaned_df[column])
                    
                    orig_stats, mod_stats, diff_metrics = calculate_basic_stats(df[column], cleaned_df[column])
                    
                    scenario_name = f"n{num_outliers}_d{deviation_pct}_w{'_'.join(map(str,window_sizes))}_z{z_threshold}"
                    scenario_results[scenario_name] = {
                        'num_outliers': num_outliers,
                        'deviation_pct': deviation_pct,
                        'window_sizes': window_sizes,
                        'z_threshold': z_threshold,
                        'detection_metrics': detection_metrics,
                        'cleaning_metrics': cleaning_metrics,
                        'statistical_metrics': diff_metrics,
                        'cleaned_data': cleaned_df[column]  
                    }
    
    if folder_path is not None:
        plot_combined_distributions(scenario_results, df[column], folder_path)
        plot_combined_metrics(scenario_results, df[column], folder_path)
    
    return scenario_results

def execute_parameter_analysis(file_path, target_column):
    try:
        test_folder = create_test_folder()
        
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return None
        
        df = pd.read_excel(file_path)
        
        if target_column not in df.columns:
            print(f"Error: Target column '{target_column}' not found in the data")
            return None
        
        # Check for missing values
        if df[target_column].isna().sum() > 0:
            print(f"Warning: Target column contains {df[target_column].isna().sum()} missing values. These will be handled automatically.")
            df[target_column] = df[target_column].fillna(method='ffill').fillna(method='bfill')
        
        test_params = {
            'Data_File': os.path.basename(file_path),
            'Target_Column': target_column,
        }
        
        scenario_results = test_outlier_scenarios(
            df=df,
            column=target_column,
            num_outliers_list=[3, 5, 7, 10],
            deviation_percentages=[5, 10, 15, 20, 25],
            window_sizes_list=[[5, 11, 21], [3, 7, 11], [7, 15, 31], [9, 19, 39]],
            z_thresholds=[1.5, 2.0, 2.5, 3.0],
            folder_path=test_folder
        )
        
        excel_path = save_results_to_excel(scenario_results, test_folder, test_params)
        print(f"Saved results to: {excel_path}")
        
        plot_combined_distributions(scenario_results, df[target_column], test_folder)
        plot_combined_metrics(scenario_results, df[target_column], test_folder)
        
        for metric in ['precision', 'recall', 'f1_score', 'rmse', 'mae']:
            plot_scenario_heatmap(scenario_results, metric, test_folder)
            plt.close()
        
        print(f"Results saved in: {test_folder}")
        return test_folder
        
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty")
    except pd.errors.ParserError:
        print(f"Error: Unable to parse the file {file_path}")
    except Exception as e:
        print(f"Error during parameter analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return None

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "..", "Data", "Test_dati_sprosti_2cikls.xlsx")
    target_column = "Average Temperature Indoor"
    
    execute_parameter_analysis(file_path, target_column)
