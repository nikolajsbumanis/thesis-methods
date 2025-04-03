#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Outlier Random Testing

This module tests the Outliers_v2 module with random data by introducing artificial
outliers and then detecting and cleaning them. It evaluates the performance of the
outlier detection and cleaning methods under various scenarios.

Main functions:
- introduce_random_outliers: Introduces realistic outlier patterns in the data
- calculate_metrics: Evaluates the performance of outlier cleaning methods
- plot_results: Visualizes original, modified, and cleaned data with outliers

Usage:
Run this script directly to test outlier detection and cleaning methods on
temperature measurements from chicken egg production data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Outliers_v2 import detect_outliers, clean_outliers, calculate_trend
import os
import datetime

def introduce_random_outliers(data, column_name, num_outliers=10):
    modified_data = data.copy()
    
    series = data[column_name]
    std = series.std()
    mean = series.mean()
    
    positions = []
    
    valid_indices = data.index[10:-10]
    min_spacing = 5
    outlier_positions = []
    
    remaining_indices = list(valid_indices)
    for i in range(num_outliers):
        if not remaining_indices:
            break
            
        idx = np.random.choice(remaining_indices)
        outlier_positions.append(idx)
        
        remaining_indices = [j for j in remaining_indices 
                            if abs(j - idx) > min_spacing]
    
    for idx in outlier_positions:
        outlier_type = np.random.choice(['spike', 'level_shift', 'trend_break'])
        
        if outlier_type == 'spike':
            multiplier = np.random.uniform(3.0, 5.0)
            sign = np.random.choice([-1, 1])
            modified_data.loc[idx, column_name] = mean + (sign * std * multiplier)
            positions.append(idx)
            
        elif outlier_type == 'level_shift':
            shift_length = np.random.randint(2, 4)
            shift_indices = range(idx, min(idx + shift_length, len(data)))
            multiplier = np.random.uniform(2.0, 3.0)
            sign = np.random.choice([-1, 1])
            
            for shift_idx in shift_indices:
                if shift_idx in data.index:
                    modified_data.loc[shift_idx, column_name] = mean + (sign * std * multiplier)
                    positions.append(shift_idx)
                
        elif outlier_type == 'trend_break':
            trend_length = np.random.randint(2, 4)
            trend_indices = range(idx, min(idx + trend_length, len(data)))
            base_multiplier = np.random.uniform(2.0, 3.0)
            sign = np.random.choice([-1, 1])
            
            for i, trend_idx in enumerate(trend_indices):
                if trend_idx in data.index:
                    progressive_multiplier = base_multiplier * (1 + i * 0.5)
                    modified_data.loc[trend_idx, column_name] = mean + (sign * std * progressive_multiplier)
                    positions.append(trend_idx)
    
    return modified_data, positions

def load_and_prepare_data(file_path, sheet_name=0):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    
    if 'Temperature Indoor' in data.columns:
        target_column = 'Temperature Indoor'
    elif 'Average Temperature Indoor' in data.columns:
        target_column = 'Average Temperature Indoor'
    else:
        temp_columns = [col for col in data.columns if 'Temperature' in col]
        if temp_columns:
            data['Average Temperature'] = data[temp_columns].mean(axis=1)
            target_column = 'Average Temperature'
        else:
            raise ValueError("No temperature column found in the data")
    
    data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
    
    return data, target_column

def calculate_metrics(original_data, modified_data, cleaned_data, modified_positions):
    true_outliers = pd.Series(False, index=modified_data.index)
    true_outliers.iloc[modified_positions] = True
    
    detected_outliers = (original_data != cleaned_data)
    
    true_positives = sum(true_outliers & detected_outliers)
    false_positives = sum(~true_outliers & detected_outliers)
    false_negatives = sum(true_outliers & ~detected_outliers)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    mse = ((original_data - cleaned_data) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = (original_data - cleaned_data).abs().mean()
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }

def print_statistics(data, cleaned_data):
    print(f"Original data - Mean: {data.mean():.2f}, Std: {data.std():.2f}")
    print(f"Cleaned data - Mean: {cleaned_data.mean():.2f}, Std: {cleaned_data.std():.2f}")
    
    diff_mean = abs(data.mean() - cleaned_data.mean())
    diff_std = abs(data.std() - cleaned_data.std())
    
    print(f"Difference - Mean: {diff_mean:.2f}, Std: {diff_std:.2f}")

def plot_results(original, modified, cleaned, outlier_indices, title):
    plt.figure(figsize=(12, 6))
    
    plt.plot(original.index, original, 'b-', alpha=0.5, linewidth=1, label='Original Data')
    plt.plot(modified.index, modified, 'g-', alpha=0.5, linewidth=1, label='Modified Data')
    plt.plot(cleaned.index, cleaned, 'r-', alpha=0.5, linewidth=1, label='Cleaned Data')
    
    outlier_mask = pd.Series(False, index=original.index)
    outlier_mask.iloc[outlier_indices] = True
    
    plt.scatter(modified.index[outlier_mask], 
                modified[outlier_mask], 
                color='orange', s=50, label='Introduced Outliers')
    
    detected_mask = (original != cleaned)
    plt.scatter(original.index[detected_mask], 
                original[detected_mask], 
                color='red', s=30, marker='x', label='Detected Outliers')
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def main():
    try:
        file_path = os.path.join(os.path.dirname(__file__), "..", "Data", "Test_dati_sprosti_2cikls.xlsx")
        
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
        
        data, column = load_and_prepare_data(file_path)
        
        num_outliers = 20
        
        modified_data, modified_positions = introduce_random_outliers(
            data, column, num_outliers=num_outliers
        )
        
        print(f"Actual outliers introduced: {len(modified_positions)}")
        
        original_series = data[column]
        modified_series = modified_data[column]
        
        window_sizes = [7, 15, 31]
        z_threshold = 2.5
        
        outliers = detect_outliers(
            modified_data, 
            column, 
            window_sizes=window_sizes,
            z_threshold=z_threshold
        )
        
        print(f"Detected {len(outliers)} potential outliers")
        
        cleaned_data = clean_outliers(
            modified_data,
            column,
            window_sizes=window_sizes,
            z_threshold=z_threshold
        )
        
        cleaned_series = cleaned_data[column]
        
        metrics = calculate_metrics(original_series, modified_series, cleaned_series, modified_positions)
        
        print("\nStatistics comparison:")
        print_statistics(original_series, cleaned_series)
        
        fig = plot_results(
            original_series,
            modified_series,
            cleaned_series,
            modified_positions,
            "Outlier Detection Results"
        )
        
        plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
    except pd.errors.EmptyDataError:
        print(f"Error: The data file is empty")
    except pd.errors.ParserError:
        print(f"Error: Unable to parse the data file")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Error during outlier testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
