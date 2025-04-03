"""
Overlap Analysis Library for Data Layering

This script provides functions for analyzing and visualizing overlap zones between data layering methods.
It allows comparing different data fusion methods and identifying common zones where both methods
show significant results.

Main functions:
- plot_fused_values: Visualizes two data fusion methods and their overlap zones
- quantify_overlap: Calculates and visualizes the overlap metric between methods
- analyze_layering_overlap: Main function that combines visualization and analysis

Usage:
Import this library and use the analyze_layering_overlap function, specifying the
weighted values, PCA values, interpolated index, and common column.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from DataLayering_v1 import data_fusion_main, interpolate_data

def plot_fused_values(interpolated_index, original_values, pca_values, common_column, 
                         layering_threshold=30, title="Overlap between Original weighted fused values and PCA-based fused values"):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.axhline(y=layering_threshold, color='red', lw=0.5, label=f'Threshold: {layering_threshold}')

    ax1.plot(interpolated_index, original_values, 'b-', 
             label='Original weighted fused values', linewidth=2)
    ax1.fill_between(interpolated_index, original_values,
                     where=original_values >= layering_threshold, 
                     color='blue', alpha=0.3)

    ax1.plot(interpolated_index, pca_values, 'r-', 
             label='PCA-based fused values', linewidth=2)
    ax1.fill_between(interpolated_index, pca_values,
                     where=pca_values >= layering_threshold, 
                     color='red', alpha=0.3)

    original_values_high_res = interpolate_data(original_values, 40)
    pca_values_high_res = interpolate_data(pca_values, 40)
    high_res_index = np.linspace(min(interpolated_index), max(interpolated_index), len(original_values_high_res))

    for values, color in [(original_values_high_res, 'blue'), (pca_values_high_res, 'red')]:
        crossing_upwards = np.where((values[:-1] < layering_threshold) & 
                                  (values[1:] >= layering_threshold))[0]
        crossing_downwards = np.where((values[:-1] >= layering_threshold) & 
                                    (values[1:] < layering_threshold))[0]
        cross_indices = np.sort(np.concatenate((crossing_upwards, crossing_downwards)))
        
        for idx in cross_indices:
            ax1.axvline(x=high_res_index[idx], color=color, lw=1, linestyle='--')

    ax1.set_xlabel(common_column, fontsize=18)
    ax1.set_ylabel('Combined Value', fontsize=18)
    ax1.legend(loc='upper left', fontsize=18)
    ax1.grid(True)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def quantify_overlap(interpolated_index, original_values, pca_values, common_column, 
                    layering_threshold=30, title="Overlap of Significant Regions"):
    plt.figure(figsize=(12, 8))

    plt.axhline(y=layering_threshold, color='red', lw=0.5, 
                label=f'Threshold: {layering_threshold}')

    plt.plot(interpolated_index, original_values, 'b-', 
             label='Original Weighted Combined Values', linewidth=2)
    plt.plot(interpolated_index, pca_values, 'r-', 
             label='PCA-based Combined Values', linewidth=2)

    overlap_values = np.minimum(original_values, pca_values)
    plt.fill_between(interpolated_index, overlap_values, 
                     where=overlap_values >= layering_threshold,
                     color='gray', alpha=0.5, label='Overlap Region')

    original_values_high_res = interpolate_data(original_values, 40)
    pca_values_high_res = interpolate_data(pca_values, 40)
    high_res_index = np.linspace(min(interpolated_index), max(interpolated_index), len(original_values_high_res))

    for values, color in [(original_values_high_res, 'blue'), (pca_values_high_res, 'red')]:
        crossing_upwards = np.where((values[:-1] < layering_threshold) & 
                                  (values[1:] >= layering_threshold))[0]
        crossing_downwards = np.where((values[:-1] >= layering_threshold) & 
                                    (values[1:] < layering_threshold))[0]
        cross_indices = np.sort(np.concatenate((crossing_upwards, crossing_downwards)))
        
        for idx in cross_indices:
            plt.axvline(x=high_res_index[idx], color=color, lw=1, linestyle='--')

    total_overlap_area = np.trapz(overlap_values, dx=1)
    total_original_area = np.trapz(original_values, dx=1)
    total_pca_area = np.trapz(pca_values, dx=1)

    proportion_original_overlap = total_overlap_area / total_original_area
    proportion_pca_overlap = total_overlap_area / total_pca_area

    def trapezoidal_error(y_values, x_values=None, degree=3):
        if x_values is None:
            x_values = np.arange(len(y_values))
        coeffs = np.polyfit(x_values, y_values, degree)
        poly_2d = np.polyder(np.poly1d(coeffs), 2)
        x_range = np.linspace(min(x_values), max(x_values), 1000)
        max_2d = max(abs(poly_2d(x_range)))
        error = (max(x_values) - min(x_values))**3 / (12 * len(x_values)**2) * max_2d
        return error

    error_original = trapezoidal_error(original_values, interpolated_index)
    error_pca = trapezoidal_error(pca_values, interpolated_index)
    error_overlap = trapezoidal_error(overlap_values, interpolated_index)

    plt.title(title, fontsize=16)
    plt.xlabel(common_column, fontsize=14)
    plt.ylabel('Combined Value', fontsize=16)
    plt.legend(loc='upper left', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        'total_overlap_area': total_overlap_area,
        'proportion_original_overlap': proportion_original_overlap,
        'proportion_pca_overlap': proportion_pca_overlap,
        'error_original': error_original,
        'error_pca': error_pca,
        'error_overlap': error_overlap
    }

def analyze_layering_overlap(weighted_values, pca_values, interpolated_index, common_column, 
                           layering_threshold=30):
    plot_fused_values(interpolated_index, weighted_values, pca_values, 
                           common_column, layering_threshold)
    metrics = quantify_overlap(interpolated_index, weighted_values, pca_values, 
                        common_column, layering_threshold)
    
    return metrics
