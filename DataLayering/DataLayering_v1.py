"""
Data Layering Main Library

This script provides functions for combining different data parameters using weights and PCA method.
Main functions:
- interpolate_data: Data interpolation to improve visualization
- data_fusion_weighted: Data fusion using weights
- data_fusion_pca: Data fusion using PCA
- data_fusion_main: Main function that combines both methods and visualizes results

Usage:
Import this library and use the data_fusion_main function, specifying the data dictionary,
parameters, common column, and weights.
"""

from scipy.interpolate import interp1d
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import cycle

def interpolate_data(data, smoothing):
    original_index = np.arange(len(data))
    interpolated_index = np.linspace(0, len(data) - 1, len(data) * smoothing)
    interpolated_data = interp1d(original_index, data, kind='linear')(interpolated_index)
    return interpolated_data

def interpolate_data_auto_smoothing(data, threshold):
    smoothing = 1
    max_smoothing = 50
    tolerance = 0.2

    while smoothing <= max_smoothing:
        interpolated_data = interpolate_data(data, smoothing)

        close_to_threshold_indices = np.where((interpolated_data >= threshold - tolerance) & (interpolated_data <= threshold + tolerance))[0]
        if close_to_threshold_indices.size > 0:
            break

        smoothing += 1

    return interpolated_data, smoothing

def data_fusion_weighted(dataframes_dict, data_params, CommonColumn, weights, layering_threshold=30, smoothing=1, FilterBy=None, FilterValue=None, AutoSmooth=0):
    filenames = [param.split('/')[0] for param in data_params]
    parameters = [param.split('/')[1] for param in data_params]

    dataframes = [dataframes_dict[filename] for filename in filenames]
    merged_data = dataframes[0]
    for dataframe in dataframes[1:]:
        merged_data = merged_data.merge(dataframe, on=CommonColumn, how='left')

    if FilterBy and FilterValue:
        merged_data = merged_data[merged_data[FilterBy] == FilterValue]

    interpolated_data = {}

    if AutoSmooth:
        _, smoothing = interpolate_data_auto_smoothing(merged_data[parameters[0]].values, layering_threshold)
        for parameter in parameters:
            interpolated_data[parameter] = interpolate_data(merged_data[parameter].values, smoothing)
    else:
        for parameter in parameters:
            interpolated_data[parameter] = interpolate_data(merged_data[parameter].values, smoothing)

    weighted_values = np.zeros(len(interpolated_data[next(iter(parameters))]))

    for weight, parameter in zip(weights, parameters):
        weighted_values += weight * (interpolated_data[parameter] / 100)

    X_min = weighted_values.min()
    X_max = weighted_values.max()
    range_min = 0
    range_max = 100
    weighted_values = ((weighted_values - X_min) / (X_max - X_min)) * (range_max - range_min) + range_min

    return interpolated_data, weighted_values, smoothing, merged_data

def data_fusion_pca(interpolated_data):
    pca_data = np.array(list(interpolated_data.values())).T
    
    pca = PCA(n_components=1)
    pca_values = pca.fit_transform(pca_data)
    
    first_param_values = pca_data[:, 0]
    correlation = np.corrcoef(first_param_values, pca_values.flatten())[0, 1]
    
    if correlation < 0:
        pca_values = -pca_values
        
    scaler = MinMaxScaler(feature_range=(0, 100))
    pca_values = scaler.fit_transform(pca_values).flatten()
    
    return pca_values

def data_fusion_main(dataframes_dict, data_params, CommonColumn, weights, layering_threshold=30, smoothing=1, FilterBy=None, FilterValue=None, AutoSmooth=0, Language='EN'):
    interpolated_data, weighted_values, smoothing, merged_data = data_fusion_weighted(
        dataframes_dict, data_params, CommonColumn, weights, layering_threshold, smoothing, FilterBy, FilterValue, AutoSmooth)

    pca_values = data_fusion_pca(interpolated_data)

    interpolated_index = np.linspace(1, len(interpolated_data[data_params[0].split('/')[1]]), 
                                   len(interpolated_data[data_params[0].split('/')[1]]))
    
    titles = {
        'EN': {
            'weighted': 'Original Weighted Fused Values',
            'pca': 'PCA-based Fused Values'
        },
        'LV': {
            'weighted': 'Sākotnējās svērtās apvienotās vērtības',
            'pca': 'Uz PCA balstītās apvienotās vērtības'
        }
    }
    
    plot_data_fusion_results(interpolated_index, interpolated_data, weighted_values, 
                         [param.split('/')[1] for param in data_params], 
                         titles[Language]['weighted'], CommonColumn, dataframes_dict, 
                         layering_threshold, smoothing, AutoSmooth, Language)
    
    plot_data_fusion_results(interpolated_index, interpolated_data, pca_values, 
                         [param.split('/')[1] for param in data_params], 
                         titles[Language]['pca'], CommonColumn, dataframes_dict, 
                         layering_threshold, smoothing, AutoSmooth, Language)

    return weighted_values, pca_values

def plot_data_fusion_results(interpolated_index, interpolated_data, fused_values, parameters, title, CommonColumn, dataframes_dict, layering_threshold=30, smoothing=1, AutoSmooth=0, Language='EN'):
    labels = {
        'EN': {
            'threshold': 'Threshold',
            'layering_value': 'Layering Value',
            'interpolated_values': 'Interpolated values',
            'title_format': '{} for common column {}'
        },
        'LV': {
            'threshold': 'Slieksnis',
            'layering_value': 'Apvienotā vērtība',
            'interpolated_values': 'Interpolētās vērtības',
            'title_format': '{} kopīgai kolonnai {}'
        }
    }

    fig, ax1 = plt.subplots(figsize=(12, 8))

    plt.axhline(y=layering_threshold, color='red', lw=0.5, label=f'{labels[Language]["threshold"]}: {layering_threshold}')

    color_cycle = cycle(['blue', 'orange', 'green', 'yellow', 'red', 'purple', 'brown'])
    for parameter in parameters:
        ax1.plot(interpolated_index, interpolated_data[parameter],
                 color=next(color_cycle), marker='', linestyle='-', linewidth=0.25, label=parameter)
        ax1.fill_between(interpolated_index, interpolated_data[parameter], alpha=0.2)

    ax1.plot(interpolated_index, fused_values, 'k-', label=title, linewidth=0.5)
    ax1.fill_between(interpolated_index, fused_values, where=fused_values >= layering_threshold, color='gray', hatch='...', alpha=0.35)

    first_dataframe_key = next(iter(dataframes_dict))
    unique_values = dataframes_dict[first_dataframe_key][CommonColumn].unique()

    ax1.set_xticks(np.linspace(min(interpolated_index), max(interpolated_index), len(unique_values)))

    ticks = ax1.get_xticks()

    selected_ticks = ticks[::5]
    selected_labels = unique_values[::5]  

    ax1.set_xticks(selected_ticks)
    ax1.set_xticklabels(selected_labels)

    ax1.set_xlabel(CommonColumn, fontsize=20)
    ax1.set_ylabel(labels[Language]['layering_value'], fontsize=20)
    ax1.grid(True)

    fused_values_high_res = interpolate_data(fused_values, 20)
    interpolated_index_high_res = np.linspace(interpolated_index[0], interpolated_index[-1], len(fused_values_high_res))

    crossing_upwards = np.where((fused_values_high_res[:-1] < layering_threshold) & (fused_values_high_res[1:] >= layering_threshold))[0]
    crossing_downwards = np.where((fused_values_high_res[:-1] >= layering_threshold) & (fused_values_high_res[1:] < layering_threshold))[0]
    cross_threshold_indices = np.concatenate((crossing_upwards, crossing_downwards))
    cross_threshold_indices.sort()  

    for idx in cross_threshold_indices:
        ax1.axvline(x=interpolated_index_high_res[idx], color='black', lw=1, linestyle='--')

    ax1.legend(loc='upper left', fontsize=14)
    plt.title(labels[Language]['title_format'].format(title, CommonColumn), fontsize=16)
    plt.tight_layout()
    plt.show()
