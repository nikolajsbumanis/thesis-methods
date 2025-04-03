"""
Data Layering Example for Chicken Egg Production Data

This script demonstrates the application of the data layering library, analyzing three parameters:
- Actual laying rate
- Standard laying rate
- Average temperature

The script normalizes the data, combines it using data layering methods, and analyzes
the overlap zones between different data fusion methods. Results are displayed both
graphically and saved to a text file.

Usage:
Run the script, specifying the path to an Excel file with data. The script will automatically
perform the analysis and save the results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataLayering_v1 import data_fusion_main, interpolate_data
from itertools import cycle
from Overlap_v1 import plot_fused_values, analyze_layering_overlap
import logging

logging.basicConfig(filename='overlap_metrics.log', level=logging.INFO, 
                   format='%(message)s', filemode='w')

test_data_filepath = "../Data/Test_dati_sprosti_1cikls.xlsx"
initial_df = pd.read_excel(test_data_filepath, sheet_name="Dati_dienas")

CommonColumn = "Birds age in weeks"
parameters = ["Actual laying rate, %", "Laying rate standard, %", "Avg Temperature Inside"]

def normalize_column(data, range_min=0, range_max=100):
    min_val = data.min()
    max_val = data.max()
    normalized_data = ((data - min_val) / (max_val - min_val)) * (range_max - range_min) + range_min
    return normalized_data

dataframes_dict = {}
for param in parameters:
    initial_df[f"{param}_normalized"] = normalize_column(initial_df[param])
    
    grouped_df = initial_df.groupby(CommonColumn)[f"{param}_normalized"].mean().reset_index()
    grouped_df.rename(columns={f"{param}_normalized": param}, inplace=True)
    dataframes_dict[param] = grouped_df

data_params_with_filename = [f"{param}/{param}" for param in parameters]

weighted_values, pca_values = data_fusion_main(
    dataframes_dict,
    data_params_with_filename,
    CommonColumn,
    [0.85, 0.50, 0.35],
    layering_threshold=40,
    smoothing=1,
    AutoSmooth=1,
    Language='EN'
)

interpolated_index = np.linspace(1, len(weighted_values), len(weighted_values))

metrics = analyze_layering_overlap(weighted_values, pca_values, interpolated_index, CommonColumn, layering_threshold=40)

def format_metrics():
    output = []
    output.append("\n" + "="*50)
    output.append("Overlap Analysis")
    output.append("="*50)
    output.append("\nOverlap Metrics:")
    output.append("-"*30)
    output.append(f"Total overlap area: {metrics['total_overlap_area']:.2f}")
    output.append(f"Proportion of overlap to original: {metrics['proportion_original_overlap']:.2%}")
    output.append(f"Proportion of overlap to PCA: {metrics['proportion_pca_overlap']:.2%}")
    output.append("\nError estimates:")
    output.append(f"  - Original: {metrics['error_original']:.2e}")
    output.append(f"  - PCA: {metrics['error_pca']:.2e}")
    output.append(f"  - Overlap: {metrics['error_overlap']:.2e}")
    output.append("="*50 + "\n")
    return "\n".join(output)

metrics_text = format_metrics()
print(metrics_text)
with open('metrics_output.txt', 'w') as f:
    f.write(metrics_text)

weighted_values, pca_values = data_fusion_main(
    dataframes_dict,
    data_params_with_filename,
    CommonColumn,
    [0.85, 0.50, 0.35],
    layering_threshold=40,
    smoothing=1,
    AutoSmooth=1,
    Language='LV'
)
