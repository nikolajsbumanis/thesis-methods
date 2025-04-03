"""
MSWARM (Modified Standard Weighted Average Robust Method) imputation testing with randomly selected data gaps

This script tests the accuracy of the MSWARM (Modified Standard Weighted Average Robust Method) 
by introducing randomly selected data gaps in a time series. The script allows comparing 
different imputation methods by calculating the mean absolute error (MAE) and mean squared 
error (MSE) between the original and imputed values.

Usage:
Run the script, specifying the path to an Excel file with data. The script will automatically
perform the imputation and display the results graphically.
"""

import pandas as pd
import numpy as np
import random
from MSWARM_v1 import MissingValuesWithFlag
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

file_path = '../../Data/Test_dati_sprosti_2cikls.xlsx'
df_full = pd.read_excel(file_path)
df_full = df_full[(df_full['Date'] >= '2021-03-23') & (df_full['Date'] <= '2021-10-09')]

def introduce_missing_values_randomly(df, column, proportion):
    total_count = len(df)
    missing_count = int(total_count * proportion)
    missing_indices = random.sample(range(total_count), missing_count)
    df.loc[missing_indices, column] = np.nan
    return df

def introduce_missing_values_in_chunks(df, column, chunk_size, num_chunks):
    total_count = len(df)
    for _ in range(num_chunks):
        start_index = random.randint(0, total_count - chunk_size)
        end_index = start_index + chunk_size
        df.iloc[start_index:end_index, df.columns.get_loc(column)] = np.nan
    return df

target_column = 'Average Temperature Indoor'

df_full['Average Temperature Indoor'] = df_full[['Temperature 1 floor', 'Temperature 8 floor', 'Temperature computer']].mean(axis=1)

original_data = df_full[target_column].copy()

random.seed(42)
df_with_gaps = df_full.copy()
df_with_gaps = introduce_missing_values_in_chunks(df_with_gaps, target_column, chunk_size=5, num_chunks=10)

missing_count = df_with_gaps[target_column].isna().sum()
total_count = len(df_with_gaps)
missing_proportion = (missing_count / total_count) * 100

plt.figure(figsize=(15, 6))
plt.plot(df_full['Date'], df_full[target_column], 'b-', label='Original data')
plt.plot(df_with_gaps['Date'], df_with_gaps[target_column], 'r-', label='Data with gaps')
plt.title(f'Original vs Data with Gaps ({missing_proportion:.2f}% missing)')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def calculate_metrics(original, imputed):
    missing_mask = df_with_gaps[target_column].isna()
    mae = mean_absolute_error(original[missing_mask], imputed[missing_mask])
    mse = mean_squared_error(original[missing_mask], imputed[missing_mask])
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")

def main():
    global df_full, df_with_gaps
    
    mvi = MissingValuesWithFlag()
    df_result, imputed_data = mvi.CalcAndPlot(
        df_with_gaps, 
        target_column,
        'Date',
        seasonal_period=30,
        deviation_threshold=None,
        DoPlot=1
    )
    
    calculate_metrics(original_data, imputed_data)
    
    plt.figure(figsize=(15, 6))
    plt.plot(df_full['Date'], original_data, 'b-', label='Original data')
    plt.plot(df_full['Date'], imputed_data, 'r-', label='MSWARM (Modified Standard Weighted Average Robust Method) imputed data')
    
    missing_mask = df_with_gaps[target_column].isna()
    plt.scatter(df_full['Date'][missing_mask], imputed_data[missing_mask], color='red', s=50, label='Imputed points')
    
    plt.title(f'Original vs MSWARM (Modified Standard Weighted Average Robust Method) Imputed Data')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
