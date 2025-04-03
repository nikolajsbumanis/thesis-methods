"""
ARIMA library for missing value imputation in time series

This library provides functions for time series analysis and missing value imputation
using the ARIMA (AutoRegressive Integrated Moving Average) method. ARIMA is a statistical
method that uses time series structure to predict missing values based on previous
observations.

Main functions:
- impute_missing_values_with_arima: Imputes missing values using the ARIMA model
- plot_imputed_values_with_arima: Visualizes original and imputed data
- analyze_and_impute_with_arima: Analyzes time series and performs imputation

Usage:
Import this library and use the plot_imputed_values_with_arima function, specifying
the data frame, value column, and date column.
"""

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from itertools import product

def check_missing_values(ts):
    return ts.isnull().sum() > 0

def check_stationarity(ts):
    result = adfuller(ts.dropna())
    return result[1] <= 0.05

def determine_arima_parameters(ts):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    plot_acf(ts.dropna(), ax=ax[0])
    plot_pacf(ts.dropna(), ax=ax[1])
    plt.show()

    best_aic = float('inf')
    best_order = None
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                except:
                    continue

    return best_order

def fit_arima(ts, order):
    model = ARIMA(ts, order=order)
    results = model.fit()
    return results.fittedvalues

def plot_arima_results(ts, fitted_values):
    plt.figure(figsize=(15, 6))
    ts.plot(label='Observed', color='blue')
    fitted_values.plot(label='ARIMA Model', color='orange')
    plt.title('Time Series with ARIMA Imputation', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def analyze_and_impute_with_arima(df, y_col, x_col):
    ts = df.set_index(x_col)[y_col]

    if not check_missing_values(ts):
        print("No missing values detected.")
        return

    if check_stationarity(ts):
        print("Series is stationary.")
    else:
        print("Series is not stationary. Differencing might be needed for ARIMA.")

    order = determine_arima_parameters(ts)
    print(f"Best ARIMA parameters (p, d, q): {order}")

    fitted_values = fit_arima(ts, order)

    plot_arima_results(ts, fitted_values)

def impute_missing_values_with_arima(df, value_column, date_column):
    if df[value_column].isnull().sum() == 0:
        print("No missing values found.")
        return df

    df = df.set_index(date_column)
    df.index = pd.to_datetime(df.index)
    df.index.freq = pd.infer_freq(df.index)

    p = d = q = range(0, 3)
    pdq = list(product(p, d, q))
    best_aic = float('inf')
    best_order = None

    interpolated_series = df[value_column].interpolate()

    for order in pdq:
        try:
            model = ARIMA(interpolated_series, order=order)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
        except:
            continue

    print(best_order)

    model = ARIMA(df[value_column], order=best_order)
    results = model.fit()
    forecast = results.predict(start=0, end=len(df) - 1)

    df[value_column] = df[value_column].combine_first(forecast)

    df.reset_index(inplace=True)
    return df

def plot_imputed_values_with_arima(data, value_col, date_col):
    """
    Impute missing values using ARIMA and plot the original vs imputed series.

    Args:
    - data (pd.DataFrame): The data containing the time series.
    - value_col (str): The column name with the values to be imputed.
    - date_col (str): The column name with the date information.

    Returns:
    - None. (But it plots the time series.)
    """
    # Use the impute function on the dataset
    imputed_data = impute_missing_values_with_arima(data.copy(), value_col, date_col)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Plot original data with gaps
    ax.plot(data[date_col], data[value_col], color='blue', label='Observed data', lw=1)
    
    # Plot imputed data
    ax.plot(imputed_data[date_col], imputed_data[value_col], color='red', label='ARIMA model', alpha=0.25, lw=5)
    
    # Formatting the plot
    ax.set_title(f'Missing Value Imputation using ARIMA model', fontsize=18)
    ax.set_xlabel("Date", fontsize=18)
    ax.set_ylabel("Average Temperature Indoors, °C", fontsize=18)
    ax.legend(fontsize=18)
    ax.grid(True)
    
    plt.show()
    
    return imputed_data
