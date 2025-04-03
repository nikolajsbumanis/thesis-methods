"""
ARIMA imputation example for temperature data

This script demonstrates the application of the ARIMA method for missing value imputation,
using temperature measurements from chicken egg production data. The script calculates the average
temperature from multiple sensors and uses the ARIMA model to fill in missing values
in the time series.

Usage:
Run the script, specifying the path to an Excel file with data. The script will automatically
perform the imputation and display the results graphically.
"""

import pandas as pd
from ARIMA_v1 import plot_imputed_values_with_arima

def main():
    egg_data_c_1 = pd.read_excel('../../Data/Test_dati_sprosti_1cikls.xlsx')
    egg_data_c_1['Average_Temperature_1st'] = egg_data_c_1[['Temperature 1 floor', 'Temperature 8 floor', 'Temperature computer']].mean(axis=1)
    
    plot_imputed_values_with_arima(egg_data_c_1, "Average_Temperature_1st", "Date")

if __name__ == "__main__":
    main()
