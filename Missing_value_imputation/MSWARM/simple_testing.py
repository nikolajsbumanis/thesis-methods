"""
MSWARM (Modified Standard Weighted Average Robust Method) imputation example for temperature data

This script demonstrates the application of the MSWARM (Modified Standard Weighted Average 
Robust Method) for missing value imputation, using temperature measurements from chicken egg 
production data. The script calculates the average temperature from multiple sensors and uses 
the MSWARM algorithm to fill in missing values in the time series.

Usage:
Run the script, specifying the path to an Excel file with data. The script will automatically
perform the imputation and display the results graphically.
"""

import pandas as pd
import numpy as np
from MSWARM_v1 import MissingValuesWithFlag

def main():
    file_path = '../../Data/Test_dati_sprosti_1cikls.xlsx'
    df_full = pd.read_excel(file_path)
    
    df_full = df_full[df_full['Date'] <= '2021-10-09']
    df_full['Average_Temperature_1st'] = df_full[['Temperature 1 floor', 'Temperature 8 floor', 'Temperature computer']].mean(axis=1)
    
    mvi = MissingValuesWithFlag()
    
    df_result, imputed_data = mvi.CalcAndPlot(
        df_full, 
        'Average_Temperature_1st',
        'Date',
        seasonal_period=30,
        deviation_threshold=1000,
        DoPlot=1
    )
    
    print("Imputation complete!")

if __name__ == "__main__":
    main()
