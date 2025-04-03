# Time Series Processing Methods

Methods for handling time series data with missing values and outliers, focused on environmental measurements in agricultural settings.

## Project Overview

This repository contains implementations of advanced algorithms for processing time series data, particularly temperature measurements from agricultural environments. The methods focus on two key challenges:

1. **Missing Value Imputation**: Filling gaps in time series data using context-aware methods
2. **Outlier Detection and Removal**: Identifying and handling anomalous values while preserving data integrity

The implementations are optimized for environmental monitoring data but can be applied to various time series applications.

## Repository Structure

```
├── Data/                                # Test datasets
│   ├── Test_dati_sprosti_1cikls.xlsx    # First cycle of egg production data
│   └── Test_dati_sprosti_2cikls.xlsx    # Second cycle of egg production data
│
├── Missing_value_imputation/            # Missing value handling algorithms
│   ├── MSWARM/                          # MSWARM implementation
│   └── ARIMA/                           # ARIMA implementation
│
├── Outlier_removal/                     # Outlier detection algorithms
│   ├── Outliers_v2.py                   # Main implementation
│   ├── Outliers_v2_Stability_Precision.py  # Parameter optimization
│   ├── Rolling_window_v1.py             # Rolling window approach
│   └── Winsorization_v1.py              # Winsorization technique
│
└── DataLayering/                        # Data fusion methods
```

## Methods

### Missing Value Imputation

#### MSWARM: Modified Standard Weighted Average Robust Method

MSWARM combines local information with global trends to impute missing values in time series data:

- **Adaptive to Local Patterns**: Adjusts to local data characteristics
- **Robust to Outliers**: Minimizes the impact of outliers on imputation
- **Trend-Aware**: Accounts for underlying trends in the data
- **Optimized for Temperature Data**: Particularly effective for environmental measurements

Key files:
- `MSWARM_v1.py`: Core implementation
- `Evaluation.py`: Performance evaluation
- `Stability_Precision.py`: Analysis of stability and precision

#### ARIMA: AutoRegressive Integrated Moving Average

Statistical method for time series forecasting and imputation:

- **Seasonal Handling**: Accounts for seasonal patterns
- **Statistical Foundation**: Based on well-established time series principles
- **Effective for Regular Patterns**: Works well with data showing clear temporal dependencies

### Outlier Detection

The outlier detection framework uses a multi-window approach with adaptive parameters:

- **Multi-Window Detection**: Uses windows of different sizes to detect outliers at various scales
- **Optimal Parameters**: Window sizes [7, 15, 31], Z-threshold 3.0
- **Trend-Based Replacement**: Replaces outliers with values derived from underlying trends
- **Comprehensive Evaluation**: Includes tools for testing detection accuracy

## Usage Examples

### Missing Value Imputation with MSWARM

```python
import pandas as pd
from Missing_value_imputation.MSWARM.MSWARM_v1 import impute_missing_values

# Load data
data = pd.read_excel('Data/Test_dati_sprosti_1cikls.xlsx')

# Apply MSWARM imputation
imputed_data = impute_missing_values(
    data, 
    'Average Temperature Indoor',
    date_column='Date'
)

# Check results
print(f"Missing values before: {data['Average Temperature Indoor'].isna().sum()}")
print(f"Missing values after: {imputed_data['Average Temperature Indoor'].isna().sum()}")
```

### Outlier Detection and Removal

```python
import pandas as pd
from Outlier_removal.Outliers_v2 import OutliersV2

# Load data
data = pd.read_excel('Data/Test_dati_sprosti_2cikls.xlsx')

# Create outlier detector with optimal parameters
outlier_detector = OutliersV2(
    window_sizes=[7, 15, 31],  # Optimal window sizes
    z_threshold=3.0            # Optimal threshold
)

# Detect and clean outliers
outlier_detector.detect_and_clean(data, 'Average Temperature Indoor')

# Get results
outliers = outlier_detector.get_outlier_mask()
cleaned_data = outlier_detector.get_cleaned_data()

print(f"Number of outliers detected: {outliers.sum()}")
```

## Parameter Optimization

The repository includes tools for optimizing algorithm parameters:

- **Outlier Detection**: `Outliers_v2_Stability_Precision.py` evaluates different window sizes and thresholds
- **MSWARM**: `Stability_Precision.py` analyzes imputation stability with different parameters

Current optimal parameters for outlier detection:
- Window sizes: [7, 15, 31]
- Z-threshold: 3.0

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn
- statsmodels

## Installation

```bash
# Clone the repository
git clone https://github.com/nikolajsbumanis/thesis-methods.git

# Install dependencies
pip install pandas numpy matplotlib scipy scikit-learn statsmodels
