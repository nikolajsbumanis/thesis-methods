#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Outlier Random Testing - Multiple Iterations

This module implements a comprehensive testing framework for evaluating outlier detection
and cleaning methods across multiple iterations with different outlier patterns. It provides
statistical analysis of the method's performance under various scenarios.

Key features:
- Introduces random outliers with configurable patterns (spikes, level shifts, trends)
- Runs multiple test iterations to ensure statistical significance
- Calculates performance metrics (RMSE, MAE, detection accuracy)
- Generates visualizations of results for easy interpretation

Main components:
- OutlierMultiTester: Class that handles the testing process and results analysis
- Various test scenarios with different outlier patterns and configurations

Usage:
Run this script directly to perform multiple iterations of outlier detection testing
on temperature measurements from chicken egg production data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from Outliers_v2 import OutliersV2
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

class OutlierMultiTester:
    def __init__(self, data_path, target_column, num_tests=100):
        try:
            self.data_path = data_path
            self.target_column = target_column
            self.num_tests = num_tests
            self.results = []
            self.window_sizes = [7, 15, 31]
            self.z_threshold = 3.0
            
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            self.df = pd.read_excel(self.data_path)
            
            if self.target_column not in self.df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in the data")
            
            # Handle missing values
            missing_count = self.df[self.target_column].isna().sum()
            if missing_count > 0:
                print(f"Warning: Target column contains {missing_count} missing values. These will be handled automatically.")
            
            self.original_data = self.df[self.target_column].fillna(method='ffill').fillna(method='bfill').values
            self.detector = OutliersV2(window_sizes=self.window_sizes, z_threshold=self.z_threshold)
            
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
            raise
        except pd.errors.EmptyDataError:
            print(f"Error: The file {self.data_path} is empty")
            raise
        except pd.errors.ParserError:
            print(f"Error: Unable to parse the file {self.data_path}")
            raise
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def generate_random_outliers(self, data):
        data_with_outliers = data.copy()
        n = len(data)
        
        num_outliers = np.random.randint(int(0.02 * n), int(0.05 * n))
        
        outlier_positions = np.random.choice(n, size=num_outliers, replace=False)
        
        for pos in outlier_positions:
            outlier_type = np.random.choice(['spike', 'shift', 'trend'])
            
            if outlier_type == 'spike':
                magnitude = np.random.uniform(2, 4)
                direction = np.random.choice([-1, 1])
                data_with_outliers[pos] += direction * magnitude * np.std(data)
                
            elif outlier_type == 'shift':
                shift_length = np.random.randint(3, 7)
                end_pos = min(pos + shift_length, n)
                magnitude = np.random.uniform(1.5, 3)
                direction = np.random.choice([-1, 1])
                data_with_outliers[pos:end_pos] += direction * magnitude * np.std(data)
                
            else:  
                trend_length = np.random.randint(4, 8)
                end_pos = min(pos + trend_length, n)
                magnitude = np.random.uniform(0.5, 1.5)
                direction = np.random.choice([-1, 1])
                trend = np.linspace(0, magnitude * np.std(data), end_pos - pos)
                data_with_outliers[pos:end_pos] += direction * trend
                
        return data_with_outliers, outlier_positions

    def run_single_test(self):
        try:
            data_with_outliers, outlier_positions = self.generate_random_outliers(self.original_data)
            
            temp_df = pd.DataFrame({
                self.target_column: data_with_outliers
            })
            
            cleaned_data = self.detector.clean(temp_df[self.target_column])
            
            cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill').values
            
            min_len = min(len(self.original_data), len(cleaned_data))
            original_data = self.original_data[:min_len]
            cleaned_data = cleaned_data[:min_len]
            
            rmse = np.sqrt(mean_squared_error(original_data, cleaned_data))
            mae = mean_absolute_error(original_data, cleaned_data)
            mape = np.mean(np.abs((original_data - cleaned_data) / original_data)) * 100
            rmspe = np.sqrt(np.mean(((original_data - cleaned_data) / original_data) ** 2)) * 100
            
            return {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'rmspe': rmspe,
                'num_outliers': len(outlier_positions)
            }
        except Exception as e:
            print(f"Error during single test: {str(e)}")
            raise

    def run_multi_test(self):
        try:
            for i in range(self.num_tests):
                result = self.run_single_test()
                self.results.append(result)
        except Exception as e:
            print(f"Error during multi-test: {str(e)}")
            raise

    def analyze_results(self):
        try:
            results_df = pd.DataFrame(self.results)
            
            os.makedirs('Tests', exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_path = f'Tests/outlier_results_{timestamp}.xlsx'
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
                
                summary_stats = pd.DataFrame({
                    metric: {
                        'Mean': results_df[metric].mean(),
                        'Std': results_df[metric].std(),
                        'Min': results_df[metric].min(),
                        'Max': results_df[metric].max(),
                        'Median': results_df[metric].median()
                    }
                    for metric in ['rmse', 'mae', 'mape', 'rmspe']
                })
                summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
            
            print(f"\nResults saved to: {excel_path}")
            
            self._plot_error_metrics_over_time(results_df, lang='en')
            self._plot_error_metrics_over_time(results_df, lang='lv')
            self._plot_rmse_boxplot(results_df, lang='en')
            self._plot_rmse_boxplot(results_df, lang='lv')
            self._plot_error_distributions(results_df, lang='en')
            self._plot_error_distributions(results_df, lang='lv')
            self._plot_metric_correlations(results_df, lang='en')
            self._plot_metric_correlations(results_df, lang='lv')
            
            self._print_summary_statistics(results_df)
        except Exception as e:
            print(f"Error during results analysis: {str(e)}")
            raise

    def _plot_error_metrics_over_time(self, df, lang='en'):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['rmse'], label='RMSE', marker='o', markersize=2)
        plt.plot(df.index, df['mae'], label='MAE', marker='o', markersize=2)
        
        titles = {
            'en': {
                'title': 'Error Metrics Over Time',
                'xlabel': 'Test Index',
                'ylabel': 'Error Value'
            },
            'lv': {
                'title': 'Kļūdu metriku laika grafiks',
                'xlabel': 'Testa indekss',
                'ylabel': 'Kļūdas vērtība'
            }
        }
        
        plt.title(titles[lang]['title'])
        plt.xlabel(titles[lang]['xlabel'])
        plt.ylabel(titles[lang]['ylabel'])
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['mape'], label='MAPE', marker='o', markersize=2)
        plt.plot(df.index, df['rmspe'], label='RMSPE', marker='o', markersize=2)
        plt.xlabel(titles[lang]['xlabel'])
        plt.ylabel('Percentage Error (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'Tests/error_metrics_time_{lang}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_rmse_boxplot(self, df, lang='en'):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[['rmse', 'mae', 'mape', 'rmspe']])
        
        titles = {
            'en': {
                'title': 'Distribution of Error Metrics',
                'ylabel': 'Value'
            },
            'lv': {
                'title': 'Kļūdu metriku sadalījums',
                'ylabel': 'Vērtība'
            }
        }
        
        plt.title(titles[lang]['title'])
        plt.ylabel(titles[lang]['ylabel'])
        plt.xticks(rotation=45)
        plt.savefig(f'Tests/error_boxplots_{lang}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error_distributions(self, df, lang='en'):
        plt.figure(figsize=(15, 5))
        
        titles = {
            'en': {'title': 'Distribution', 'density': 'Density'},
            'lv': {'title': 'Sadalījums', 'density': 'Blīvums'}
        }
        
        metrics = ['rmse', 'mae', 'mape']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, i)
            sns.histplot(df[metric], kde=True)
            plt.title(f'{metric.upper()} {titles[lang]["title"]}')
            plt.xlabel(metric.upper())
            plt.ylabel(titles[lang]['density'])
            
        plt.tight_layout()
        plt.savefig(f'Tests/error_distributions_{lang}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_metric_correlations(self, df, lang='en'):
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[['rmse', 'mae', 'mape', 'rmspe', 'num_outliers']].corr()
        
        titles = {
            'en': {'title': 'Metric Correlation Matrix'},
            'lv': {'title': 'Metriku korelācijas matrica'}
        }
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title(titles[lang]['title'])
        plt.savefig(f'Tests/metric_correlations_{lang}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _print_summary_statistics(self, df):
        print("\nSummary Statistics:")
        print("=" * 50)
        
        metrics = ['rmse', 'mae', 'mape', 'rmspe']
        for metric in metrics:
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {df[metric].mean():.3f}")
            print(f"  Std: {df[metric].std():.3f}")
            print(f"  Min: {df[metric].min():.3f}")
            print(f"  Max: {df[metric].max():.3f}")
            print(f"  Median: {df[metric].median():.3f}")

if __name__ == "__main__":
    tester = OutlierMultiTester(
        data_path='Data/Test_dati_sprosti_2cikls.xlsx',
        target_column='Average Temperature Indoor',
        num_tests=100
    )
    
    tester.run_multi_test()
    
    tester.analyze_results()
