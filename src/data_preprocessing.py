"""Data Preprocessing Module

This module handles all data preprocessing tasks including:
- Data loading and validation
- Missing value imputation
- Feature scaling and normalization
- Outlier detection and treatment
- Data splitting for train/test
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self, data_path=None):
        """
        Initialize the DataPreprocessor
        
        Args:
            data_path (str): Path to the data file
        """
        self.data_path = data_path
        self.data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, path=None):
        """
        Load data from CSV file
        
        Args:
            path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        if path:
            self.data_path = path
            
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
            
    def handle_missing_values(self, strategy='mean', columns=None):
        """
        Handle missing values in the dataset
        
        Args:
            strategy (str): Strategy to use ('mean', 'median', 'mode', 'drop')
            columns (list): List of columns to process
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
            
        if columns is None:
            columns = self.data.columns
            
        for col in columns:
            if self.data[col].isnull().sum() > 0:
                if strategy == 'mean':
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.data[col].fillna(self.data[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    self.data.dropna(subset=[col], inplace=True)
                    
        print(f"Missing values handled using {strategy} strategy")
        
    def encode_categorical(self, columns):
        """
        Encode categorical variables
        
        Args:
            columns (list): List of categorical columns to encode
        """
        for col in columns:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                
        print(f"Encoded {len(columns)} categorical columns")
        
    def scale_features(self, columns):
        """
        Scale numerical features
        
        Args:
            columns (list): List of columns to scale
        """
        if columns:
            self.data[columns] = self.scaler.fit_transform(self.data[columns])
            print(f"Scaled {len(columns)} features")
            
    def split_data(self, target_column, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            target_column (str): Name of the target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Data split: Train={X_train.shape}, Test={X_test.shape}")
        return X_train, X_test, y_train, y_test
        
    def get_data_info(self):
        """
        Get information about the dataset
        """
        if self.data is not None:
            print("\nDataset Information:")
            print(f"Shape: {self.data.shape}")
            print(f"\nColumn Types:\n{self.data.dtypes}")
            print(f"\nMissing Values:\n{self.data.isnull().sum()}")
            print(f"\nBasic Statistics:\n{self.data.describe()}")
        else:
            print("No data loaded")


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    print("Data Preprocessing Module loaded successfully")
