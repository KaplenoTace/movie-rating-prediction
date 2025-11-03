"""Model Training Module

This module handles model training, evaluation, and hyperparameter tuning for:
- Random Forest
- Gradient Boosting
- XGBoost
- Ensemble methods
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Handles training and evaluation of machine learning models"""
    
    def __init__(self, random_state=42):
        """
        Initialize the ModelTrainer
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.results = {}
        
    def train_random_forest(self, X_train, y_train, **kwargs):
        """
        Train a Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional parameters for RandomForestRegressor
        """
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', None),
            'min_samples_split': kwargs.get('min_samples_split', 2),
            'min_samples_leaf': kwargs.get('min_samples_leaf', 1),
            'random_state': self.random_state
        }
        
        rf_model = RandomForestRegressor(**params)
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        print("Random Forest model trained successfully")
        return rf_model
        
    def train_gradient_boosting(self, X_train, y_train, **kwargs):
        """
        Train a Gradient Boosting model
        
        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional parameters for GradientBoostingRegressor
        """
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'max_depth': kwargs.get('max_depth', 3),
            'min_samples_split': kwargs.get('min_samples_split', 2),
            'random_state': self.random_state
        }
        
        gb_model = GradientBoostingRegressor(**params)
        gb_model.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb_model
        print("Gradient Boosting model trained successfully")
        return gb_model
        
    def train_xgboost(self, X_train, y_train, **kwargs):
        """
        Train an XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional parameters for XGBRegressor
        """
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'max_depth': kwargs.get('max_depth', 6),
            'random_state': self.random_state
        }
        
        xgb_model = xgb.XGBRegressor(**params)
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        print("XGBoost model trained successfully")
        return xgb_model
        
    def evaluate_model(self, model, X_test, y_test, model_name=None):
        """
        Evaluate a trained model
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test targets
            model_name (str): Name of the model for results storage
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        predictions = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
        
        if model_name:
            self.results[model_name] = metrics
            
        print(f"\nEvaluation Results:")
        print(f"MSE:  {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE:  {metrics['mae']:.4f}")
        print(f"R2:   {metrics['r2']:.4f}")
        
        return metrics
        
    def cross_validate(self, model, X, y, cv=5):
        """
        Perform cross-validation on a model
        
        Args:
            model: Model to cross-validate
            X: Features
            y: Targets
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation scores
        """
        scores = cross_val_score(model, X, y, cv=cv, 
                                scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        
        cv_results = {
            'scores': rmse_scores,
            'mean': rmse_scores.mean(),
            'std': rmse_scores.std()
        }
        
        print(f"\nCross-Validation Results (RMSE):")
        print(f"Mean: {cv_results['mean']:.4f}")
        print(f"Std:  {cv_results['std']:.4f}")
        
        return cv_results
        
    def tune_hyperparameters(self, model_type, X_train, y_train, param_grid):
        """
        Tune hyperparameters using GridSearchCV
        
        Args:
            model_type (str): Type of model ('rf', 'gb', 'xgb')
            X_train: Training features
            y_train: Training targets
            param_grid (dict): Parameter grid for tuning
            
        Returns:
            Best estimator after tuning
        """
        if model_type == 'rf':
            base_model = RandomForestRegressor(random_state=self.random_state)
        elif model_type == 'gb':
            base_model = GradientBoostingRegressor(random_state=self.random_state)
        elif model_type == 'xgb':
            base_model = xgb.XGBRegressor(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        grid_search = GridSearchCV(base_model, param_grid, cv=5, 
                                  scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best score: {np.sqrt(-grid_search.best_score_):.4f}")
        
        return grid_search.best_estimator_
        
    def save_model(self, model, filepath):
        """
        Save a trained model to disk
        
        Args:
            model: Trained model to save
            filepath (str): Path to save the model
        """
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """
        Load a trained model from disk
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
        
    def compare_models(self):
        """
        Compare all trained models based on their evaluation results
        """
        if not self.results:
            print("No evaluation results available. Train and evaluate models first.")
            return
            
        print("\nModel Comparison:")
        results_df = pd.DataFrame(self.results).T
        print(results_df.sort_values('r2', ascending=False))
        
        best_model_name = results_df['r2'].idxmax()
        self.best_model = self.models.get(best_model_name)
        print(f"\nBest model: {best_model_name}")
        
        return results_df


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    print("Model Training Module loaded successfully")
