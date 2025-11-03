# Movie Rating Prediction - Part 2: Model Building & Evaluation

## Overview
Building ensemble machine learning models to predict IMDb movie ratings. We'll compare Random Forest and Gradient Boosting, optimize hyperparameters, and evaluate performance.

---

## 1. Data Preparation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('movies_dataset.csv')
print(f"Dataset loaded: {df.shape}")

# Display basic stats
print(f"\nTarget variable (Rating) statistics:")
print(df['rating'].describe())
```

## 2. Feature Engineering & Preprocessing

```python
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# Create a copy for processing
X = df.copy()
y = X.pop('rating')

# Encode categorical variables
le_genre = LabelEncoder()
X['genre_encoded'] = le_genre.fit_transform(X['genre'])

le_production = LabelEncoder()
X['production_encoded'] = le_production.fit_transform(X['production_company'])

le_director = LabelEncoder()
X['director_encoded'] = le_director.fit_transform(X['director'])

# Select numerical features for modeling
feature_cols = ['genre_encoded', 'budget_millions', 'runtime_minutes', 'cast_size', 
                'director_films_count', 'budget_per_minute', 'production_encoded']

X = X[feature_cols]

print(f"Features selected: {len(feature_cols)}")
print(f"Features: {feature_cols}")
print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Check for missing values
print(f"\nMissing values in features:")
print(X.isnull().sum())

# Basic statistics
print(f"\nFeature statistics:")
print(X.describe())
```

## 3. Train-Test Split

```python
print("\n" + "="*60)
print("TRAIN-TEST SPLIT & SCALING")
print("="*60)

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Features scaled using StandardScaler")

# Save scaler for later use
joblib.dump(scaler, 'scaler.pkl')
```

## 4. K-Fold Cross-Validation Setup

```python
print("\n" + "="*60)
print("CROSS-VALIDATION SETUP")
print("="*60)

# Create k-fold cross-validator
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"K-Fold Cross-Validation: k={kfold.n_splits}")
print(f"Splits: {kfold.get_n_splits()}")
print("\nThis ensures robust evaluation and prevents overfitting")
```

## 5. Model 1: Random Forest Regressor

```python
print("\n" + "="*80)
print("MODEL 1: RANDOM FOREST REGRESSOR")
print("="*80)

# Initialize Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("\nHyperparameters:")
print(f"  • n_estimators: 100")
print(f"  • max_depth: 15")
print(f"  • min_samples_split: 5")
print(f"  • min_samples_leaf: 2")

# Train model
print("\nTraining Random Forest...")
rf_model.fit(X_train_scaled, y_train)
print("✓ Training complete")

# Predictions
rf_train_pred = rf_model.predict(X_train_scaled)
rf_test_pred = rf_model.predict(X_test_scaled)

# Evaluation metrics
rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_test_r2 = r2_score(y_test, rf_test_pred)
rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
rf_test_mape = mean_absolute_percentage_error(y_test, rf_test_pred)

print(f"\nPerformance Metrics:")
print(f"  Train R² Score: {rf_train_r2:.4f}")
print(f"  Test R² Score:  {rf_test_r2:.4f}")
print(f"  Train MAE:      {rf_train_mae:.4f}")
print(f"  Test MAE:       {rf_test_mae:.4f}")
print(f"  Test RMSE:      {rf_test_rmse:.4f}")
print(f"  Test MAPE:      {rf_test_mape:.4f}")

# Cross-validation
rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=kfold, scoring='r2')
print(f"\nCross-Validation Results:")
print(f"  CV Scores: {rf_cv_scores}")
print(f"  CV Mean:   {rf_cv_scores.mean():.4f}")
print(f"  CV Std:    {rf_cv_scores.std():.4f}")

# Feature importance
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Important Features:")
print(rf_importance.head())

# Save model
joblib.dump(rf_model, 'rf_model.pkl')
```

## 6. Model 2: Gradient Boosting Regressor

```python
print("\n" + "="*80)
print("MODEL 2: GRADIENT BOOSTING REGRESSOR")
print("="*80)

# Initialize Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    verbose=0
)

print("\nHyperparameters:")
print(f"  • n_estimators: 100")
print(f"  • learning_rate: 0.1")
print(f"  • max_depth: 5")
print(f"  • min_samples_split: 5")
print(f"  • min_samples_leaf: 2")

# Train model
print("\nTraining Gradient Boosting...")
gb_model.fit(X_train_scaled, y_train)
print("✓ Training complete")

# Predictions
gb_train_pred = gb_model.predict(X_train_scaled)
gb_test_pred = gb_model.predict(X_test_scaled)

# Evaluation metrics
gb_train_r2 = r2_score(y_train, gb_train_pred)
gb_test_r2 = r2_score(y_test, gb_test_pred)
gb_train_mae = mean_absolute_error(y_train, gb_train_pred)
gb_test_mae = mean_absolute_error(y_test, gb_test_pred)
gb_test_rmse = np.sqrt(mean_squared_error(y_test, gb_test_pred))
gb_test_mape = mean_absolute_percentage_error(y_test, gb_test_pred)

print(f"\nPerformance Metrics:")
print(f"  Train R² Score: {gb_train_r2:.4f}")
print(f"  Test R² Score:  {gb_test_r2:.4f}")
print(f"  Train MAE:      {gb_train_mae:.4f}")
print(f"  Test MAE:       {gb_test_mae:.4f}")
print(f"  Test RMSE:      {gb_test_rmse:.4f}")
print(f"  Test MAPE:      {gb_test_mape:.4f}")

# Cross-validation
gb_cv_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=kfold, scoring='r2')
print(f"\nCross-Validation Results:")
print(f"  CV Scores: {gb_cv_scores}")
print(f"  CV Mean:   {gb_cv_scores.mean():.4f}")
print(f"  CV Std:    {gb_cv_scores.std():.4f}")

# Feature importance
gb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Important Features:")
print(gb_importance.head())

# Save model
joblib.dump(gb_model, 'gb_model.pkl')
```

## 7. Hyperparameter Tuning

```python
print("\n" + "="*80)
print("HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("="*80)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [3, 5, 7]
}

print(f"\nParameter grid:")
print(f"  • n_estimators: {param_grid['n_estimators']}")
print(f"  • learning_rate: {param_grid['learning_rate']}")
print(f"  • max_depth: {param_grid['max_depth']}")
print(f"  • min_samples_split: {param_grid['min_samples_split']}")

# Grid Search
print(f"\nRunning GridSearchCV with 5-fold cross-validation...")
print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\n✓ Grid Search Complete")
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV R² Score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_

# Predictions with best model
best_train_pred = best_model.predict(X_train_scaled)
best_test_pred = best_model.predict(X_test_scaled)

# Evaluation
best_train_r2 = r2_score(y_train, best_train_pred)
best_test_r2 = r2_score(y_test, best_test_pred)
best_test_mae = mean_absolute_error(y_test, best_test_pred)
best_test_rmse = np.sqrt(mean_squared_error(y_test, best_test_pred))
best_test_mape = mean_absolute_percentage_error(y_test, best_test_pred)

print(f"\nBest Model Performance:")
print(f"  Train R² Score: {best_train_r2:.4f}")
print(f"  Test R² Score:  {best_test_r2:.4f}")
print(f"  Test MAE:       {best_test_mae:.4f}")
print(f"  Test RMSE:      {best_test_rmse:.4f}")
print(f"  Test MAPE:      {best_test_mape:.4f}")

# Save best model
joblib.dump(best_model, 'best_model.pkl')
```

## 8. Model Comparison & Visualization

```python
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

# Create comparison table
comparison_data = {
    'Model': ['Random Forest', 'Gradient Boosting', 'Tuned GB'],
    'Train R²': [rf_train_r2, gb_train_r2, best_train_r2],
    'Test R²': [rf_test_r2, gb_test_r2, best_test_r2],
    'Test MAE': [rf_test_mae, gb_test_mae, best_test_mae],
    'Test RMSE': [rf_test_rmse, gb_test_rmse, best_test_rmse],
    'CV Mean': [rf_cv_scores.mean(), gb_cv_scores.mean(), grid_search.best_score_]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n", comparison_df.to_string(index=False))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Model R² Comparison
models = comparison_data['Model']
test_r2_scores = comparison_data['Test R²']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
axes[0, 0].bar(models, test_r2_scores, color=colors, edgecolor='black', alpha=0.7)
axes[0, 0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Model Test R² Score Comparison', fontsize=13, fontweight='bold')
axes[0, 0].set_ylim([0, 1])
for i, v in enumerate(test_r2_scores):
    axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Actual vs Predicted (Best Model)
axes[0, 1].scatter(y_test, best_test_pred, alpha=0.5, s=30, edgecolors='k')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Rating', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Predicted Rating', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Actual vs Predicted Ratings (Best Model)', fontsize=13, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Residuals Distribution
residuals = y_test - best_test_pred
axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='purple')
axes[1, 0].set_xlabel('Residuals', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Residuals Distribution', fontsize=13, fontweight='bold')
axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].grid(alpha=0.3)

# Plot 4: Feature Importance
best_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=True)

axes[1, 1].barh(best_importance['feature'], best_importance['importance'], color='teal', edgecolor='black')
axes[1, 1].set_xlabel('Importance', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Feature Importance (Best Model)', fontsize=13, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'model_performance.png'")
plt.show()
```

## 9. Model Summary & Predictions

```python
print("\n" + "="*80)
print("FINAL MODEL SUMMARY")
print("="*80)

# Calculate accuracy as percentage
accuracy_percentage = best_test_r2 * 100

print(f"""
DATASET:
  • Total movies: {len(df):,}
  • Training samples: {len(X_train)}
  • Test samples: {len(X_test)}
  • Features used: {len(feature_cols)}

BEST MODEL CONFIGURATION:
  • Algorithm: Gradient Boosting Regressor
  • Parameters: {grid_search.best_params_}

PERFORMANCE METRICS:
  • Test R² Score: {best_test_r2:.4f}
  • Test Accuracy: {accuracy_percentage:.2f}%
  • Test MAE: {best_test_mae:.4f}
  • Test RMSE: {best_test_rmse:.4f}
  • Test MAPE: {best_test_mape:.4f}%
  • Cross-Val Mean: {grid_search.best_score_:.4f}

TOP 5 IMPORTANT FEATURES:
""")

for idx, row in best_importance.tail(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print("\n" + "="*80)
print("✓ Model training and evaluation complete!")
print("✓ Best model saved as 'best_model.pkl'")
print("✓ Model ready for deployment")
print("="*80)
```

---

**Model Performance Summary:**
- **Dataset:** 5,000 movies
- **Best Model Accuracy:** To be filled in with actual results
- **Features:** 7 engineered features
- **Cross-validation:** 5-fold CV for robust evaluation
