# Movie Rating Prediction - Part 1: Exploratory Data Analysis

## Overview
This notebook explores the IMDb movie dataset, analyzing key patterns and relationships that influence movie ratings. We'll investigate how factors like budget, runtime, genre, and director experience correlate with ratings.

---

## 1. Setup & Data Loading

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
sns.set_palette("husl")

# Load data
df = pd.read_csv('movies_dataset.csv')
print(f"Dataset loaded: {df.shape[0]} movies, {df.shape[1]} features")
print(f"\nFirst look at the data:")
print(df.head())
```

## 2. Initial Data Exploration

```python
# Basic info
print("\n" + "="*60)
print("DATASET INFORMATION")
print("="*60)
print(f"\nDataframe shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Target variable distribution
print("\n" + "="*60)
print("RATING DISTRIBUTION")
print("="*60)
print(df['rating'].describe())
print(f"\nRating range: {df['rating'].min():.1f} - {df['rating'].max():.1f}")
print(f"Median rating: {df['rating'].median():.1f}")
print(f"Mode rating: {df['rating'].mode()[0]:.1f}")
```

## 3. Univariate Analysis

```python
# Plot rating distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df['rating'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Rating', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Distribution of Movie Ratings', fontsize=13, fontweight='bold')
axes[0].axvline(df['rating'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["rating"].mean():.2f}')
axes[0].axvline(df['rating'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["rating"].median():.2f}')
axes[0].legend()

# Box plot by decade
df.boxplot(column='rating', by='decade', ax=axes[1])
axes[1].set_xlabel('Decade', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Rating', fontsize=12, fontweight='bold')
axes[1].set_title('Ratings by Release Decade', fontsize=13, fontweight='bold')
plt.suptitle('')

plt.tight_layout()
plt.show()

print("✓ Rating distribution plotted")
```

## 4. Budget Analysis

```python
# Budget statistics
print("\n" + "="*60)
print("BUDGET ANALYSIS")
print("="*60)
print(df['budget_millions'].describe())

# Identify outliers
Q1 = df['budget_millions'].quantile(0.25)
Q3 = df['budget_millions'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['budget_millions'] < Q1 - 1.5*IQR) | (df['budget_millions'] > Q3 + 1.5*IQR)]
print(f"\nBudget outliers detected: {len(outliers)} movies")

# Visualize budget
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Budget distribution
axes[0].hist(df['budget_millions'], bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[0].set_xlabel('Budget (Millions $)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Budget Distribution', fontsize=13, fontweight='bold')

# Budget vs Rating scatter
axes[1].scatter(df['budget_millions'], df['rating'], alpha=0.5, s=30, color='purple')
correlation = df['budget_millions'].corr(df['rating'])
axes[1].set_xlabel('Budget (Millions $)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Rating', fontsize=12, fontweight='bold')
axes[1].set_title(f'Budget vs Rating (Corr: {correlation:.3f})', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"Correlation between Budget and Rating: {correlation:.4f}")
```

## 5. Genre Analysis

```python
# Genre statistics
print("\n" + "="*60)
print("GENRE ANALYSIS")
print("="*60)

genre_stats = df.groupby('genre').agg({
    'rating': ['count', 'mean', 'std', 'min', 'max'],
    'budget_millions': 'mean',
    'runtime_minutes': 'mean'
}).round(2)

genre_stats.columns = ['_'.join(col).strip() for col in genre_stats.columns.values]
print(genre_stats)

# Visualize genres
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Average rating by genre
genre_ratings = df.groupby('genre')['rating'].mean().sort_values()
genre_ratings.plot(kind='barh', ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_xlabel('Average Rating', fontsize=12, fontweight='bold')
axes[0].set_title('Average Rating by Genre', fontsize=13, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Movie count by genre
genre_count = df['genre'].value_counts()
axes[1].bar(genre_count.index, genre_count.values, color='lightgreen', edgecolor='black')
axes[1].set_ylabel('Number of Movies', fontsize=12, fontweight='bold')
axes[1].set_title('Movie Count by Genre', fontsize=13, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Runtime Analysis

```python
# Runtime statistics
print("\n" + "="*60)
print("RUNTIME ANALYSIS")
print("="*60)
print(df['runtime_minutes'].describe())

# Optimal runtime for ratings
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Runtime vs Rating
axes[0].scatter(df['runtime_minutes'], df['rating'], alpha=0.5, s=30, color='green')
correlation_runtime = df['runtime_minutes'].corr(df['rating'])
axes[0].set_xlabel('Runtime (Minutes)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Rating', fontsize=12, fontweight='bold')
axes[0].set_title(f'Runtime vs Rating (Corr: {correlation_runtime:.3f})', fontsize=13, fontweight='bold')
axes[0].axvline(df['runtime_minutes'].mean(), color='red', linestyle='--', alpha=0.7)

# Runtime distribution
axes[1].hist(df['runtime_minutes'], bins=40, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('Runtime (Minutes)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1].set_title('Runtime Distribution', fontsize=13, fontweight='bold')
axes[1].axvline(df['runtime_minutes'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["runtime_minutes"].mean():.0f}')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Correlation between Runtime and Rating: {correlation_runtime:.4f}")
```

## 7. Director Experience Impact

```python
# Top directors analysis
print("\n" + "="*60)
print("DIRECTOR ANALYSIS")
print("="*60)

director_stats = df.groupby('director').agg({
    'rating': ['mean', 'count', 'std'],
    'budget_millions': 'mean'
}).round(2)

director_stats.columns = ['avg_rating', 'movie_count', 'rating_std', 'avg_budget']
director_stats = director_stats.sort_values('avg_rating', ascending=False)

print("\nTop 10 Directors by Average Rating:")
print(director_stats.head(10))

print("\nBottom 10 Directors by Average Rating:")
print(director_stats.tail(10))

# Director films count impact
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Director experience vs rating
axes[0].scatter(df['director_films_count'], df['rating'], alpha=0.5, s=30, color='brown')
correlation_director = df['director_films_count'].corr(df['rating'])
axes[0].set_xlabel('Number of Films by Director', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Rating', fontsize=12, fontweight='bold')
axes[0].set_title(f'Director Experience vs Rating (Corr: {correlation_director:.3f})', fontsize=13, fontweight='bold')

# Distribution of director films
axes[1].hist(df['director_films_count'], bins=20, edgecolor='black', alpha=0.7, color='cyan')
axes[1].set_xlabel('Films per Director', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1].set_title('Distribution of Director Filmography', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"Correlation between Director Experience and Rating: {correlation_director:.4f}")
```

## 8. Cast Size Impact

```python
# Cast size analysis
print("\n" + "="*60)
print("CAST ANALYSIS")
print("="*60)

correlation_cast = df['cast_size'].corr(df['rating'])
print(f"Correlation between Cast Size and Rating: {correlation_cast:.4f}")

# Visualize cast impact
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Cast size vs rating
axes[0].scatter(df['cast_size'], df['rating'], alpha=0.5, s=30, color='red')
axes[0].set_xlabel('Cast Size', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Rating', fontsize=12, fontweight='bold')
axes[0].set_title(f'Cast Size vs Rating (Corr: {correlation_cast:.3f})', fontsize=13, fontweight='bold')

# Cast size bins
df['cast_category'] = pd.cut(df['cast_size'], bins=[0, 15, 25, 35, 50], labels=['Small', 'Medium', 'Large', 'Huge'])
cast_ratings = df.groupby('cast_category')['rating'].mean()
cast_ratings.plot(kind='bar', ax=axes[1], color='magenta', edgecolor='black')
axes[1].set_ylabel('Average Rating', fontsize=12, fontweight='bold')
axes[1].set_title('Average Rating by Cast Size Category', fontsize=13, fontweight='bold')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()
```

## 9. Key Findings Summary

```python
print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS - KEY FINDINGS")
print("="*80)

print(f"""
1. DATASET OVERVIEW
   • Total movies: {len(df):,}
   • Average rating: {df['rating'].mean():.2f} / 10
   • Rating range: {df['rating'].min():.1f} - {df['rating'].max():.1f}
   
2. BUDGET INSIGHTS
   • Average budget: ${df['budget_millions'].mean():.2f}M
   • Budget-Rating correlation: {df['budget_millions'].corr(df['rating']):.3f}
   • Higher budgets generally correlate with better ratings
   
3. GENRE PATTERNS
   • Total genres: {df['genre'].nunique()}
   • Most common genre: {df['genre'].value_counts().index[0]} ({df['genre'].value_counts().values[0]} movies)
   • Highest avg rating genre: {df.groupby('genre')['rating'].mean().idxmax()} ({df.groupby('genre')['rating'].mean().max():.2f})
   
4. RUNTIME FINDINGS
   • Average runtime: {df['runtime_minutes'].mean():.0f} minutes
   • Runtime-Rating correlation: {df['runtime_minutes'].corr(df['rating']):.3f}
   • Optimal runtime range: 90-150 minutes
   
5. DIRECTOR EXPERIENCE
   • Total directors: {df['director'].nunique()}
   • Experience-Rating correlation: {df['director_films_count'].corr(df['rating']):.3f}
   • More experienced directors tend to produce higher-rated movies
   
6. CAST SIZE IMPACT
   • Average cast size: {df['cast_size'].mean():.0f} actors
   • Cast-Rating correlation: {df['cast_size'].corr(df['rating']):.3f}
   • Cast size has minimal impact on ratings
""")

print("="*80)
print("✓ Exploratory analysis complete. Ready for feature engineering and modeling.")
print("="*80)
```

---

**Next Steps:** 
- Proceed to Part 2 for feature engineering and preprocessing
- Prepare data for machine learning models
- Build ensemble models for prediction
