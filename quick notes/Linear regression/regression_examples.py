"""
Complete Linear Regression Examples
Simple & Multiple Linear Regression with Python
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("LINEAR REGRESSION: SIMPLE AND MULTIPLE")
print("=" * 80)

# ============================================================================
# PART 1: SIMPLE LINEAR REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: SIMPLE LINEAR REGRESSION")
print("=" * 80)

# Create sample data
np.random.seed(42)
X_simple = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y_simple = 2.5 * X_simple.flatten() + 1 + np.random.normal(0, 1, 10)

print("\nDataset:")
print(f"Independent variable (X): {X_simple.flatten()}")
print(f"Dependent variable (y):   {np.round(y_simple, 2)}")

# Train simple linear regression
model_simple = LinearRegression()
model_simple.fit(X_simple, y_simple)

# Predictions
y_pred_simple_plot = model_simple.predict(X_simple)

# Coefficients
intercept_simple = model_simple.intercept_
slope_simple = model_simple.coef_[0]

print("\n--- Simple Linear Regression Results ---")
print(f"Equation: y = {intercept_simple:.4f} + {slope_simple:.4f}x")
print(f"Intercept (β₀): {intercept_simple:.4f}")
print(f"Slope (β₁):     {slope_simple:.4f}")
print(f"R² Score:       {model_simple.score(X_simple, y_simple):.4f}")
print(f"RMSE:           {np.sqrt(mean_squared_error(y_simple, y_pred_simple_plot)):.4f}")
print(f"MAE:            {mean_absolute_error(y_simple, y_pred_simple_plot):.4f}")

# ============================================================================
# PART 2: MULTIPLE LINEAR REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: MULTIPLE LINEAR REGRESSION")
print("=" * 80)

# Create sample data with 3 predictors
np.random.seed(42)
n_samples = 100

# Generate features
X1 = np.random.uniform(20, 40, n_samples)  # Age
X2 = np.random.uniform(1000, 5000, n_samples)  # Square feet
X3 = np.random.uniform(1, 10, n_samples)  # Rooms

# Create target variable
y_multiple = (5000 + 
              50 * X1 +           # Age effect
              3 * X2 +            # Size effect
              20000 * X3 +        # Rooms effect
              np.random.normal(0, 10000, n_samples))  # Noise

# Combine into DataFrame
df = pd.DataFrame({
    'Age': X1,
    'Size_sqft': X2,
    'Rooms': X3,
    'Price': y_multiple
})

print("\nDataset Summary:")
print(df.describe())

# Prepare data
X_multiple = df[['Age', 'Size_sqft', 'Rooms']].values
y = df['Price'].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_multiple, y, test_size=0.2, random_state=42
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size:  {X_test.shape[0]}")

# Train multiple linear regression
model_multiple = LinearRegression()
model_multiple.fit(X_train, y_train)

# Predictions
y_train_pred = model_multiple.predict(X_train)
y_test_pred = model_multiple.predict(X_test)

print("\n--- Multiple Linear Regression Results ---")
print(f"Intercept (β₀): {model_multiple.intercept_:.4f}")
print(f"\nCoefficients:")
print(f"  β₁ (Age):      {model_multiple.coef_[0]:>10.4f}")
print(f"  β₂ (Size):     {model_multiple.coef_[1]:>10.4f}")
print(f"  β₃ (Rooms):    {model_multiple.coef_[2]:>10.4f}")

print(f"\nEquation:")
print(f"Price = {model_multiple.intercept_:.2f} + " +
      f"{model_multiple.coef_[0]:.2f}*Age + " +
      f"{model_multiple.coef_[1]:.2f}*Size + " +
      f"{model_multiple.coef_[2]:.2f}*Rooms")

# Evaluation metrics - Training
r2_train = model_multiple.score(X_train, y_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
mae_train = mean_absolute_error(y_train, y_train_pred)

# Evaluation metrics - Testing
r2_test = model_multiple.score(X_test, y_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"\n--- Training Metrics ---")
print(f"R² Score: {r2_train:.4f}")
print(f"RMSE:     {rmse_train:,.2f}")
print(f"MAE:      {mae_train:,.2f}")

print(f"\n--- Testing Metrics ---")
print(f"R² Score: {r2_test:.4f}")
print(f"RMSE:     {rmse_test:,.2f}")
print(f"MAE:      {mae_test:,.2f}")

# ============================================================================
# PART 3: RESIDUAL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: RESIDUAL ANALYSIS")
print("=" * 80)

# Calculate residuals
residuals = y_test - y_test_pred

print(f"\nResiduals Statistics:")
print(f"Mean:       {np.mean(residuals):>10.4f} (should be ≈ 0)")
print(f"Std Dev:    {np.std(residuals):>10.4f}")
print(f"Min:        {np.min(residuals):>10,.2f}")
print(f"Max:        {np.max(residuals):>10,.2f}")

# ============================================================================
# PART 4: PREDICTIONS ON NEW DATA
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: PREDICTIONS ON NEW DATA")
print("=" * 80)

# New sample data
new_data = np.array([
    [30, 2500, 4],  # Age=30, Size=2500 sqft, Rooms=4
    [25, 3000, 5],  # Age=25, Size=3000 sqft, Rooms=5
    [35, 1800, 3],  # Age=35, Size=1800 sqft, Rooms=3
])

new_df = pd.DataFrame(new_data, columns=['Age', 'Size_sqft', 'Rooms'])
predictions = model_multiple.predict(new_data)

print("\nNew Samples:")
print(new_df)
print("\nPredicted Prices:")
for i, (idx, row) in enumerate(new_df.iterrows()):
    print(f"  Sample {i+1}: ${predictions[i]:>12,.2f}")

# ============================================================================
# PART 5: COMPARISON - SIMPLE VS MULTIPLE
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: COMPARISON - SIMPLE VS MULTIPLE REGRESSION")
print("=" * 80)

# Train simple regression with just Size
X_size_only = df[['Size_sqft']].values
model_simple_size = LinearRegression()
model_simple_size.fit(X_size_only, y)

y_pred_simple = model_simple_size.predict(X_size_only)
r2_simple = model_simple_size.score(X_size_only, y)
rmse_simple = np.sqrt(mean_squared_error(y, y_pred_simple))

print("\nUsing ONLY Size (Simple Regression):")
print(f"  R² Score: {r2_simple:.4f}")
print(f"  RMSE:     {rmse_simple:,.2f}")

print("\nUsing Age + Size + Rooms (Multiple Regression):")
print(f"  R² Score: {r2_test:.4f}")
print(f"  RMSE:     {rmse_test:,.2f}")

improvement = ((r2_test - r2_simple) / r2_simple) * 100
print(f"\nImprovement: {improvement:.2f}% increase in R²")

# ============================================================================
# PART 6: VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("Creating visualizations...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Simple Linear Regression
ax1 = axes[0, 0]
ax1.scatter(X_simple.flatten(), y_simple, color='blue', alpha=0.6, label='Actual')
ax1.plot(X_simple.flatten(), y_pred_simple_plot, color='red', linewidth=2, label='Fitted line')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Simple Linear Regression')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Multiple Regression - Actual vs Predicted
ax2 = axes[0, 1]
ax2.scatter(y_test, y_test_pred, alpha=0.6, color='green')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Price')
ax2.set_ylabel('Predicted Price')
ax2.set_title('Multiple Regression: Actual vs Predicted')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Residuals
ax3 = axes[1, 0]
ax3.scatter(y_test_pred, residuals, alpha=0.6, color='purple')
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted Price')
ax3.set_ylabel('Residuals')
ax3.set_title('Residual Plot (Multiple Regression)')
ax3.grid(True, alpha=0.3)

# 4. Residuals Distribution
ax4 = axes[1, 1]
ax4.hist(residuals, bins=15, color='orange', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Residuals')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Residuals')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/regression_plots.png', dpi=300, bbox_inches='tight')
print("✓ Plots saved as 'regression_plots.png'")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
