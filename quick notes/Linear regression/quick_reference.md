# Linear Regression Quick Reference

## Side-by-Side Comparison

| Feature | Simple Linear Regression | Multiple Linear Regression |
|---------|--------------------------|----------------------------|
| **Formula** | Y = β₀ + β₁X + ε | Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε |
| **Predictors** | 1 | 2 or more |
| **Visual** | 2D line | Multi-dimensional hyperplane |
| **Complexity** | Low | High |
| **Interpretability** | Very High | Medium |
| **Common Accuracy** | Lower | Higher (usually) |
| **Data Requirements** | Small datasets OK | Larger datasets preferred |
| **Overfitting Risk** | Low | High |
| **Real-world Use** | Limited | Very Common |

---

## Example Results from Code Execution

### Simple Linear Regression
```
Dataset: 10 points
Equation: y = 1.4860 + 2.4931x
R² Score: 0.9909 (99.09% variance explained)
RMSE: 0.6856
MAE: 0.5737
```

**Interpretation:**
- For every 1 unit increase in X, Y increases by ~2.49 units
- The model explains almost all the variance (excellent fit)
- Average prediction error is 0.69 units

---

### Multiple Linear Regression
```
Dataset: 100 house prices
Features: Age, Square Feet, Number of Rooms
Training R²: 0.9756
Testing R²: 0.9341

Equation:
Price = -8020.60 + 382.02×Age + 4.05×Size + 20472.47×Rooms

Coefficients:
  Age:   $382 per year older
  Size:  $4.05 per sq ft
  Rooms: $20,472 per additional room
```

**Interpretation:**
- A house gains ~$382 per year of age (depreciation)
- Each sq ft adds ~$4 to the price
- Each additional room adds ~$20,472
- R² = 0.93 means the model explains 93% of price variation

---

## Performance Comparison in Example

Using house price data:

**Using Only Size (Simple):**
- R²: 0.0049 (very poor)
- RMSE: $53,763 (huge errors)

**Using Age + Size + Rooms (Multiple):**
- R²: 0.9341 (excellent)
- RMSE: $14,496 (much better)

**Improvement: 18,930% increase in R²**

---

## Key Formulas

### Slope for Simple Regression
```
β₁ = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ[(xᵢ - x̄)²]
```

### Intercept
```
β₀ = ȳ - β₁x̄
```

### R-squared
```
R² = 1 - (SSres / SStot)
   = 1 - (Σ(y - ŷ)² / Σ(y - ȳ)²)
```

### RMSE (Root Mean Square Error)
```
RMSE = √(Σ(y - ŷ)² / n)
```

### MAE (Mean Absolute Error)
```
MAE = Σ|y - ŷ| / n
```

---

## When to Use Each

### Use Simple Linear Regression When:
✓ You want to understand the relationship between 2 variables
✓ You're doing exploratory analysis
✓ You have limited data
✓ Simplicity and interpretability matter most
✓ You know only one variable affects the outcome

### Use Multiple Linear Regression When:
✓ You have multiple factors affecting the outcome
✓ You want better predictive accuracy
✓ You have sufficient data (ideally n > 10 × number of predictors)
✓ You need to understand relative importance of variables
✓ You're building production models

---

## Assumptions to Check

1. **Linearity** - Plot Y vs each X, should show linear pattern
2. **Independence** - Observations should be independent
3. **Homoscedasticity** - Residuals should have constant variance
4. **Normality** - Residuals should be normally distributed
5. **No Multicollinearity** (multiple only) - Predictors shouldn't be highly correlated

**How to Check:**
- Residual plot: Should show random scatter around zero
- Q-Q plot: Points should lie on the diagonal line
- Correlation matrix: No correlations > 0.8 (for multiple)

---

## Common Issues and Fixes

| Problem | Cause | Solution |
|---------|-------|----------|
| Low R² | Missing important variables | Add relevant predictors |
| High RMSE | Poor model fit | Check assumptions, try transformations |
| Non-random residuals | Non-linear relationship | Try polynomial regression |
| High correlations | Multicollinearity | Remove correlated variables |
| Large train-test gap | Overfitting | Use regularization (Ridge/Lasso) |
| Non-normal residuals | Outliers present | Check for and handle outliers |

---

## Python Implementation Quick Start

### Simple Linear Regression
```python
from sklearn.linear_model import LinearRegression

# Prepare data
X = [[1], [2], [3], [4], [5]]  # Must be 2D
y = [2, 4, 5, 4, 5]

# Train
model = LinearRegression()
model.fit(X, y)

# Predict
predictions = model.predict([[6]])

# Evaluate
r2 = model.score(X, y)
print(f"y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x")
print(f"R² = {r2:.4f}")
```

### Multiple Linear Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Prepare data (n_samples × n_features)
X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y = [10, 15, 20, 25, 30]

# Train
model = LinearRegression()
model.fit(X, y)

# Predict
predictions = model.predict([[6, 7]])

# Evaluate
r2 = model.score(X, y)
rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
print(f"Intercept: {model.intercept_:.4f}")
print(f"Coefficients: {model.coef_}")
print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}")
```

---

## Evaluation Metrics Interpretation

| Metric | Range | Interpretation | Goal |
|--------|-------|-----------------|------|
| **R²** | 0-1 | % of variance explained | Maximize (close to 1) |
| **Adj R²** | 0-1 | R² penalized for variables | Maximize |
| **RMSE** | 0-∞ | Average error (same units as Y) | Minimize |
| **MAE** | 0-∞ | Average absolute error | Minimize |
| **MSE** | 0-∞ | Average squared error | Minimize |

**Rule of Thumb:**
- R² > 0.7 = Good fit
- R² > 0.5 = Acceptable fit
- R² < 0.3 = Poor fit

---

## Diagnostic Plots Explained

### 1. Fitted vs Actual
- Points close to red line = good predictions
- Systematic deviations = model doesn't capture all patterns

### 2. Residual Plot
- Random scatter around zero = good
- Funnel shape = heteroscedasticity (violations of homoscedasticity)
- Clear pattern = non-linear relationship

### 3. Residual Distribution
- Bell curve shape = normal distribution (good)
- Skewed or multiple peaks = may violate normality assumption

---

## Advanced Topics (Beyond Basics)

- **Polynomial Regression** - Use polynomial terms for non-linear relationships
- **Ridge/Lasso Regression** - Add penalties to prevent overfitting
- **Elastic Net** - Combines Ridge and Lasso
- **Stepwise Selection** - Automated feature selection
- **Regularization** - Shrink coefficients toward zero
- **Cross-Validation** - Robust model evaluation

---

## Summary

**Simple Linear Regression:**
- 1 predictor → Easy to understand & interpret
- Limited accuracy → Use for exploration
- Few assumptions to check

**Multiple Linear Regression:**
- Multiple predictors → Better accuracy
- More complex → More assumptions to validate
- Standard choice for real-world problems

**Key Takeaway:** Always start simple, add complexity only when needed, and always validate assumptions!
