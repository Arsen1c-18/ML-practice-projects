# Cost Functions: Complete Summary

## What You Need to Know

### The Basics
A **cost function** (also called loss function) measures how wrong a model's predictions are:
```
Cost = (Error)² or |Error| or -log(probability) etc.
       ↓
    Lower cost = Better model
```

The goal of training is to **minimize the cost function** using optimization algorithms like gradient descent.

---

## Quick Decision Guide

### For Regression Problems:

```
Do you have outliers?
├─ NO  → Use MSE (most standard)
└─ YES → Use MAE or Huber Loss (more robust)

Are you using deep learning?
├─ YES → Consider Log-Cosh (smooth gradients)
└─ NO  → Use MSE or MAE
```

### For Classification Problems:

```
Is it binary or multi-class?
├─ BINARY
│   └─ Imbalanced? ─→ YES  → Use Focal Loss
│                    NO   → Use Binary Cross-Entropy
│
└─ MULTI-CLASS
    └─ Imbalanced? ─→ YES  → Use Focal Loss
                     NO   → Use Categorical Cross-Entropy
```

---

## The 10 Most Important Cost Functions

### REGRESSION (5 main ones)

#### 1. Mean Squared Error (MSE) ⭐⭐⭐
```python
MSE = average of squared errors
    = (1/m) × Σ(actual - predicted)²

When to use:
  ✓ Standard choice for regression
  ✓ Well-behaved data without outliers
  ✓ Need smooth optimization

When NOT to use:
  ✗ Data has extreme outliers
  ✗ Different error magnitudes matter differently
```

#### 2. Mean Absolute Error (MAE) ⭐⭐⭐
```python
MAE = average of absolute errors
    = (1/m) × Σ|actual - predicted|

When to use:
  ✓ Data has outliers
  ✓ Need interpretable error (same units)
  ✓ Linear penalty makes sense

When NOT to use:
  ✗ Need perfect smoothness (not differentiable at 0)
  ✗ Large errors should be heavily penalized
```

#### 3. Root Mean Squared Error (RMSE) ⭐⭐⭐
```python
RMSE = √MSE
     = √(average of squared errors)

When to use:
  ✓ Same as MSE but need same-unit interpretability
  ✓ Report results to non-technical people
  ✓ Evaluation metric (not for training)

Why RMSE over MSE?
  MSE = 4.0 (units are "dollars²" - what does that mean?)
  RMSE = 2.0 (units are "dollars" - much clearer!)
```

#### 4. Huber Loss ⭐⭐
```python
Huber = {
  0.5 × error²        if |error| ≤ 1.0
  |error| - 0.5       if |error| > 1.0
}

When to use:
  ✓ Safe middle-ground choice
  ✓ Have both normal data and some outliers
  ✓ Want smooth optimization AND robustness

Why it's great:
  • Smooth near zero (good for gradient descent)
  • Robust for large errors (like MAE)
  • Balances MSE and MAE benefits
```

#### 5. Log-Cosh Loss ⭐⭐
```python
Log-Cosh = log(cosh(error))

When to use:
  ✓ Deep learning regression models
  ✓ Need twice-differentiable function
  ✓ Smooth optimization matters

Why it's great:
  • Acts like MSE for small errors (smooth)
  • Acts like MAE for large errors (robust)
  • Smooth everywhere (better than Huber)
```

### CLASSIFICATION (5 main ones)

#### 1. Binary Cross-Entropy ⭐⭐⭐
```python
BCE = -(y × log(pred) + (1-y) × log(1-pred))

For: Binary classification (2 classes)

When to use:
  ✓ Always for binary classification
  ✓ Use with sigmoid activation
  ✓ Want probability outputs

Key insight:
  If y=1 and pred=0.9 → Loss ≈ 0.1 ✓ Good
  If y=1 and pred=0.5 → Loss ≈ 0.7 ✗ Bad
  If y=1 and pred=0.1 → Loss ≈ 2.3 ✗✗ Terrible
  
  Heavily penalizes wrong confident predictions!
```

#### 2. Categorical Cross-Entropy ⭐⭐⭐
```python
CCE = -Σ(true_one_hot × log(predictions))

For: Multi-class classification (3+ classes)

When to use:
  ✓ Always for multi-class problems
  ✓ Use with softmax activation
  ✓ When labels are one-hot encoded: [1,0,0]

Example:
  True:  [1, 0, 0] (class 0)
  Pred:  [0.7, 0.2, 0.1]
  Loss = -log(0.7) = 0.357
  
  Only the correct class probability matters!
```

#### 3. Sparse Categorical Cross-Entropy ⭐⭐
```python
Same as CCE but labels are integers: 0, 1, 2

When to use:
  ✓ Multi-class with integer labels
  ✓ Have many classes (saves memory)
  ✓ Don't want to one-hot encode

Difference:
  Regular:  y = [1, 0, 0]  (one-hot encoded)
  Sparse:   y = 0          (just the index)
  
  Result is identical, but sparse saves memory!
```

#### 4. Focal Loss ⭐⭐
```python
FL = -(1-pₜ)^γ × log(pₜ)

For: Imbalanced classification

When to use:
  ✓ Severe class imbalance (e.g., 99% vs 1%)
  ✓ Hard examples matter most
  ✓ Object detection, medical imaging
  ✓ When standard loss fails

How it works:
  • Easy examples (high confidence) get low weight
  • Hard examples (low confidence) get high weight
  • Model focuses on learning the hard cases
  
  Example:
  Easy correct (pₜ=0.95): weight = 0.0025 (almost ignore)
  Hard wrong (pₜ=0.05):   weight = 0.90   (heavily penalize)
```

#### 5. Hinge Loss ⭐
```python
HL = max(0, 1 - y·ŷ)

For: SVM and margin-based learning

When to use:
  ✓ Support Vector Machines
  ✓ Want margin-based decisions
  ✓ Don't need probability outputs

Note: y ∈ {-1, +1} not {0, 1}!
```

---

## The Real Examples from the Code

### Regression Example
```
Data: House prices
Actual:    [10, 20, 30, 40, 50]
Predicted: [12, 18, 32, 38, 52]
Errors:    [2, -2, 2, -2, 2]

MSE   = 4.0      (average squared error)
RMSE  = 2.0      (in same units as price)
MAE   = 2.0      (average absolute error)
Huber = 1.5      (more balanced)

With outlier (last actual = 500):
MSE   = 40,321   ← HUGE jump!
MAE   = 90.6     ← More stable
Huber = 90.1     ← Good balance
```

### Classification Example
```
Data: Binary classification
Actual:    [0, 1, 1, 0, 1]
Predicted: [0.1, 0.9, 0.8, 0.3, 0.95]

Sample 0: y=0, pred=0.1 → BCE = 0.105 ✓
Sample 1: y=1, pred=0.9 → BCE = 0.105 ✓
Sample 2: y=1, pred=0.8 → BCE = 0.223 ✓
Sample 3: y=0, pred=0.3 → BCE = 0.357 ✓
Sample 4: y=1, pred=0.95 → BCE = 0.051 ✓✓

Average BCE = 0.168

Lower is better!
```

---

## Visual Summary

### Regression Losses Shape

```
Loss
 ^
 |     MSE ╱╲
 |        ╱  ╲  (steep, sensitive to outliers)
 |       ╱    ╲
 |      ╱      ╲
 |─────┴────────┴───→ Error
 |
 |     MAE ╱╲╲
 |        ╱  ╲╲  (linear, robust)
 |       ╱    ╲╲
 |──────┴──────┴──→ Error
 |
 |    HUBER ╱╲╲
 |         ╱  ╲╲  (combined)
 |        ╱    ╲╲
 |───────┴──────┴──→ Error
```

### Binary Cross-Entropy Shape

```
Loss
 ^
 |        (when y=1)
 | ╲      Wrong answers penalized heavily
 |  ╲     
 |   ╲___  Correct answers rewarded
 |────────→ Predicted Probability
   0.5   1.0
```

---

## Python Quick Reference

### Scikit-learn (Metrics - for evaluation)
```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    log_loss
)

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
bce = log_loss(y_true, y_pred_proba)
```

### TensorFlow/Keras (For training)
```python
# Regression
model.compile(loss='mse')
model.compile(loss='mae')
model.compile(loss='huber')
model.compile(loss='log_cosh')

# Classification
model.compile(loss='binary_crossentropy')
model.compile(loss='categorical_crossentropy')
model.compile(loss='sparse_categorical_crossentropy')
model.compile(loss='binary_focal_crossentropy')
```

### NumPy (Manual implementation)
```python
# MSE
mse = np.mean((y_true - y_pred) ** 2)

# MAE
mae = np.mean(np.abs(y_true - y_pred))

# Binary Cross-Entropy
epsilon = 1e-7
y_pred = np.clip(y_pred, epsilon, 1-epsilon)
bce = -np.mean(y_true * np.log(y_pred) + 
               (1-y_true) * np.log(1-y_pred))
```

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Using MSE with Outliers
```
Bad:  MSE with data containing extreme values
      → One outlier will dominate the loss
      
Good: Use MAE or Huber Loss instead
      → Much more robust
```

### ❌ Mistake 2: Wrong Loss for Problem Type
```
Bad:  Using Binary Cross-Entropy for multi-class
      → Will produce errors
      
Good: Use Categorical Cross-Entropy for 3+ classes
      → Designed for multi-class
```

### ❌ Mistake 3: Not Scaling/Normalizing
```
Bad:  MSE on [price in $, temperature in K, ...]
      → Different scales dominate loss
      
Good: Normalize features first
      → Fair contribution from each feature
```

### ❌ Mistake 4: Forgetting Epsilon for Log
```
Bad:  loss = -np.log(y_pred)
      → If y_pred = 0, get log(0) = -∞
      
Good: epsilon = 1e-7
      y_pred = np.clip(y_pred, epsilon, 1-epsilon)
      → Prevents numerical errors
```

### ❌ Mistake 5: Wrong Activation Function
```
Bad:  Binary Cross-Entropy with ReLU activation
      → Won't produce probabilities (0-1)
      
Good: Binary Cross-Entropy with Sigmoid
      → Produces valid probabilities
      
Good: Categorical Cross-Entropy with Softmax
      → Produces valid probability distribution
```

---

## Activation Function Pairing

| Loss Function | Activation | Output Range | Use Case |
|---------------|-----------|--------------|----------|
| MSE/MAE | Linear | (-∞, ∞) | Regression |
| Binary BCE | Sigmoid | (0, 1) | Binary classification |
| Categorical CCE | Softmax | (0, 1), Σ=1 | Multi-class classification |
| Hinge | Linear | (-∞, ∞) | SVM |

---

## Training Workflow

```
1. Choose problem type
   ├─ Regression → Use MSE/MAE/Huber
   └─ Classification → Use Cross-Entropy

2. Check for issues
   ├─ Outliers? → Use MAE/Huber
   ├─ Imbalanced? → Use Focal Loss
   └─ Normal? → Use standard loss

3. Set up training
   ├─ Choose activation (sigmoid/softmax/linear)
   ├─ Choose optimizer (Adam/SGD)
   └─ Specify loss function

4. Monitor training
   ├─ Watch loss decrease
   ├─ Check for overfitting
   └─ Validate on test set

5. Evaluate
   ├─ Use appropriate metric
   ├─ Report results clearly
   └─ Compare with baseline
```

---

## When Each Loss Wins

| Scenario | Best Loss | Why |
|----------|-----------|-----|
| Standard regression | MSE | Simple, standard |
| Regression + outliers | MAE | Robust |
| Regression + deep learning | Log-Cosh | Smooth gradients |
| Binary classification | Binary CE | Standard |
| Multi-class classification | Categorical CE | Standard |
| Imbalanced classification | Focal Loss | Handles imbalance |
| SVM | Hinge Loss | Designed for SVM |
| Regression + safety | Huber | Balanced approach |
| Uncertainty quantification | Quantile | Intervals |
| Distribution learning | KL Divergence | Matches distributions |

---

## Key Takeaways

### The Big Picture
1. **Loss guides learning** - Model improves by minimizing it
2. **Different losses suit different problems** - No one-size-fits-all
3. **Choice affects training dynamics** - Impacts convergence and quality
4. **Always validate choice** - Test performance matters most

### Most Used
- **Regression**: MSE (if no outliers) or MAE (if outliers)
- **Binary Classification**: Binary Cross-Entropy
- **Multi-class Classification**: Categorical Cross-Entropy
- **Imbalanced Data**: Focal Loss

### Pro Tips
1. Start simple → upgrade if needed
2. Monitor training loss closely
3. Validate with appropriate metrics
4. Consider your data characteristics
5. Test different losses if results are poor

---

## Files Provided

1. **cost_functions_guide.md** - Complete detailed guide with all formulas
2. **cost_functions_formulas.md** - Mathematical deep dive with examples
3. **cost_functions_cheatsheet.md** - Quick reference for quick lookups
4. **cost_functions_examples.py** - Working Python code (executable)
5. **cost_functions_plots.png** - Visualizations of different losses

---

## Next Steps

1. Read the detailed guide to understand concepts
2. Review the formulas document for mathematical details
3. Check the cheatsheet when you need quick reference
4. Run the Python code to see examples in action
5. Study the visualizations to understand loss behavior
6. Choose appropriate loss for your problem
7. Implement and monitor during training

Good luck with your machine learning projects! 🚀

