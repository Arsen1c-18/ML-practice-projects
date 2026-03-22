# Cost Functions Quick Reference Cheat Sheet

## At a Glance

| Cost Function | Type | When to Use | Formula |
|---------------|------|-------------|---------|
| **MSE** | Regression | Standard, well-behaved data | (1/m)Σ(y-ŷ)² |
| **RMSE** | Regression | Need interpretability | √MSE |
| **MAE** | Regression | Outliers present | (1/m)Σ\|y-ŷ\| |
| **Huber Loss** | Regression | Robust + smooth | MSE for small, MAE for large |
| **Log-Cosh** | Regression | Deep learning | log(cosh(y-ŷ)) |
| **Binary Cross-Entropy** | Classification | Binary problems | -[y·log(ŷ) + (1-y)·log(1-ŷ)] |
| **Categorical Cross-Entropy** | Classification | Multi-class | -Σy·log(ŷ) |
| **Focal Loss** | Classification | Imbalanced data | -(1-pₜ)^γ·log(pₜ) |
| **Hinge Loss** | Classification | SVM | max(0, 1-y·ŷ) |

---

## Decision Tree

```
START
  ↓
What problem?
  ├─ REGRESSION
  │   ├─ Outliers? 
  │   │   ├─ NO → Use MSE/RMSE
  │   │   └─ YES → Use MAE or Huber Loss
  │   └─ Deep Learning? 
  │       ├─ YES → Use Log-Cosh
  │       └─ NO → Use MSE or MAE
  │
  └─ CLASSIFICATION
      ├─ Binary?
      │   ├─ YES → Use Binary Cross-Entropy
      │   │   └─ Imbalanced? → Use Focal Loss
      │   └─ NO → Use Categorical Cross-Entropy
      │       └─ Imbalanced? → Use Focal Loss
      │
      └─ SVM?
          ├─ YES → Use Hinge Loss
          └─ NO → See above
```

---

## Regression Cost Functions

### Mean Squared Error (MSE)
```python
MSE = (1/m) × Σ(yᵢ - ŷᵢ)²

Pros:
  ✓ Standard choice
  ✓ Smooth, differentiable
  ✓ Penalizes large errors
  
Cons:
  ✗ Sensitive to outliers
  ✗ Units are squared (not intuitive)
  
Use when: Data is clean, no extreme outliers
```

### Root Mean Squared Error (RMSE)
```python
RMSE = √MSE

Pros:
  ✓ Same units as target (interpretable)
  ✓ Penalizes large errors
  
Cons:
  ✗ Sensitive to outliers
  
Use when: You need to communicate results to non-technical people
```

### Mean Absolute Error (MAE)
```python
MAE = (1/m) × Σ|yᵢ - ŷᵢ|

Pros:
  ✓ Robust to outliers
  ✓ Linear penalty (simple)
  ✓ Same units as target
  
Cons:
  ✗ Not differentiable at 0
  ✗ Harder to optimize
  
Use when: Data has outliers or extreme values
```

### Huber Loss
```python
L = { 0.5(y-ŷ)²        if |y-ŷ| ≤ δ
    { δ(|y-ŷ|-0.5δ)    if |y-ŷ| > δ

δ = threshold (default 1.0)

Pros:
  ✓ Combines MSE and MAE benefits
  ✓ Smooth and differentiable
  ✓ Robust to outliers
  
Cons:
  ✗ Requires tuning δ parameter
  
Use when: You want a balanced, robust approach
```

### Log-Cosh Loss
```python
L = log(cosh(y - ŷ))

Pros:
  ✓ Twice differentiable (good for optimization)
  ✓ Smooth function
  ✓ Robust to outliers
  ✓ Works well for deep learning
  
Cons:
  ✗ Slightly slower to compute
  
Use when: Building deep neural networks
```

---

## Classification Cost Functions

### Binary Cross-Entropy (Log Loss)
```python
BCE = -(1/m) × Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]

y = 0 or 1 (true class)
ŷ = probability in [0, 1]

Pros:
  ✓ Standard for binary classification
  ✓ Works with sigmoid activation
  ✓ Differentiable everywhere
  
Cons:
  ✗ Can be unstable with extreme predictions
  
Use when: Binary classification problem
```

**Behavior:**
- If y=1 and ŷ=0.9 → Loss = 0.105 (good)
- If y=1 and ŷ=0.5 → Loss = 0.693 (bad)
- If y=1 and ŷ=0.1 → Loss = 2.303 (terrible)

### Categorical Cross-Entropy
```python
CCE = -(1/m) × Σ Σ y[i,j]·log(ŷ[i,j])

y[i,j] = 1 if sample i is class j, else 0
ŷ[i,j] = probability of class j for sample i

Pros:
  ✓ Standard for multi-class problems
  ✓ Works with softmax activation
  
Cons:
  ✗ Only correct class affects loss
  
Use when: Multi-class classification (3+ classes)
```

### Sparse Categorical Cross-Entropy
```python
Same as Categorical Cross-Entropy, but:
- y is a single integer (class index)
- Not one-hot encoded

Pros:
  ✓ Saves memory with many classes
  ✓ Cleaner code
  
Use when: Labels are integers, not one-hot encoded
```

### Focal Loss (RetinaNet Loss)
```python
FL = -α(1 - pₜ)^γ·log(pₜ)

pₜ = confidence of correct class
α = weighting factor (0-1, default 0.25)
γ = focusing parameter (default 2)

Pros:
  ✓ Handles class imbalance
  ✓ Focuses on hard examples
  ✓ Down-weights easy examples
  
Cons:
  ✗ Requires tuning α and γ
  ✗ More complex
  
Use when: Highly imbalanced dataset (e.g., 99% negative)
```

**Gamma Effect:**
- γ=0: No focusing (standard cross-entropy)
- γ=1: Moderate focusing
- γ=2: Strong focusing (recommended)
- γ=5: Very strong focusing

### Hinge Loss
```python
HL = max(0, 1 - y·ŷ)

y = -1 or +1 (class labels)
ŷ = prediction score

Pros:
  ✓ Standard for SVM
  ✓ Margin-based thinking
  ✓ Sparse solutions
  
Cons:
  ✗ Not differentiable at margin
  
Use when: Using Support Vector Machines
```

---

## Outlier Robustness Comparison

When you have outliers in your data:

```
MOST SENSITIVE (avoid)
    ↑
    │ MSE ──────────────────── Squares errors
    │ Squared Hinge Loss ────── Heavy quadratic penalty
    │
    │ Log-Cosh ─────────────── Moderate robustness
    │
    │ Huber Loss ────────────── Good robustness
    │
    ↓ MAE ────────────────────── Linear penalty (most robust)
    │
LEAST SENSITIVE (use)
```

---

## Deep Learning Recommendations

### For Neural Networks:

**Regression:**
```python
# Clean data
model.compile(optimizer='adam', loss='mse')

# Data with outliers
model.compile(optimizer='adam', loss='mae')

# Best overall (smooth)
model.compile(optimizer='adam', loss='log_cosh')
```

**Classification (Binary):**
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# If imbalanced
model.compile(
    optimizer='adam',
    loss='binary_focal_crossentropy',
    metrics=['accuracy']
)
```

**Classification (Multi-class):**
```python
# If one-hot encoded
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# If integer labels
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# If imbalanced
model.compile(
    optimizer='adam',
    loss='sparse_categorical_focal_crossentropy',
    metrics=['accuracy']
)
```

---

## Scikit-learn Metrics

```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    log_loss,
    hinge_loss,
    zero_one_loss
)

# Regression
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)

# Classification
bce = log_loss(y_true, y_pred_proba)
hl = hinge_loss(y_true, y_pred_scores)
```

---

## TensorFlow/Keras Reference

```python
import tensorflow as tf

# Regression losses
tf.keras.losses.MeanSquaredError()
tf.keras.losses.MeanAbsoluteError()
tf.keras.losses.Huber(delta=1.0)
tf.keras.losses.LogCosh()
tf.keras.losses.MeanSquaredLogarithmicError()

# Classification losses (Binary)
tf.keras.losses.BinaryCrossentropy()
tf.keras.losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2.0)
tf.keras.losses.Hinge()
tf.keras.losses.SquaredHinge()

# Classification losses (Multi-class)
tf.keras.losses.CategoricalCrossentropy()
tf.keras.losses.SparseCategoricalCrossentropy()

# Advanced
tf.keras.losses.KLDivergence()  # Distribution matching
tf.keras.losses.Poisson()        # Count data
```

---

## NumPy Implementation Templates

```python
import numpy as np

# MSE
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# MAE
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Huber
def huber(y_true, y_pred, delta=1.0):
    errors = y_true - y_pred
    is_small = np.abs(errors) <= delta
    return np.mean(
        np.where(is_small, 0.5*errors**2, 
                 delta*(np.abs(errors) - 0.5*delta))
    )

# Binary Cross-Entropy
def bce(y_true, y_pred):
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1-eps)
    return -np.mean(
        y_true*np.log(y_pred) + 
        (1-y_true)*np.log(1-y_pred)
    )

# Categorical Cross-Entropy
def cce(y_true, y_pred):
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1)
    return -np.mean(np.sum(y_true*np.log(y_pred), axis=1))
```

---

## Common Mistakes to Avoid

❌ Using MSE when you have severe outliers
- Solution: Use MAE or Huber Loss

❌ Using Binary Cross-Entropy for multi-class
- Solution: Use Categorical Cross-Entropy

❌ Not normalizing when using MSE
- Solution: Normalize or standardize inputs

❌ Using the same loss for training and evaluation
- Solution: Use appropriate metric for evaluation

❌ Not considering class imbalance
- Solution: Use Focal Loss or weighted loss

❌ Forgetting to clip probabilities (→ log(0) error)
- Solution: Add small epsilon (1e-7) to avoid numerical issues

---

## Quick Comparison Example

```python
y_actual = [10, 20, 30, 40, 50]
y_pred =   [12, 18, 32, 38, 52]
errors =   [2, -2, 2, -2, 2]

MSE  = mean([4, 4, 4, 4, 4]) = 4.0
RMSE = √4.0 = 2.0
MAE  = mean([2, 2, 2, 2, 2]) = 2.0

Huber(δ=1) = mean([1.0, 1.0, 1.0, 1.0, 1.0]) = 1.0
Log-Cosh   ≈ 1.32

Same data:
MSE = 4.0 (large errors)
MAE = 2.0 (moderate)
Huber = 1.0 (balanced)
```

---

## Summary: When to Choose Each

| Situation | Best Loss Function | Why |
|-----------|-------------------|-----|
| Regression, clean data | MSE | Standard, simple |
| Regression, with outliers | MAE or Huber | Robust |
| Deep Learning Regression | Log-Cosh | Smooth gradients |
| Binary Classification | Binary Cross-Entropy | Standard, works with sigmoid |
| Multi-class Classification | Categorical Cross-Entropy | Standard, works with softmax |
| Imbalanced Classification | Focal Loss | Handles imbalance |
| SVM Classification | Hinge Loss | Margin-based |
| Metric Learning | Triplet Loss | Similarity learning |
| GAN Training | Wasserstein Loss | Better gradients |
| Uncertainty Quantification | Quantile Loss | Prediction intervals |

