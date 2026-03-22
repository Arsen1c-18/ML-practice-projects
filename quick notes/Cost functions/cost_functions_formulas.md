# Cost Functions: Complete Formula Reference

## Overview

A cost function (or loss function) quantifies the prediction error of a model. The goal of training is to **minimize** the cost function.

```
Cost Function = F(predictions, actual)
                ↓
        Lower cost = Better predictions
```

---

## REGRESSION COST FUNCTIONS

### 1. Mean Squared Error (MSE)

```
Formula:  J(θ) = (1/m) × Σ(yᵢ - ŷᵢ)²
          
          m = number of samples
          yᵢ = actual value
          ŷᵢ = predicted value

Also called: L2 Loss, Quadratic Loss, Squared Error

Example:
  Actual:     [10, 20, 30]
  Predicted:  [12, 18, 32]
  Errors:     [2, -2, 2]
  Squared:    [4, 4, 4]
  MSE = (4 + 4 + 4) / 3 = 4.0

Key Characteristics:
  • Penalizes large errors heavily (quadratic)
  • Sensitive to outliers
  • Always positive: J ≥ 0
  • Differentiable: good for gradient descent
  • J = 0 only when perfect predictions

Python:
  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(y_true, y_pred)
  
  # Or manually
  mse = np.mean((y_true - y_pred) ** 2)
```

---

### 2. Root Mean Squared Error (RMSE)

```
Formula:  J(θ) = √(MSE) = √((1/m) × Σ(yᵢ - ŷᵢ)²)

Also called: L2 norm, Standard deviation of residuals

Relationship:
  RMSE = √MSE
  
  If MSE = 4
  Then RMSE = √4 = 2

Why use RMSE instead of MSE?
  ✓ Same units as target variable (interpretable)
  ✓ "On average, predictions are off by RMSE units"
  ✓ More intuitive for communication

Example:
  MSE = 4.0 (what does "4.0" mean in dollars²?)
  RMSE = 2.0 (average error is $2, much clearer!)

Python:
  from sklearn.metrics import mean_squared_error
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

---

### 3. Mean Absolute Error (MAE)

```
Formula:  J(θ) = (1/m) × Σ|yᵢ - ŷᵢ|
          
          |...| = absolute value

Also called: L1 Loss, Mean Absolute Deviation

Example:
  Actual:     [10, 20, 30, 1000]
  Predicted:  [12, 18, 32, 15]
  Errors:     [2, -2, 2, 985]
  Abs Errors: [2, 2, 2, 985]
  MAE = (2 + 2 + 2 + 985) / 4 = 247.75

Comparison with MSE on outlier:
  MSE = (4 + 4 + 4 + 970225) / 4 = 242,559.5 ← HUGE!
  MAE = 247.75 ← Much more reasonable

Key Characteristics:
  • Linear penalty for all errors
  • Robust to outliers
  • Same units as target
  • Not differentiable at zero (minor optimization issue)
  • Represents median of errors

Python:
  from sklearn.metrics import mean_absolute_error
  mae = mean_absolute_error(y_true, y_pred)
  
  # Or manually
  mae = np.mean(np.abs(y_true - y_pred))
```

---

### 4. Huber Loss

```
Formula:  L(y, ŷ) = {
            0.5(y - ŷ)²              if |y - ŷ| ≤ δ
            δ(|y - ŷ| - 0.5δ)        if |y - ŷ| > δ
          }
          
          δ = threshold parameter (typically 1.0)

What it does:
  • For small errors: acts like MSE (smooth)
  • For large errors: acts like MAE (robust)

Visual interpretation:
  Small error (|e| ≤ 1): quadratic penalty
  Large error (|e| > 1): linear penalty

Example with δ = 1:
  Error = 0.5  → Loss = 0.5 × 0.5² = 0.125 (MSE-like)
  Error = 1.0  → Loss = 0.5 × 1.0² = 0.5 (at boundary)
  Error = 2.0  → Loss = 1.0×(2.0-0.5) = 1.5 (MAE-like)
  
  Compare:
    MSE:  2² = 4.0 (penalizes heavily)
    Huber: 1.5 (moderate penalty)
    MAE:  2.0 (linear penalty)

Key Characteristics:
  ✓ Robust to outliers
  ✓ Smooth and differentiable
  ✓ Combines MSE and MAE benefits
  ✗ Requires tuning δ parameter

Python:
  from tensorflow.keras.losses import Huber
  loss = Huber(delta=1.0)
  
  # Or manually
  errors = y_true - y_pred
  is_small = np.abs(errors) <= delta
  loss = np.where(
      is_small,
      0.5 * errors ** 2,
      delta * (np.abs(errors) - 0.5 * delta)
  ).mean()
```

---

### 5. Log-Cosh Loss

```
Formula:  L(y, ŷ) = log(cosh(y - ŷ))
          
          cosh(x) = (eˣ + e⁻ˣ) / 2

What log(cosh) does:
  • Smooth and twice differentiable
  • Approximately MSE for small errors
  • Approximately MAE for large errors
  • Better than MSE for outliers

Mathematical properties:
  cosh(x) = 1 + x²/2! + x⁴/4! + ...  (Taylor series)
  
  For small x: log(cosh(x)) ≈ x²/2 (like MSE)
  For large x: log(cosh(x)) ≈ |x| (like MAE)

Example:
  Error = 0.1  → log(cosh(0.1)) ≈ 0.005 (MSE-like)
  Error = 1.0  → log(cosh(1.0)) ≈ 0.433
  Error = 2.0  → log(cosh(2.0)) ≈ 1.998
  Error = 10.0 → log(cosh(10)) ≈ 9.999 (MAE-like)

Key Characteristics:
  ✓ Smooth optimization
  ✓ Robust to outliers
  ✓ Good numerical properties
  ✓ Popular in deep learning
  • Slightly more computation

Python:
  from tensorflow.keras.losses import LogCosh
  loss = LogCosh()
  
  # Or manually
  loss = np.log(np.cosh(y_true - y_pred)).mean()
```

---

### 6. Quantile Loss

```
Formula:  L_q(y, ŷ) = {
            q(y - ŷ)        if y ≥ ŷ
            (1-q)(ŷ - y)    if y < ŷ
          }
          
          q = quantile (0 < q < 1)

Special cases:
  q = 0.5: Median loss (regular MAE-like)
  q = 0.25: 25th percentile (lower bound)
  q = 0.75: 75th percentile (upper bound)

Example with q = 0.5:
  If y > ŷ: Loss = 0.5(y - ŷ)      (penalty for underprediction)
  If y < ŷ: Loss = 0.5(ŷ - y)      (penalty for overprediction)
  
  Symmetric! Both under/overprediction penalized equally.

Example with q = 0.75 (75th percentile):
  If y > ŷ: Loss = 0.75(y - ŷ)     (heavier penalty for under)
  If y < ŷ: Loss = 0.25(ŷ - y)     (lighter penalty for over)
  
  Asymmetric! Underestimation penalized more.

Use Cases:
  • q = 0.5: Standard prediction (median)
  • q = 0.1, 0.5, 0.9: Prediction intervals
  • q = 0.95: Conservative predictions (avoid underestimation)
  • q = 0.05: Aggressive predictions (avoid overestimation)

Key Characteristics:
  ✓ Estimates quantiles, not just mean
  ✓ Asymmetric penalties possible
  ✓ Robust to outliers
  ✓ Good for uncertainty estimation
  • Multiple models needed for intervals

Python:
  def quantile_loss(y_true, y_pred, q=0.5):
      error = y_true - y_pred
      return np.mean(
          np.where(error >= 0, q*error, (q-1)*error)
      )
```

---

## CLASSIFICATION COST FUNCTIONS

### 1. Binary Cross-Entropy (Log Loss)

```
Formula:  J = -(1/m) × Σ[yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]
          
          y ∈ {0, 1} = true class label
          ŷ ∈ (0, 1) = predicted probability

Also called: Log Loss, Sigmoid Cross-Entropy

Breaking it down:
  When y = 1: Loss = -log(ŷ)
    If ŷ = 0.99: Loss ≈ 0.010  ✓ Good
    If ŷ = 0.50: Loss ≈ 0.693  ✗ Bad
    If ŷ = 0.01: Loss ≈ 4.605  ✗✗ Terrible
    
  When y = 0: Loss = -log(1-ŷ)
    If ŷ = 0.01: Loss ≈ 0.010  ✓ Good
    If ŷ = 0.50: Loss ≈ 0.693  ✗ Bad
    If ŷ = 0.99: Loss ≈ 4.605  ✗✗ Terrible

Key insight:
  Heavily penalizes wrong confident predictions!
  Getting the correct answer with low confidence is bad.

Example:
  True labels: [0, 1, 1, 0, 1]
  Predictions: [0.1, 0.9, 0.8, 0.3, 0.95]
  
  Sample 0: y=0, ŷ=0.1 → -log(0.9) = 0.105 ✓
  Sample 1: y=1, ŷ=0.9 → -log(0.9) = 0.105 ✓
  Sample 2: y=1, ŷ=0.8 → -log(0.8) = 0.223 ✓
  Sample 3: y=0, ŷ=0.3 → -log(0.7) = 0.357 ✓
  Sample 4: y=1, ŷ=0.95 → -log(0.95) = 0.051 ✓✓
  
  Average: 0.168

Key Characteristics:
  ✓ Standard for binary classification
  ✓ Works with sigmoid activation
  ✓ Differentiable everywhere
  ✓ Rewards confident correct predictions
  ✓ Punishes confident wrong predictions
  • Can be unstable with probabilities very close to 0 or 1

Python:
  from sklearn.metrics import log_loss
  bce = log_loss(y_true, y_pred_proba)
  
  from tensorflow.keras.losses import BinaryCrossentropy
  loss = BinaryCrossentropy()
  
  # Or manually (with numerical stability)
  epsilon = 1e-7
  y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
  bce = -np.mean(
      y_true * np.log(y_pred) + 
      (1 - y_true) * np.log(1 - y_pred)
  )

Activation function: Sigmoid σ(z) = 1 / (1 + e^(-z))
```

---

### 2. Categorical Cross-Entropy

```
Formula:  J = -(1/m) × Σ Σ yᵢⱼ·log(ŷᵢⱼ)
          
          i = sample index
          j = class index
          y[i,j] = 1 if sample i belongs to class j, else 0
          ŷ[i,j] = predicted probability of class j for sample i

Also called: Softmax Cross-Entropy, Multi-class Log Loss

Key insight: Only the correct class contributes to loss!

Example (3 classes: Cat, Dog, Bird):
  True:      [1, 0, 0]  (Cat)
  Predicted: [0.7, 0.2, 0.1]
  
  Loss = -(1×log(0.7) + 0×log(0.2) + 0×log(0.1))
       = -log(0.7)
       = 0.357

  Another sample:
  True:      [0, 1, 0]  (Dog)
  Predicted: [0.3, 0.4, 0.3]
  
  Loss = -log(0.4) = 0.916

Comparison:
  Good prediction [0.7, 0.2, 0.1]: Loss = 0.357 ✓
  Bad prediction  [0.3, 0.4, 0.3]: Loss = 0.916 ✗

Key Characteristics:
  ✓ Standard for multi-class problems
  ✓ Works with softmax activation
  ✓ Generalizes binary cross-entropy
  ✓ Only considers correct class
  • Need one-hot encoded labels

Python:
  from tensorflow.keras.losses import CategoricalCrossentropy
  loss = CategoricalCrossentropy()
  
  # Or manually
  epsilon = 1e-7
  y_pred = np.clip(y_pred, epsilon, 1)
  cce = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

Activation function: Softmax σ(zⱼ) = e^(zⱼ) / Σ e^(zₖ)
```

---

### 3. Sparse Categorical Cross-Entropy

```
Formula:  Same as Categorical Cross-Entropy
          
          BUT y is a single integer (class index)
          NOT one-hot encoded

Difference from Categorical Cross-Entropy:
  Categorical:      y = [1, 0, 0] (one-hot)
  Sparse Categorical: y = 0 (class index)

Example:
  True class: 0 (just the index)
  Predicted: [0.7, 0.2, 0.1]
  
  Loss = -log(0.7) = 0.357 (same result!)

Memory efficiency:
  With 1000 classes:
    Categorical: 1000 values per sample
    Sparse:      1 integer per sample
    
    Savings: 1000× less memory for labels!

Python:
  from tensorflow.keras.losses import SparseCategoricalCrossentropy
  loss = SparseCategoricalCrossentropy()
  
  # Used when labels are: [0, 1, 2, 0, 1]
  # Not: [[1,0,0], [0,1,0], [0,0,1], ...]

When to use:
  ✓ Labels already as integers
  ✓ Very many classes
  ✓ Memory is limited
  ✗ Not using one-hot encoding
```

---

### 4. Focal Loss

```
Formula:  FL(pₜ) = -α(1 - pₜ)^γ × log(pₜ)
          
          pₜ = predicted probability of true class
          α = weighting factor (default 0.25)
          γ = focusing parameter (default 2)

Also called: Focal Loss for Dense Object Detection

Why it exists:
  Standard cross-entropy treats all examples equally.
  Hard examples (class imbalance) can be lost.

The focusing parameter γ:
  γ = 0: No modification (standard cross-entropy)
  γ = 1: Moderate focusing
  γ = 2: Strong focusing (recommended)
  γ = 5: Very strong focusing

How it works:
  (1 - pₜ)^γ is the "focusing weight"
  
  If pₜ = 0.95 (easy, correct prediction):
    weight = (1 - 0.95)^2 = 0.0025 (nearly zero!)
    Loss ≈ 0.0025 × 0.051 ≈ 0.00013 (ignored)
    
  If pₜ = 0.50 (hard, uncertain prediction):
    weight = (1 - 0.50)^2 = 0.25 (moderate)
    Loss = 0.25 × 0.693 ≈ 0.173 (focused)
    
  If pₜ = 0.05 (very hard, wrong prediction):
    weight = (1 - 0.05)^2 = 0.9025 (large!)
    Loss = 0.9025 × 3.0 ≈ 2.71 (heavily penalized)

Key insight:
  Easy examples are down-weighted
  Hard examples are up-weighted
  Model focuses on learning hard cases

Use cases:
  ✓ Class imbalance (e.g., 99% background, 1% object)
  ✓ Object detection
  ✓ Medical imaging (rare diseases)
  ✓ When standard loss fails

Python:
  from tensorflow.keras.losses import BinaryFocalCrossentropy
  loss = BinaryFocalCrossentropy(alpha=0.25, gamma=2.0)
  
  # Or manually
  alpha = 0.25
  gamma = 2.0
  epsilon = 1e-7
  y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
  ce = -(y_true * np.log(y_pred) + 
         (1-y_true) * np.log(1-y_pred))
  pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
  focal_weight = alpha * (1 - pt) ** gamma
  focal_loss = focal_weight * ce
  return focal_loss.mean()
```

---

### 5. Hinge Loss

```
Formula:  L(y, ŷ) = max(0, 1 - y·ŷ)
          
          y ∈ {-1, +1} (not 0, 1!)
          ŷ = predicted score (not probability)

Also called: SVM Loss, Margin Loss

How it works:
  The "margin" is 1.
  
  If y·ŷ > 1: Correct prediction with confidence
    Loss = max(0, 1 - y·ŷ) = 0 ✓
    
  If y·ŷ = 1: Correct prediction at margin
    Loss = 0 (barely acceptable)
    
  If y·ŷ = 0: On the decision boundary
    Loss = max(0, 1) = 1
    
  If y·ŷ < 0: Wrong side of boundary
    Loss = max(0, 1 - y·ŷ) = 1 + |y·ŷ|

Visual interpretation:
```
        Loss
         |
       3 |     /
         |    /
       2 |   /
         |  /
       1 | /
         |/_______ y·ŷ
       0 |_1_2_3
         |
```

Example:
  Sample 1: y = +1 (positive class)
    ŷ = 0.8  → y·ŷ = 0.8 → Loss = max(0, 1-0.8) = 0.2
    ŷ = 1.5  → y·ŷ = 1.5 → Loss = max(0, 1-1.5) = 0
    ŷ = -0.5 → y·ŷ =-0.5 → Loss = max(0, 1-(-0.5)) = 1.5
    
  Sample 2: y = -1 (negative class)
    ŷ = -0.8 → y·ŷ = 0.8 → Loss = 0.2
    ŷ = -1.5 → y·ŷ = 1.5 → Loss = 0
    ŷ = 0.5  → y·ŷ =-0.5 → Loss = 1.5

Key Characteristics:
  ✓ Margin-based thinking (useful for understanding)
  ✓ Standard for SVM
  ✓ Sparse solutions possible
  ✗ Not differentiable at y·ŷ = 1
  ✗ Doesn't output probabilities
  
Python:
  from sklearn.metrics import hinge_loss
  hl = hinge_loss(y_true, y_pred_scores)
  
  # Or manually
  # Convert y from {0, 1} to {-1, +1} if needed
  y_svm = 2 * y_true - 1
  hl = np.mean(np.maximum(0, 1 - y_svm * y_pred))
```

---

## COMPARISON SUMMARY

### When to Use Each Loss

| Loss | Best For | Output | Formula Complexity |
|------|----------|--------|-------------------|
| MSE | Standard regression | Continuous | Simple |
| MAE | Robust regression | Continuous | Simple |
| Huber | Safe regression | Continuous | Medium |
| BCE | Binary classification | Probability | Medium |
| CCE | Multi-class classification | Probabilities | Medium |
| Focal Loss | Imbalanced classification | Probability | Complex |
| Hinge Loss | SVM | Score | Simple |

### Sensitivity to Outliers

```
Most Sensitive:  MSE (quadratic)
                 ↓
             Log-Cosh (smooth)
                 ↓
             Huber (balanced)
                 ↓
Least Sensitive: MAE (linear)
```

### Differentiability

```
Fully Differentiable:  MSE, MAE, Log-Cosh, BCE, CCE
Non-differentiable:    Hinge Loss (at margin)
```

---

## Implementation Checklist

- [ ] Understand the problem type (regression/classification)
- [ ] Choose appropriate loss function
- [ ] Handle numerical stability (epsilon for log operations)
- [ ] Verify output format (probability, score, etc.)
- [ ] Consider class imbalance if present
- [ ] Check for outliers
- [ ] Use appropriate activation function with loss
- [ ] Validate on test set
- [ ] Monitor loss during training

---

## Key Takeaways

1. **Loss function guides the learning process** - The model learns to minimize it

2. **Different losses suit different problems**:
   - Regression: MSE/RMSE, MAE, Huber, Log-Cosh
   - Classification: Cross-Entropy, Focal Loss, Hinge

3. **MSE is sensitive to outliers** - Consider MAE or Huber Loss if needed

4. **Cross-Entropy heavily penalizes wrong confident predictions** - Good for classification

5. **Focal Loss handles class imbalance** - Down-weights easy examples

6. **Always match loss with activation function**:
   - Sigmoid + Binary Cross-Entropy (binary)
   - Softmax + Categorical Cross-Entropy (multi-class)
   - Linear + MSE/MAE (regression)

