# Cost Functions in Machine Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Regression Cost Functions](#regression-cost-functions)
3. [Classification Cost Functions](#classification-cost-functions)
4. [Advanced Cost Functions](#advanced-cost-functions)
5. [Comparison and When to Use](#comparison-and-when-to-use)
6. [Implementation Examples](#implementation-examples)

---

## Introduction

### What is a Cost Function?

A **cost function** (also called loss function or objective function) measures how well a machine learning model performs. It quantifies the difference between predicted values and actual values.

**Core Idea:**
```
Cost = How wrong the model is
       (Lower is better)
```

### Why Cost Functions Matter

1. **Optimization** - Guide the learning algorithm to improve predictions
2. **Evaluation** - Measure model performance
3. **Comparison** - Compare different models
4. **Regularization** - Prevent overfitting
5. **Hyperparameter Tuning** - Select best parameters

### General Form

```
J(θ) = (1/m) Σ L(ŷᵢ, yᵢ)

Where:
  J(θ) = Total cost
  m = Number of samples
  L = Loss for individual sample
  ŷ = Predicted value
  y = Actual value
  θ = Model parameters
```

---

## Regression Cost Functions

Used when predicting continuous values (prices, temperatures, distances, etc.)

### 1. Mean Squared Error (MSE)

**Formula:**
```
MSE = (1/m) Σ(yᵢ - ŷᵢ)²

Where:
  yᵢ = Actual value
  ŷᵢ = Predicted value
  m = Number of samples
```

**Properties:**
- Squares the error → Penalizes large errors heavily
- Differentiable everywhere (good for optimization)
- All errors treated equally in magnitude
- Sensitive to outliers
- Same units as y² (not intuitive)

**When to Use:**
- Standard choice for regression
- When you want to penalize large errors more
- When outliers should matter

**Example:**
```
Actual:    [10, 20, 30]
Predicted: [12, 18, 32]
Errors:    [2,  -2, 2]
Squared:   [4,   4, 4]
MSE = (4 + 4 + 4) / 3 = 4.0
```

---

### 2. Root Mean Squared Error (RMSE)

**Formula:**
```
RMSE = √(MSE) = √((1/m) Σ(yᵢ - ŷᵢ)²)
```

**Properties:**
- Same units as target variable (more interpretable)
- √ of MSE to bring error back to original scale
- Still penalizes large errors
- Most commonly used evaluation metric

**When to Use:**
- When you want interpretability (same units as y)
- When communicating results to stakeholders
- Preferred for reporting final performance

**Example:**
```
If MSE = 4.0 and target is in dollars
RMSE = √4 = $2.0 (average prediction error)
More interpretable than "4.0"
```

---

### 3. Mean Absolute Error (MAE)

**Formula:**
```
MAE = (1/m) Σ |yᵢ - ŷᵢ|

Where |...| means absolute value
```

**Properties:**
- Uses absolute value instead of squaring
- Same units as target variable (interpretable)
- Treats all errors equally (linear penalty)
- Robust to outliers
- Not differentiable at zero (optimization issues)

**When to Use:**
- When you want robustness to outliers
- When all errors are equally important
- When simplicity matters

**Example:**
```
Actual:    [10, 20, 30, 1000]
Predicted: [12, 18, 32, 15]
Errors:    [2,  2,  2,  985]

MSE = (4 + 4 + 4 + 970225) / 4 = 242559.5
      (outlier heavily penalized)

MAE = (2 + 2 + 2 + 985) / 4 = 247.75
      (outlier less influential)
```

---

### 4. Huber Loss

**Formula:**
```
L(y, ŷ) = {
    0.5(y - ŷ)²           if |y - ŷ| ≤ δ
    δ(|y - ŷ| - 0.5δ)     if |y - ŷ| > δ
}

δ = Tunable threshold (typically 1.0)
```

**Properties:**
- Combines MSE (for small errors) and MAE (for large errors)
- Robust to outliers
- Smooth (differentiable everywhere)
- Requires tuning δ parameter

**When to Use:**
- When you have outliers but want smooth optimization
- Balance between MSE and MAE
- Professional/robust models

---

### 5. Log-Cosh Loss

**Formula:**
```
L(y, ŷ) = log(cosh(ŷ - y))

Where cosh(x) = (eˣ + e⁻ˣ) / 2
```

**Properties:**
- Smooth and differentiable
- Behaves like MSE for small errors
- Behaves like MAE for large errors
- Twice differentiable (good for optimization)
- Less sensitive to outliers than MSE

**When to Use:**
- When you need smooth optimization
- Deep learning applications
- Outier-robust gradient-based learning

---

### 6. Quantile Loss

**Formula:**
```
L(y, ŷ) = {
    q(y - ŷ)        if y ≥ ŷ
    (1-q)(ŷ - y)    if y < ŷ
}

q = Quantile (0.5 for median, 0.25 for 25th percentile)
```

**Properties:**
- Estimates specific quantiles, not just mean
- Asymmetric loss for different quantiles
- Robust to outliers
- Useful for uncertainty quantification

**When to Use:**
- Quantile regression
- When you need prediction intervals
- When prediction uncertainty matters
- Risk assessment scenarios

---

## Classification Cost Functions

Used when predicting categories/classes (spam/not spam, cat/dog, etc.)

### 1. Binary Cross-Entropy (Log Loss)

**Formula:**
```
J = -(1/m) Σ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]

Where:
  y = Actual label (0 or 1)
  ŷ = Predicted probability (0 to 1)
```

**Interpretation:**
```
If y=1: Loss = -log(ŷ)      (penalizes if ŷ is close to 0)
If y=0: Loss = -log(1-ŷ)    (penalizes if ŷ is close to 1)
```

**Properties:**
- Works with probabilities
- Smooth and differentiable
- Strongly penalizes wrong confident predictions
- Standard for binary classification
- Requires probability outputs (sigmoid)

**Example:**
```
Actual: 1 (positive class)

Prediction: 0.9 → Loss = -log(0.9) = 0.105 (small loss)
Prediction: 0.5 → Loss = -log(0.5) = 0.693 (medium loss)
Prediction: 0.1 → Loss = -log(0.1) = 2.303 (large loss)

More confident wrong predictions = larger loss
```

---

### 2. Categorical Cross-Entropy

**Formula:**
```
J = -(1/m) Σ Σ yᵢⱼ·log(ŷᵢⱼ)

Where:
  i = Sample
  j = Class
  yᵢⱼ = 1 if sample i is class j, else 0
  ŷᵢⱼ = Predicted probability for class j
```

**Properties:**
- Generalization of binary cross-entropy
- For multi-class problems (3+ classes)
- Only the correct class contributes to loss
- Requires softmax activation
- Standard for multi-class classification

**Example (3 classes: Cat, Dog, Bird):**
```
Actual: [1, 0, 0] (Cat)

Prediction: [0.7, 0.2, 0.1]
Loss = -log(0.7) = 0.357 (good prediction)

Prediction: [0.3, 0.4, 0.3]
Loss = -log(0.3) = 1.204 (poor prediction)
```

---

### 3. Sparse Categorical Cross-Entropy

**Formula:**
```
Same as Categorical Cross-Entropy but:
- Takes class indices instead of one-hot encoded labels
- y is single integer (class number)
- Otherwise identical
```

**When to Use:**
- Multi-class when labels are integers (0, 1, 2, ...)
- Saves memory for many classes
- Alternative to one-hot encoding

---

### 4. Focal Loss

**Formula:**
```
FL(p, y) = -α(1-pₜ)^γ·log(pₜ)

Where:
  pₜ = Model predicted probability for true class
  α = Weighting factor (0 to 1)
  γ = Focusing parameter (typically 2)
```

**Properties:**
- Addresses class imbalance
- Down-weights easy examples
- Focuses on hard examples
- Particularly useful for object detection
- Introduced by RetinaNet paper

**When to Use:**
- Highly imbalanced datasets
- When hard examples matter more
- Computer vision tasks
- When standard cross-entropy fails

---

### 5. Hinge Loss

**Formula:**
```
L(y, ŷ) = max(0, 1 - y·ŷ)

Where:
  y = ±1 (class labels)
  ŷ = Predicted value (score)
```

**Properties:**
- Primarily used for SVMs
- Works with margin concept
- Penalizes wrong predictions and close decisions
- Non-differentiable at margin
- Sparse solutions

**When to Use:**
- Support Vector Machines (SVMs)
- When margin-based classification desired
- Linear classifiers

---

### 6. Squared Hinge Loss

**Formula:**
```
L(y, ŷ) = max(0, 1 - y·ŷ)²
```

**Properties:**
- Squared version of hinge loss
- More differentiable
- Penalizes violations more heavily
- Smoother optimization

**When to Use:**
- When you need smoother loss than hinge loss
- Support Vector Regression
- Margin-based learning

---

## Advanced Cost Functions

### 1. Triplet Loss

**Formula:**
```
L = max(d(aᵢ, pᵢ) - d(aᵢ, nᵢ) + m, 0)

Where:
  aᵢ = Anchor sample
  pᵢ = Positive sample (same class)
  nᵢ = Negative sample (different class)
  d(·,·) = Distance function
  m = Margin
```

**Use Cases:**
- Face recognition
- Metric learning
- Siamese networks
- Similarity learning

---

### 2. Contrastive Loss

**Formula:**
```
L = (1-Y)·0.5·d² + Y·0.5·max(m-d, 0)²

Where:
  Y = Label (0 if similar, 1 if different)
  d = Distance between samples
  m = Margin
```

**Use Cases:**
- Similarity learning
- Pairing samples
- Metric learning

---

### 3. KL Divergence (Kullback-Leibler)

**Formula:**
```
D_KL(P||Q) = Σ P(x)·log(P(x)/Q(x))

Where:
  P = True distribution
  Q = Predicted distribution
```

**Properties:**
- Measures difference between distributions
- Not symmetric
- Always non-negative
- Zero only if distributions identical

**Use Cases:**
- Distribution matching
- Variational autoencoders
- Knowledge distillation

---

### 4. Wasserstein Loss

**Formula:**
```
L = E[preds_real] - E[preds_fake]

(Simplified form)
```

**Properties:**
- From optimal transport theory
- Better gradient properties than cross-entropy
- Works with Wasserstein GANs
- More stable training

**Use Cases:**
- Generative adversarial networks (GANs)
- When standard loss is unstable
- Distribution generation

---

## Comparison and When to Use

### Regression Losses Summary

```
┌─────────────────┬──────────────┬─────────────────┬────────────┐
│ Loss Function   │ Penalty      │ Outlier Robust  │ Best For   │
├─────────────────┼──────────────┼─────────────────┼────────────┤
│ MSE/RMSE        │ Quadratic    │ No              │ Standard   │
│ MAE             │ Linear       │ Yes             │ Robust     │
│ Huber Loss      │ Hybrid       │ Very Yes        │ Safe       │
│ Log-Cosh        │ Smooth       │ Yes             │ Deep Lear. │
│ Quantile        │ Asymmetric   │ Yes             │ Intervals  │
└─────────────────┴──────────────┴─────────────────┴────────────┘
```

### Classification Losses Summary

```
┌───────────────────────┬──────────────┬──────────────────┬──────────┐
│ Loss Function         │ Task         │ Output Format    │ Best For │
├───────────────────────┼──────────────┼──────────────────┼──────────┤
│ Binary Cross-Entropy  │ Binary       │ Probability      │ Std 2-way│
│ Categorical Cross-E.  │ Multi-class  │ Probabilities    │ Std Mway │
│ Sparse Cat Cross-E.   │ Multi-class  │ Class index      │ Mway mem │
│ Focal Loss            │ Imbalanced   │ Probability      │ Imbalance│
│ Hinge Loss            │ Binary       │ Margin-based     │ SVM      │
│ Squared Hinge Loss    │ Binary       │ Margin-based     │ SVR      │
└───────────────────────┴──────────────┴──────────────────┴──────────┘
```

---

## Implementation Examples

### Python Code Examples

#### 1. Manual Implementations
```python
import numpy as np

# Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Mean Absolute Error
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Root Mean Squared Error
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

# Huber Loss
def huber_loss(y_true, y_pred, delta=1.0):
    errors = y_true - y_pred
    is_small_error = np.abs(errors) <= delta
    small_error_loss = 0.5 * errors ** 2
    large_error_loss = delta * (np.abs(errors) - 0.5 * delta)
    return np.mean(
        np.where(is_small_error, small_error_loss, large_error_loss)
    )

# Binary Cross-Entropy
def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * np.log(1 - y_pred)
    )

# Categorical Cross-Entropy
def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
```

#### 2. Using Scikit-learn
```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    log_loss,
    hinge_loss
)

y_true = [1, 2, 3, 4, 5]
y_pred = [1.1, 2.2, 2.9, 4.1, 5.2]

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

# Classification
y_true_class = [0, 1, 1, 0, 1]
y_pred_proba = [0.1, 0.9, 0.8, 0.3, 0.95]

bce = log_loss(y_true_class, y_pred_proba)
```

#### 3. Using TensorFlow/Keras
```python
import tensorflow as tf

# Regression losses
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()
huber = tf.keras.losses.Huber(delta=1.0)
log_cosh = tf.keras.losses.LogCosh()

# Classification losses
bce = tf.keras.losses.BinaryCrossentropy()
cce = tf.keras.losses.CategoricalCrossentropy()
sparse_cce = tf.keras.losses.SparseCategoricalCrossentropy()
focal = tf.keras.losses.BinaryFocalCrossentropy()

# Using in model
model.compile(optimizer='adam', loss=mse)
# or
model.compile(optimizer='adam', loss='mse')
```

---

## Key Concepts

### Loss vs Accuracy
```
Loss = What we minimize during training
Accuracy = What we care about for evaluation
(Not always the same!)
```

### Gradient-Based Optimization
```
θ_new = θ_old - α·∂J/∂θ

Where:
  θ = Parameters
  α = Learning rate
  ∂J/∂θ = Gradient of loss
```

**Why it matters:** Loss function must be differentiable for gradient descent!

### Regularization Terms
```
Total Loss = Data Loss + λ·Regularization

J_total(θ) = J_data(θ) + λ·||θ||²  (L2 regularization)
J_total(θ) = J_data(θ) + λ·||θ||   (L1 regularization)
```

---

## Choosing the Right Cost Function

### Decision Tree

```
                    Task?
                   /      \
            Regression    Classification
              /                \
        Outliers?          Imbalanced?
          /    \              /    \
        No     Yes          No     Yes
        |       |           |       |
       MSE   MAE/Huber   CrossE   FocalLoss
        |       |           |       |
      Good    Robust      Standard Robust
```

### Quick Selection Guide

**For Regression:**
1. Start with MSE (standard)
2. If outliers present → MAE or Huber Loss
3. If deep learning → Log-Cosh
4. If need uncertainty → Quantile Loss

**For Binary Classification:**
1. Use Binary Cross-Entropy (standard)
2. If imbalanced → Focal Loss
3. For SVM → Hinge Loss

**For Multi-class Classification:**
1. Use Categorical Cross-Entropy (standard)
2. If imbalanced → Focal Loss
3. If labels are integers → Sparse Categorical Cross-Entropy

---

## Summary Table

| Aspect | Regression | Classification |
|--------|-----------|-----------------|
| **Standard Loss** | MSE/RMSE | Cross-Entropy |
| **Outlier Robust** | MAE/Huber | Focal Loss |
| **Output Format** | Continuous | Probability |
| **Differentiable** | Most are | Most are |
| **Deep Learning** | Log-Cosh | Cross-Entropy |
| **SVM** | Hinge Loss | Hinge Loss |

---

## References and Further Reading

- Cross-entropy: Information theory concept
- Focal Loss: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
- Huber Loss: Robust statistics
- Triplet Loss: "FaceNet: A Unified Embedding for Face Recognition" (Schroff et al., 2015)
- Wasserstein Loss: Optimal transport theory

