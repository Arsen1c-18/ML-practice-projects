# Cost Functions: Quick Start Guide (2-Minute Version)

## The Absolute Essentials

### What is a Cost Function?
It measures how wrong your predictions are. **Lower = Better**

### Why Does It Matter?
Your model learns by minimizing it. Good choice = fast, accurate learning.

---

## The Golden Rule

| You're solving... | Use this loss | Why |
|------------------|--------------|-----|
| **Regression** (predict numbers) | **MSE** | Standard choice |
| **Regression** + outliers | **MAE** | Robust to extremes |
| **Binary** classification | **Binary Cross-Entropy** | Standard choice |
| **Multi-class** classification | **Categorical Cross-Entropy** | Standard choice |
| **Imbalanced** data | **Focal Loss** | Handles unfair class distribution |

---

## 30-Second Formula Reference

```python
# REGRESSION

MSE = average((actual - predicted)²)
MAE = average(|actual - predicted|)
Huber = MSE for small errors, MAE for large errors
Log-Cosh = smooth function combining MSE + MAE benefits

# CLASSIFICATION

Binary Cross-Entropy = -[y*log(p) + (1-y)*log(1-p)]
Categorical Cross-Entropy = -Σ(true_label * log(predicted_prob))
Focal Loss = -(1-prob)^2 * log(prob)  [good for imbalance]
Hinge Loss = max(0, 1 - y*score)  [for SVM]
```

---

## Implementation

### TensorFlow/Keras
```python
# Regression
model.compile(loss='mse')           # Standard
model.compile(loss='mae')           # With outliers
model.compile(loss='log_cosh')      # Deep learning

# Classification
model.compile(loss='binary_crossentropy')           # Binary
model.compile(loss='categorical_crossentropy')      # Multi-class
model.compile(loss='sparse_categorical_crossentropy') # Multi-class, integer labels
model.compile(loss='binary_focal_crossentropy')     # Imbalanced
```

### Scikit-learn (evaluation)
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
bce = log_loss(y_true, y_pred_proba)
```

### NumPy (manual)
```python
import numpy as np

# MSE
mse = np.mean((y_true - y_pred) ** 2)

# MAE
mae = np.mean(np.abs(y_true - y_pred))

# Binary Cross-Entropy
epsilon = 1e-7
y_pred = np.clip(y_pred, epsilon, 1-epsilon)
bce = -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
```

---

## Decision Flowchart (Pick Your Loss in 10 Seconds)

```
START
  ↓
Is it regression or classification?
  ├─ REGRESSION
  │   ↓
  │   Do you have outliers?
  │   ├─ YES  → Use MAE (or Huber)
  │   └─ NO   → Use MSE (or RMSE)
  │
  └─ CLASSIFICATION
      ↓
      Binary (2 classes) or Multi-class (3+ classes)?
      ├─ BINARY
      │   ├─ Imbalanced? ─→ YES  → Use Focal Loss
      │   └─            ─→ NO   → Use Binary Cross-Entropy
      │
      └─ MULTI-CLASS
          ├─ Imbalanced? ─→ YES  → Use Focal Loss
          └─            ─→ NO   → Use Categorical Cross-Entropy
```

---

## Visual Comparison

### Regression Losses (How sensitive to errors)

```
MSE:   Quadratic penalty (VERY sensitive to big errors)
       Error=1 → Loss=1
       Error=2 → Loss=4    ← Huge jump!
       Error=10 → Loss=100 ← Outlier dominates!

MAE:   Linear penalty (balanced)
       Error=1 → Loss=1
       Error=2 → Loss=2    ← Proportional
       Error=10 → Loss=10  ← Robust!

Huber: Smart combo
       Error=1 → Loss=0.5 (like MSE, smooth)
       Error=10 → Loss=9.5 (like MAE, robust)
```

### Classification Losses (Confidence penalties)

```
Binary Cross-Entropy:
  Correct with high confidence (0.99) → Loss ≈ 0.01 ✓
  Correct with low confidence (0.51) → Loss ≈ 0.69 ✗
  Wrong with high confidence (0.01) → Loss ≈ 4.6 ✗✗

Focal Loss: (for imbalanced data)
  Easy examples (high confidence) → Ignored
  Hard examples (low confidence) → Focused on
```

---

## Real Example: Which Loss?

### Example 1: House Price Prediction
- Problem: Regression (predicting prices)
- Data: Clean, no extreme outliers
- **Answer: Use MSE** (standard for regression)
- TensorFlow: `model.compile(loss='mse')`

### Example 2: Credit Risk Assessment
- Problem: Regression (predicting default probability as number)
- Data: Some extreme cases exist
- **Answer: Use MAE** (robust to outliers)
- TensorFlow: `model.compile(loss='mae')`

### Example 3: Email Spam Detection
- Problem: Binary classification
- Data: Balanced (50% spam, 50% not spam)
- **Answer: Use Binary Cross-Entropy** (standard for binary)
- TensorFlow: `model.compile(loss='binary_crossentropy')`

### Example 4: Animal Image Classification
- Problem: Multi-class classification (100 different animals)
- Data: Balanced
- **Answer: Use Categorical Cross-Entropy** (standard for multi-class)
- TensorFlow: `model.compile(loss='categorical_crossentropy')`

### Example 5: Disease Detection in Medical Images
- Problem: Binary classification
- Data: Imbalanced (99.9% healthy, 0.1% disease)
- **Answer: Use Focal Loss** (handles imbalance)
- TensorFlow: `model.compile(loss='binary_focal_crossentropy')`

---

## Common Mistakes

| ❌ Mistake | ✅ Fix |
|-----------|--------|
| Using MSE with extreme outliers | Use MAE or Huber |
| Using Binary CE for 3+ classes | Use Categorical CE |
| BCE without sigmoid activation | Add sigmoid to output layer |
| Not clipping probabilities (log(0) error) | Add epsilon: `np.clip(pred, 1e-7, 1-1e-7)` |
| Ignoring class imbalance | Use Focal Loss |
| Not matching activation to loss | See pairing table below |

---

## Activation Function Pairing

| Loss Function | Activation Required | Output |
|---------------|-------------------|--------|
| MSE / MAE | Linear (or None) | Any value (-∞ to ∞) |
| Binary Cross-Entropy | Sigmoid | Probability (0 to 1) |
| Categorical Cross-Entropy | Softmax | Probabilities (0-1, sum=1) |
| Hinge Loss | Linear | Score (-∞ to ∞) |

**Important**: Using wrong activation kills your model!

---

## Cheat Sheet: Loss Properties

| Property | MSE | MAE | Huber | BCE | CCE | Focal |
|----------|-----|-----|-------|-----|-----|-------|
| Type | Regression | Regression | Regression | Classification | Classification | Classification |
| For outliers | ❌ | ✅ | ✅ | N/A | N/A | ✅ |
| For imbalance | N/A | N/A | N/A | ❌ | ❌ | ✅ |
| Smooth | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Fast | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Standard | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |

---

## Training Tips

### Monitor Loss During Training

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100
)

# Plot loss
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Signs of healthy training:
✓ Loss decreases smoothly
✓ No big jumps
✓ Training and validation losses track together

✗ Signs of problems:
✗ Loss increasing
✗ NaN or infinity
✗ Validation diverging from training (overfitting)
```

### Adjust if Needed

```python
# If loss is NaN → Check:
- Are predictions clipped? (prevent log(0))
- Is learning rate too high?
- Are there invalid values in data?

# If loss not decreasing → Try:
- Check loss is for your problem type
- Ensure activation functions match loss
- Try different loss if stuck
- Reduce learning rate
```

---

## When to Switch Losses

### Start with standard, switch if:

1. **High error on outliers**
   - From: MSE → To: MAE

2. **Can't train (NaN losses)**
   - From: BCE → To: use epsilon clipping

3. **Model ignores minority class**
   - From: Cross-Entropy → To: Focal Loss

4. **Too many false positives/negatives**
   - Try: Weighted loss or threshold adjustment

---

## One-Minute Setup Guide

```python
# 1. Import
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 2. Create model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # For binary classification
])

# 3. Compile with appropriate loss
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # ← Choose based on your problem!
    metrics=['accuracy']
)

# 4. Train
model.fit(X_train, y_train, epochs=10)

# 5. Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
```

---

## Summary Checklist

- [ ] Identify problem type (regression/classification)
- [ ] Check for outliers or class imbalance
- [ ] Choose appropriate loss from table
- [ ] Set matching activation function
- [ ] Implement in framework
- [ ] Monitor loss during training
- [ ] Adjust if needed
- [ ] Evaluate with appropriate metrics

---

## Resources Provided

| File | What | When to Read |
|------|------|--------------|
| `cost_functions_summary.md` | Overview + workflows | First! (5 min) |
| `cost_functions_cheatsheet.md` | Quick reference | When you need lookup (1 min) |
| `cost_functions_formulas.md` | Math + deep details | For understanding formulas |
| `cost_functions_guide.md` | Complete reference | For comprehensive learning |
| `cost_functions_examples.py` | Working code | To run and experiment |
| `cost_functions_plots.png` | Visualizations | To understand loss behavior |

---

## Bottom Line

✅ **Know your problem type** → Rest follows
✅ **Start with standard loss** → Works 90% of time
✅ **Change only if needed** → Outliers? Imbalance? Try next tier
✅ **Monitor training** → Loss tells you if it's working
✅ **Match activation + loss** → Non-negotiable pair

**You're ready to go! Pick your loss and start training.** 🚀

