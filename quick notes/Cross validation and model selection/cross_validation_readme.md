# 🔁 Cross Validation & Model Selection — Complete Guide

> A comprehensive reference on evaluation strategies used to reliably assess and select machine learning models.

---

## 📝 Summary

Cross-validation is the practice of evaluating a model on held-out data to estimate how well it will perform on unseen examples. Rather than trusting a single train/test split, CV techniques repeatedly train and evaluate across different data partitions, producing a more reliable performance estimate. Choosing the right CV strategy — and understanding validation curves — is essential for avoiding overfitting, tuning hyperparameters, and selecting the best model for deployment.

---

## 1. ✂️ Train-Test Split

### What is it?
The simplest evaluation strategy: divide the dataset into two non-overlapping parts — one for **training** and one for **testing**.

### How it works:
```
Full Dataset (1000 rows)
├── Training Set  → 800 rows (80%)  — model learns from this
└── Test Set      → 200 rows (20%)  — model is evaluated on this
```

### Code:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # reproducibility
    stratify=y           # preserve class distribution (for classification)
)
```

### Common Split Ratios:
| Split      | Train | Test |
|------------|-------|------|
| 80/20      | 80%   | 20%  |
| 70/30      | 70%   | 30%  |
| 75/25      | 75%   | 25%  |

### ✅ Pros:
- Simple and fast.
- Good for large datasets.

### ⚠️ Cons:
- High **variance** — result depends heavily on *which* rows end up in the test set.
- Wastes data — the test set is never used for training.
- Unreliable for **small datasets**.

---

## 2. 🔄 K-Fold Cross-Validation

### What is it?
Splits the dataset into **K equal folds**. The model is trained K times — each time using K-1 folds for training and 1 fold for validation. Final score = average of all K scores.

### How it works (K=5):
```
Fold 1: [TEST ] [TRAIN] [TRAIN] [TRAIN] [TRAIN]
Fold 2: [TRAIN] [TEST ] [TRAIN] [TRAIN] [TRAIN]
Fold 3: [TRAIN] [TRAIN] [TEST ] [TRAIN] [TRAIN]
Fold 4: [TRAIN] [TRAIN] [TRAIN] [TEST ] [TRAIN]
Fold 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST ]

Final Score = mean(score_1, score_2, ..., score_5)
```

### Code:
```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

print(f"Scores:  {scores}")
print(f"Mean:    {scores.mean():.4f}")
print(f"Std Dev: {scores.std():.4f}")
```

### Choosing K:
| K Value | Bias      | Variance  | Speed    | Recommended When         |
|---------|-----------|-----------|----------|--------------------------|
| K = 5   | Moderate  | Moderate  | Fast     | General use (default)    |
| K = 10  | Low       | Low       | Moderate | Standard best practice   |
| K = N   | Very Low  | Very High | Slow     | Small datasets (LOOCV)   |

### ✅ Pros:
- Every data point is used for both training and validation.
- Much lower variance than a single train-test split.

### ⚠️ Cons:
- K× more computation than a single split.
- Does **not** preserve class balance unless stratified.

---

## 3. ⚖️ Stratified K-Fold

### What is it?
A variant of K-Fold that ensures **each fold preserves the original class distribution**. Critical for imbalanced classification problems.

### Why it matters:
```
Original Dataset: 90% Class 0 | 10% Class 1

Standard K-Fold  → Some folds may have 0% Class 1 (model never sees minority class)
Stratified K-Fold → Every fold has ~90% Class 0 | ~10% Class 1 ✅
```

### Code:
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=skf, scoring="f1")
print(f"Stratified CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
```

### ✅ Pros:
- Prevents folds with missing or underrepresented classes.
- More **reliable estimates** for imbalanced datasets.
- Default behavior in `cross_val_score` when the estimator is a classifier.

### ⚠️ Cons:
- Only applicable to **classification** (not regression).

---

## 4. 🔍 Leave-One-Out Cross-Validation (LOOCV)

### What is it?
An extreme case of K-Fold where **K = N** (number of samples). Each iteration uses a single data point as the test set and all remaining N-1 points as training data.

### How it works (N=5):
```
Iter 1: [TEST] [TRAIN] [TRAIN] [TRAIN] [TRAIN]
Iter 2: [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN]
Iter 3: [TRAIN] [TRAIN] [TEST] [TRAIN] [TRAIN]
Iter 4: [TRAIN] [TRAIN] [TRAIN] [TEST] [TRAIN]
Iter 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST]
```
Final score = average of N individual scores.

### Code:
```python
from sklearn.model_selection import LeaveOneOut, cross_val_score

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring="accuracy")

print(f"LOOCV Accuracy: {scores.mean():.4f}")
```

### ✅ Pros:
- Uses maximum possible training data each iteration.
- Near-**unbiased** performance estimate.
- Deterministic — no randomness involved.

### ⚠️ Cons:
- Extremely **computationally expensive** for large datasets (N model fits).
- High **variance** in individual test scores (each test set is just 1 point).
- Impractical for datasets with N > a few hundred.

### When to use:
- Very **small datasets** (N < 100) where every data point is precious.
- Medical or scientific datasets where data collection is expensive.

---

## 5. 📈 Time Series Split

### What is it?
A CV strategy for **time-ordered data** that respects temporal order — future data is never used to predict the past (no data leakage).

### How it works:
```
Split 1: [TRAIN: t1–t3          ] [TEST: t4]
Split 2: [TRAIN: t1–t4          ] [TEST: t5]
Split 3: [TRAIN: t1–t5          ] [TEST: t6]
Split 4: [TRAIN: t1–t6          ] [TEST: t7]

Training window always grows forward in time ➡
```

### Code:
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Fold {fold+1}: {score:.4f}")
```

### ⚠️ Why not use standard K-Fold for time series?
Standard K-Fold randomly shuffles data — a model could train on data from 2024 and predict 2022, causing **data leakage** and falsely inflated scores.

### ✅ Pros:
- Prevents **temporal data leakage**.
- Mimics real-world deployment (always predict future from past).

### ⚠️ Cons:
- Early folds have **much less training data**.
- Cannot shuffle data.

---

## 6. 📉 Validation Curves

### What is it?
A diagnostic plot that shows how **training score** and **validation score** change as a single **hyperparameter** varies. Used to detect underfitting and overfitting.

### How it works:
Train and evaluate the model across a range of hyperparameter values, plotting both training and CV scores.

### Code:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier

param_range = [1, 5, 10, 20, 50, 100, 200]

train_scores, val_scores = validation_curve(
    RandomForestClassifier(),
    X, y,
    param_name="n_estimators",
    param_range=param_range,
    cv=5,
    scoring="accuracy"
)

train_mean = train_scores.mean(axis=1)
val_mean   = val_scores.mean(axis=1)

plt.plot(param_range, train_mean, label="Train Score")
plt.plot(param_range, val_mean,   label="Val Score")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Validation Curve")
plt.show()
```

### Reading the Curve:

```
Score
  │
1 │  ████ Train
  │  ████████
  │     ████████████████ ← Train plateaus (good)
  │         ████
  │            ████ Val
  │               ████████████ ← Val improves then plateaus (good)
  └─────────────────────────────────────► Hyperparameter Value
```

| Pattern Observed                          | Diagnosis        | Fix                                      |
|-------------------------------------------|------------------|------------------------------------------|
| Both scores low                           | Underfitting     | Increase model complexity                |
| Train high, Val low, big gap              | Overfitting      | Regularize, reduce complexity            |
| Train ≈ Val, both high                    | Good fit ✅      | Use this hyperparameter value            |
| Val improves then degrades                | Sweet spot exists| Pick value at Val score peak             |

### Related: Learning Curves
Learning curves plot score vs **training set size** (not hyperparameter), diagnosing whether more data would help.

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)
```

---

## 🗂️ Quick Reference Cheat Sheet

| Technique            | Best For                        | Key Concern             | Computation   |
|----------------------|---------------------------------|-------------------------|---------------|
| Train-Test Split     | Large datasets, quick baseline  | High variance           | Very Fast     |
| K-Fold CV            | General purpose                 | Class imbalance         | Moderate      |
| Stratified K-Fold    | Imbalanced classification       | —                       | Moderate      |
| Leave-One-Out CV     | Very small datasets             | Very slow on large data | Very Slow     |
| Time Series Split    | Temporal/sequential data        | Data leakage            | Moderate      |
| Validation Curves    | Hyperparameter tuning           | Choosing right param    | Slow (sweep)  |

---

## 🧭 Decision Guide — Which CV Should I Use?

```
Is your data time-ordered?
  └─ YES → Time Series Split

Is your dataset very small (N < 100)?
  └─ YES → Leave-One-Out CV

Is it a classification problem with class imbalance?
  └─ YES → Stratified K-Fold

Do you just need a quick baseline?
  └─ YES → Train-Test Split

General / default recommendation?
  └─ K-Fold (K=5 or K=10) or Stratified K-Fold
```

---

*Validate well, generalize better! 🚀*
