# Regularization Techniques: L1, L2, and Elastic Net

## Overview
Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function. It discourages the model from learning overly complex patterns that don't generalize well to new data.

---

## 1. L1 Regularization (Lasso)

### What It Does
L1 regularization adds a penalty proportional to the **absolute values** of the coefficients.

### Key Characteristics
- **Penalty Effect**: Shrinks some coefficients all the way to **zero**
- **Feature Selection**: Automatically performs feature selection by eliminating less important features
- **Sparsity**: Creates sparse models (many coefficients become exactly zero)
- **Shape**: Penalty term forms a diamond-shaped constraint region

### When to Use L1
✅ **Use when you have:**
- High-dimensional datasets (many features)
- Need to identify which features are important
- Believe many features are irrelevant
- Want automatic feature selection
- Working with interpretability as a priority

### Example Use Case
- **Text Classification**: You have 10,000 words; L1 helps identify which words matter most
- **Gene Expression**: Filter genes that are actually predictive

### Formula Representation
```
Loss = Original Loss + λ × Σ|coefficient|
```

---

## 2. L2 Regularization (Ridge)

### What It Does
L2 regularization adds a penalty proportional to the **squared values** of the coefficients.

### Key Characteristics
- **Penalty Effect**: Shrinks all coefficients towards zero, but **none become exactly zero**
- **No Feature Elimination**: Keeps all features, just reduces their impact
- **Smooth**: Gradually penalizes large coefficients
- **Shape**: Penalty term forms a circular constraint region

### When to Use L2
✅ **Use when you have:**
- Multicollinearity (correlated features)
- Want to keep all features in the model
- Prefer distributed weights across features
- Small to medium-sized datasets
- Features are similarly important

### Example Use Case
- **Polynomial Regression**: Multiple polynomial terms that are correlated
- **Image Classification**: Where all pixel features should contribute slightly
- **When features measure similar things**: L2 distributes weight across correlated features

### Formula Representation
```
Loss = Original Loss + λ × Σ(coefficient²)
```

---

## 3. Elastic Net

### What It Does
Elastic Net is a **combination** of both L1 and L2 regularization. It applies both penalties simultaneously.

### Key Characteristics
- **Hybrid Approach**: Gets benefits of both L1 and L2
- **Feature Selection**: Can eliminate features (like L1)
- **Grouping Effect**: Groups correlated features together (like L2)
- **Balance**: You control the mix between L1 and L2 penalties
- **Flexibility**: Most versatile option

### When to Use Elastic Net
✅ **Use when you have:**
- High-dimensional data with correlated features
- Want feature selection AND handle multicollinearity
- Not sure whether L1 or L2 is better
- Need both sparsity and smooth coefficient distribution
- Medium to large datasets

### Example Use Case
- **Real-world datasets**: Most real data has both many features and correlations
- **Healthcare**: Many medical features that correlate but some are irrelevant
- **Recommendation Systems**: Thousands of features with varying importance and relationships

### Formula Representation
```
Loss = Original Loss + λ₁ × Σ|coefficient| + λ₂ × Σ(coefficient²)
```

Or equivalently:
```
Loss = Original Loss + λ × [α × Σ|coefficient| + (1-α) × Σ(coefficient²)]
```

---

## Quick Comparison Table

| Aspect | L1 (Lasso) | L2 (Ridge) | Elastic Net |
|--------|-----------|-----------|------------|
| **Feature Selection** | ✅ Yes (exact zero) | ❌ No | ✅ Yes |
| **Handles Multicollinearity** | ⚠️ Arbitrary choice | ✅ Yes (well) | ✅ Yes |
| **Sparse Model** | ✅ Very sparse | ❌ Dense | ⚠️ Moderately sparse |
| **Computational Speed** | Fast | Fast | Slightly slower |
| **Interpretability** | ✅ Excellent | ❌ All features included | ✅ Good |
| **Correlated Features** | Picks one | Distributes weight | Groups them |
| **Complexity** | Simple | Simple | Moderate |

---

## Regularization Parameter (λ - Lambda)

### What Is Lambda?
Lambda (λ) is a **hyperparameter** that controls the **strength** of regularization:
- **λ = 0**: No regularization (original model, can overfit)
- **λ = small value**: Weak regularization (model still somewhat flexible)
- **λ = large value**: Strong regularization (model heavily constrained, can underfit)
- **λ = infinity**: All coefficients become zero (useless model)

### How Lambda Works

#### Low λ (Weak Regularization)
```
✅ Model learns complex patterns
❌ Overfitting risk is high
❌ High variance in predictions
```

#### High λ (Strong Regularization)
```
✅ Simpler model, less overfitting
❌ May miss important patterns
❌ High bias, underfitting risk
```

#### Optimal λ (Sweet Spot)
```
✅ Good generalization
✅ Captures important patterns
✅ Avoids overfitting and underfitting
```

### Choosing Lambda: The Trade-off
```
Total Error = Bias Error + Variance Error

Low λ   → Low Bias, High Variance (Overfitting)
High λ  → High Bias, Low Variance (Underfitting)
Optimal → Balanced Bias-Variance
```

### How to Find Optimal Lambda
1. **Cross-Validation** (Most Common)
   - Try different λ values
   - Use k-fold cross-validation
   - Pick λ with best validation performance

2. **Grid Search**
   - Test λ values: [0.001, 0.01, 0.1, 1, 10, 100]
   - Evaluate each on validation set

3. **Random Search**
   - Randomly sample λ values
   - More efficient for large ranges

4. **Visual Method**
   - Plot training vs validation error vs λ
   - Find the "elbow" point

### Lambda in Code Example
```python
# Different lambda values to test
lambdas = [0.001, 0.01, 0.1, 1, 10, 100]

# Use cross-validation to find best lambda
# Sklearn does this automatically with GridSearchCV
```

### Impact of Lambda on Model Behavior
```
λ = 0.001  →  Flexible, complex model (may overfit)
λ = 0.1    →  Moderate constraints (balanced)
λ = 10     →  Heavy constraints (may underfit)
```

---

## Decision Tree: Which Regularization to Use?

```
Do you have many features?
├─ NO → Use L2 (Ridge)
│
└─ YES: Do you want automatic feature selection?
    ├─ YES: Are features correlated?
    │   ├─ NO → Use L1 (Lasso)
    │   └─ YES → Use Elastic Net ✅
    │
    └─ NO: Are features correlated?
        ├─ YES → Use L2 (Ridge) ✅
        └─ NO → L1 or L2 (both work)
```

---

## Practical Tips

### L1 Tips
- Start with λ around 0.01 to 0.1
- Use when interpretability matters
- Good for high-dimensional sparse data
- Performs well with categorical data one-hot encoded

### L2 Tips
- Try λ around 0.01 to 1.0
- Better with correlated features
- Produces more stable coefficients
- Good for continuous features

### Elastic Net Tips
- Start with equal mix of L1 and L2 (α = 0.5)
- Fine-tune α between 0 and 1
- More robust than pure L1 or L2
- Best general-purpose choice if unsure

### General Tips
- **Always scale features** before regularization
- **Use cross-validation** to find optimal λ
- **Start simple**: Try L2 first, then L1, then Elastic Net if needed
- **Monitor both** training and validation error
- **Domain knowledge** should guide your choice

---

## Real-World Example: House Price Prediction

**Scenario**: Predicting house prices with features like: size, bedrooms, location, age, etc.

### Using L1 (Lasso)
```
If age feature is irrelevant → coefficient becomes 0
Other features weighted by importance
Result: "Size, Location, and Bedrooms matter most"
```

### Using L2 (Ridge)
```
All features keep some weight
Correlated features (e.g., size & bedrooms) 
   → weights distributed between them
Result: All features contribute smoothly
```

### Using Elastic Net
```
Important features kept (L1 effect)
Correlated features grouped (L2 effect)
Result: Balance between simplicity and handling correlations
```

---

## Summary Chart

| Situation | Best Choice | Why |
|-----------|-------------|-----|
| Few features, clear importance | L2 | Simple and handles correlations |
| Many features, need sparsity | L1 | Automatic feature selection |
| Many features + correlations | **Elastic Net** | Gets both benefits |
| Unsure about correlations | **Elastic Net** | Most robust choice |
| Need perfect interpretability | L1 | Only non-zero features matter |
| Multicollinearity is main issue | L2 | Distributes weight across correlated features |

---

## Final Takeaway

**Remember**: Regularization is about **preventing overfitting**, not about picking the "best" method. 

- **Start with L2** (Ridge) if you're new to regularization
- **Switch to L1** (Lasso) when feature selection is important
- **Use Elastic Net** when you have both many features and correlations
- **Always tune λ** with cross-validation for your specific dataset

The goal is finding the sweet spot between **bias and variance** through careful λ tuning! 🎯
