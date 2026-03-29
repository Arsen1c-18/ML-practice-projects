# 📊 Feature Engineering — Complete Guide

> A comprehensive reference on feature engineering techniques used in Machine Learning preprocessing pipelines.

---

## 📝 Summary

Feature engineering is the process of transforming raw data into meaningful inputs that improve a machine learning model's performance. It involves scaling numeric features, encoding categorical variables, creating new features from existing ones, selecting the most relevant features, and extracting useful signals from datetime columns. Done well, it can often matter more than the choice of algorithm itself.

---

## 1. 📐 Feature Scaling — Why It Matters

### What is it?
Feature scaling ensures that all numeric features lie on a comparable range. Many ML algorithms are sensitive to the magnitude of input values.

### Why is it important?
- Algorithms like **K-Nearest Neighbors**, **SVM**, **Logistic Regression**, and **Neural Networks** rely on distances or gradients — they are heavily affected by features with large ranges dominating smaller ones.
- **Gradient descent** converges faster when features are on similar scales.
- Tree-based models (Decision Trees, Random Forests, XGBoost) are **scale-invariant** and generally don't require scaling.

### Example Problem Without Scaling:
| Feature     | Range         |
|-------------|---------------|
| Age         | 0 – 100       |
| Salary      | 10,000 – 200,000 |

Without scaling, `Salary` would dominate distance-based calculations simply due to its magnitude.

---

## 2. 📏 Standardization (Z-score Normalization)

### Formula:
```
z = (x - μ) / σ
```
Where `μ` = mean, `σ` = standard deviation.

### What it does:
Transforms features to have **mean = 0** and **standard deviation = 1**.

### When to use:
- When data follows a **roughly normal (Gaussian) distribution**.
- Algorithms that assume normality: Linear/Logistic Regression, LDA, PCA.
- When **outliers** are present (Z-score handles them better than Min-Max).

### Example:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Output range: Roughly `-3 to +3` (unbounded, centered at 0)

---

## 3. 🔢 Normalization (Min-Max Scaling)

### Formula:
```
x' = (x - x_min) / (x_max - x_min)
```

### What it does:
Rescales features to a **fixed range**, typically `[0, 1]`.

### When to use:
- When you need values **strictly bounded** between 0 and 1 (e.g., image pixel values).
- Neural networks and algorithms using activation functions like sigmoid/tanh.
- When the data does **not** follow a Gaussian distribution.

### Example:
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

### ⚠️ Caution:
Very sensitive to **outliers** — a single extreme value will compress all others into a tiny range.

### Comparison Table:

| Aspect              | Standardization    | Normalization       |
|---------------------|--------------------|---------------------|
| Output Range        | ~(-3, 3), unbounded| [0, 1], bounded     |
| Handles Outliers    | Better             | Sensitive           |
| Assumes Distribution| Normal             | No assumption       |
| Best For            | Regression, PCA    | Neural nets, KNN    |

---

## 4. 🏷️ Label Encoding

### What is it?
Converts each **categorical value** into a unique **integer**.

### Example:
```
Color: ["Red", "Green", "Blue"]
  →    [  2,      1,      0   ]
```

### Code:
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["color_encoded"] = le.fit_transform(df["color"])
```

### When to use:
- **Ordinal** categorical variables where order matters (e.g., Low < Medium < High).
- Tree-based models that can handle integer-encoded categories.

### ⚠️ Warning:
Do **not** use for nominal (unordered) categories with non-tree models — the model may infer a false ordinal relationship (e.g., `Red=0 < Green=1 < Blue=2`).

---

## 5. 🔥 One-Hot Encoding

### What is it?
Creates a **new binary column** for each category value (0 or 1).

### Example:
```
Color: ["Red", "Green", "Blue"]

  → color_Red | color_Green | color_Blue
        1     |      0      |     0       ← Red
        0     |      1      |     0       ← Green
        0     |      0      |     1       ← Blue
```

### Code:
```python
import pandas as pd

df = pd.get_dummies(df, columns=["color"], drop_first=True)
# drop_first=True avoids multicollinearity (dummy variable trap)
```

### When to use:
- **Nominal** categories (no natural order).
- Linear models, Logistic Regression, SVM, Neural Networks.

### ⚠️ Caution:
- High-cardinality columns (e.g., 500 cities) → **curse of dimensionality**. Use target encoding or embeddings instead.
- Always drop one column (`drop_first=True`) to avoid multicollinearity.

---

## 6. 🛠️ Feature Creation (Feature Engineering)

### What is it?
Deriving **new features** from existing ones to better capture patterns in data.

### Techniques:

#### a) Arithmetic Combinations
```python
df["BMI"] = df["weight_kg"] / (df["height_m"] ** 2)
df["price_per_sqft"] = df["price"] / df["area"]
```

#### b) Polynomial Features
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

#### c) Aggregations / Group Statistics
```python
df["avg_spend_by_city"] = df.groupby("city")["spend"].transform("mean")
```

#### d) Interaction Terms
```python
df["age_x_income"] = df["age"] * df["income"]
```

#### e) Domain-Specific Features
- Text: word count, sentiment score, TF-IDF
- Finance: rolling averages, volatility
- E-commerce: days since last purchase, purchase frequency

### Why it matters:
Raw features often don't directly capture the relationship with the target. Feature creation exposes hidden signals and can dramatically improve model performance.

---

## 7. 🎯 Feature Selection

### What is it?
Identifying and keeping only the **most relevant features**, reducing noise and dimensionality.

### Why do it?
- Removes **irrelevant/redundant** features that add noise.
- Reduces **overfitting** risk.
- Speeds up **training time**.
- Improves **model interpretability**.

### Methods:

#### a) Filter Methods (Statistical Tests)
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
```
Uses: Correlation, Chi-square, ANOVA F-test.

#### b) Wrapper Methods
- **RFE (Recursive Feature Elimination)**: Trains model, removes weakest features iteratively.
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

rfe = RFE(LogisticRegression(), n_features_to_select=5)
rfe.fit(X, y)
```

#### c) Embedded Methods
- Feature importance from **tree-based models**.
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_
```

#### d) Variance Threshold (remove near-constant features)
```python
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0.01)
X_reduced = sel.fit_transform(X)
```

---

## 8. 📅 Handling Datetime Features

### What is it?
Extracting meaningful components and signals from raw **date/time columns**.

### Basic Extraction:
```python
df["datetime"] = pd.to_datetime(df["datetime"])

df["year"]        = df["datetime"].dt.year
df["month"]       = df["datetime"].dt.month
df["day"]         = df["datetime"].dt.day
df["hour"]        = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek   # 0=Monday, 6=Sunday
df["is_weekend"]  = df["datetime"].dt.dayofweek >= 5
df["quarter"]     = df["datetime"].dt.quarter
df["week_of_year"]= df["datetime"].dt.isocalendar().week
```

### Advanced Features:
```python
# Days since a reference event
df["days_since_signup"] = (df["purchase_date"] - df["signup_date"]).dt.days

# Is it a business day?
import numpy as np
df["is_business_day"] = df["datetime"].dt.dayofweek < 5

# Time-based cyclical encoding (important for hours/months)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
```

### ⚠️ Why cyclical encoding?
Hour 23 and Hour 0 (midnight) are close in time, but numerically far apart. **Sine/cosine encoding** preserves this circular relationship.

### Useful Time-Based Features by Domain:
| Domain     | Useful Features                              |
|------------|----------------------------------------------|
| Retail     | is_holiday, days_until_payday, season        |
| Finance    | market_open, trading_day, fiscal_quarter     |
| Healthcare | age_at_visit, days_since_last_appointment    |
| Web/App    | hour_of_day, session_duration, recency       |

---

## 🗂️ Quick Reference Cheat Sheet

| Technique           | Input Type    | Use Case                                 |
|---------------------|---------------|------------------------------------------|
| Standardization     | Numeric       | Normal dist., PCA, regression            |
| Min-Max Scaling     | Numeric       | Neural nets, bounded range needed        |
| Label Encoding      | Categorical   | Ordinal data or tree-based models        |
| One-Hot Encoding    | Categorical   | Nominal data, linear/neural models       |
| Feature Creation    | Any           | Capture domain knowledge, interactions   |
| Feature Selection   | Any           | Reduce dimensions, remove noise          |
| Datetime Features   | DateTime      | Time series, seasonal patterns           |

---

*Happy Engineering! 🚀*
