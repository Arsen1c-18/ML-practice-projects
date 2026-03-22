"""
Cost Functions in Machine Learning
Complete Implementation with Visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import xlogy
import seaborn as sns

print("=" * 80)
print("COST FUNCTIONS IN MACHINE LEARNING")
print("=" * 80)

# ============================================================================
# PART 1: REGRESSION COST FUNCTIONS
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: REGRESSION COST FUNCTIONS")
print("=" * 80)

# Create sample data
np.random.seed(42)
y_actual = np.array([10, 20, 30, 40, 50])
y_pred = np.array([12, 18, 32, 38, 52])

errors = y_actual - y_pred

print("\nSample Data:")
print(f"Actual:     {y_actual}")
print(f"Predicted:  {y_pred}")
print(f"Errors:     {errors}")

# ============================================================================
# 1.1 Mean Squared Error (MSE)
# ============================================================================
print("\n--- 1. Mean Squared Error (MSE) ---")

def mse(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

mse_value = mse(y_actual, y_pred)
print(f"MSE = (1/m) × Σ(y - ŷ)²")
print(f"MSE = {mse_value:.4f}")
print(f"Interpretation: Average squared error is {mse_value:.4f}")

# ============================================================================
# 1.2 Root Mean Squared Error (RMSE)
# ============================================================================
print("\n--- 2. Root Mean Squared Error (RMSE) ---")

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))

rmse_value = rmse(y_actual, y_pred)
print(f"RMSE = √MSE = √({mse_value:.4f}) = {rmse_value:.4f}")
print(f"Interpretation: Average error is ±{rmse_value:.4f} units")

# ============================================================================
# 1.3 Mean Absolute Error (MAE)
# ============================================================================
print("\n--- 3. Mean Absolute Error (MAE) ---")

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

mae_value = mae(y_actual, y_pred)
print(f"MAE = (1/m) × Σ|y - ŷ|")
print(f"MAE = {mae_value:.4f}")
print(f"Interpretation: Average absolute error is {mae_value:.4f} units")

# ============================================================================
# 1.4 Huber Loss
# ============================================================================
print("\n--- 4. Huber Loss ---")

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber Loss - Robust combination of MSE and MAE"""
    errors = y_true - y_pred
    is_small_error = np.abs(errors) <= delta
    small_error_loss = 0.5 * errors ** 2
    large_error_loss = delta * (np.abs(errors) - 0.5 * delta)
    return np.mean(
        np.where(is_small_error, small_error_loss, large_error_loss)
    )

huber_value = huber_loss(y_actual, y_pred, delta=1.0)
print(f"Huber Loss (δ=1.0) = {huber_value:.4f}")
print(f"Combines MSE for small errors and MAE for large errors")

# ============================================================================
# 1.5 Log-Cosh Loss
# ============================================================================
print("\n--- 5. Log-Cosh Loss ---")

def log_cosh_loss(y_true, y_pred):
    """Log-Cosh Loss"""
    errors = y_pred - y_true
    return np.mean(np.log(np.cosh(errors)))

log_cosh_value = log_cosh_loss(y_actual, y_pred)
print(f"Log-Cosh Loss = {log_cosh_value:.4f}")
print(f"Smooth loss function combining benefits of MSE and MAE")

# ============================================================================
# Summary of Regression Losses
# ============================================================================
print("\n" + "-" * 80)
print("REGRESSION LOSS SUMMARY")
print("-" * 80)
print(f"MSE:           {mse_value:>10.4f}")
print(f"RMSE:          {rmse_value:>10.4f} ← Same units as target")
print(f"MAE:           {mae_value:>10.4f} ← Robust to outliers")
print(f"Huber Loss:    {huber_value:>10.4f} ← Balanced approach")
print(f"Log-Cosh:      {log_cosh_value:>10.4f} ← Smooth optimization")

# ============================================================================
# PART 2: CLASSIFICATION COST FUNCTIONS
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: CLASSIFICATION COST FUNCTIONS")
print("=" * 80)

# ============================================================================
# 2.1 Binary Cross-Entropy
# ============================================================================
print("\n--- 1. Binary Cross-Entropy (Log Loss) ---")

def binary_crossentropy(y_true, y_pred):
    """Binary Cross-Entropy Loss"""
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * np.log(1 - y_pred)
    )

y_true_binary = np.array([0, 1, 1, 0, 1])
y_pred_proba = np.array([0.1, 0.9, 0.8, 0.3, 0.95])

bce_value = binary_crossentropy(y_true_binary, y_pred_proba)
print(f"\nActual:        {y_true_binary}")
print(f"Predicted:     {np.round(y_pred_proba, 2)}")
print(f"Binary Cross-Entropy = {bce_value:.4f}")
print(f"Interpretation: Lower is better, heavily penalizes wrong confident predictions")

# Individual losses
individual_bce = -y_true_binary * np.log(np.clip(y_pred_proba, 1e-7, 1)) - \
                  (1 - y_true_binary) * np.log(np.clip(1 - y_pred_proba, 1e-7, 1))
print(f"\nPer-sample losses: {np.round(individual_bce, 4)}")
print(f"  Sample 0: Actual=0, Pred=0.10 → Loss={individual_bce[0]:.4f} ✓")
print(f"  Sample 1: Actual=1, Pred=0.90 → Loss={individual_bce[1]:.4f} ✓")

# ============================================================================
# 2.2 Categorical Cross-Entropy
# ============================================================================
print("\n--- 2. Categorical Cross-Entropy ---")

def categorical_crossentropy(y_true, y_pred):
    """Categorical Cross-Entropy Loss"""
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Multi-class example (3 classes)
y_true_multiclass = np.array([
    [1, 0, 0],  # Class 0
    [0, 1, 0],  # Class 1
    [0, 0, 1],  # Class 2
    [1, 0, 0],  # Class 0
    [0, 1, 0],  # Class 1
])

y_pred_multiclass = np.array([
    [0.7, 0.2, 0.1],   # Good prediction for class 0
    [0.1, 0.8, 0.1],   # Good prediction for class 1
    [0.1, 0.2, 0.7],   # Good prediction for class 2
    [0.3, 0.4, 0.3],   # Poor prediction for class 0
    [0.4, 0.4, 0.2],   # Poor prediction for class 1
])

cce_value = categorical_crossentropy(y_true_multiclass, y_pred_multiclass)
print(f"Categorical Cross-Entropy = {cce_value:.4f}")

individual_cce = -np.sum(y_true_multiclass * np.log(np.clip(y_pred_multiclass, 1e-7, 1)), axis=1)
print(f"\nPer-sample losses: {np.round(individual_cce, 4)}")
for i, loss in enumerate(individual_cce):
    actual_class = np.argmax(y_true_multiclass[i])
    pred_prob = y_pred_multiclass[i, actual_class]
    print(f"  Sample {i}: Class {actual_class}, Pred Prob={pred_prob:.2f} → Loss={loss:.4f}")

# ============================================================================
# 2.3 Focal Loss
# ============================================================================
print("\n--- 3. Focal Loss (for imbalanced data) ---")

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal Loss - addresses class imbalance"""
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate base cross entropy
    ce = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    
    # Get pt (probability of true class)
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    
    # Calculate focal weight
    focal_weight = alpha * (1 - pt) ** gamma
    
    return np.mean(focal_weight * ce)

focal_value = focal_loss(y_true_binary, y_pred_proba, gamma=2.0)
print(f"Focal Loss (γ=2.0) = {focal_value:.4f}")
print(f"Down-weights easy examples, focuses on hard examples")

# ============================================================================
# 2.4 Hinge Loss
# ============================================================================
print("\n--- 4. Hinge Loss (for SVM) ---")

def hinge_loss(y_true, y_pred):
    """Hinge Loss"""
    # Convert to {-1, +1}
    y_true_svm = 2 * y_true - 1
    return np.mean(np.maximum(0, 1 - y_true_svm * y_pred))

y_pred_scores = np.array([-0.5, 0.8, 0.6, -0.2, 0.9])
hl_value = hinge_loss(y_true_binary, y_pred_scores)
print(f"Hinge Loss = {hl_value:.4f}")
print(f"Used in Support Vector Machines (SVM)")

# ============================================================================
# PART 3: EFFECTS OF OUTLIERS
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: EFFECT OF OUTLIERS ON DIFFERENT LOSSES")
print("=" * 80)

# Normal case
y_normal = np.array([10, 20, 30, 40, 50])
y_pred_normal = np.array([11, 19, 31, 39, 51])

# With outlier
y_outlier = np.array([10, 20, 30, 40, 500])
y_pred_outlier = np.array([11, 19, 31, 39, 51])

print("\nNORMAL CASE (all errors are ~1):")
print(f"  MSE:  {mse(y_normal, y_pred_normal):.4f}")
print(f"  MAE:  {mae(y_normal, y_pred_normal):.4f}")
print(f"  Huber: {huber_loss(y_normal, y_pred_normal):.4f}")

print("\nWITH OUTLIER (last value is 500):")
print(f"  MSE:  {mse(y_outlier, y_pred_outlier):.4f} ← Heavily affected!")
print(f"  MAE:  {mae(y_outlier, y_pred_outlier):.4f} ← More robust")
print(f"  Huber: {huber_loss(y_outlier, y_pred_outlier):.4f} ← Balanced")

print("\nObservation: MSE is heavily influenced by outliers")
print("             MAE is more robust to outliers")
print("             Huber Loss is a good middle ground")

# ============================================================================
# PART 4: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Regression Loss Functions
ax1 = axes[0, 0]
errors_range = np.linspace(-3, 3, 100)

mse_vals = errors_range ** 2
mae_vals = np.abs(errors_range)
huber_vals = np.where(
    np.abs(errors_range) <= 1,
    0.5 * errors_range ** 2,
    np.abs(errors_range) - 0.5
)
log_cosh_vals = np.log(np.cosh(errors_range))

ax1.plot(errors_range, mse_vals, label='MSE', linewidth=2)
ax1.plot(errors_range, mae_vals, label='MAE', linewidth=2)
ax1.plot(errors_range, huber_vals, label='Huber (δ=1)', linewidth=2)
ax1.plot(errors_range, log_cosh_vals, label='Log-Cosh', linewidth=2)
ax1.set_xlabel('Error (y - ŷ)', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Regression Loss Functions', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 10])

# 2. Binary Cross-Entropy
ax2 = axes[0, 1]
probs = np.linspace(0.01, 0.99, 100)

bce_true = -np.log(probs)  # When y=1
bce_false = -np.log(1 - probs)  # When y=0

ax2.plot(probs, bce_true, label='When y=1', linewidth=2, color='red')
ax2.plot(probs, bce_false, label='When y=0', linewidth=2, color='blue')
ax2.set_xlabel('Predicted Probability', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('Binary Cross-Entropy', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

# 3. Hinge Loss
ax3 = axes[0, 2]
scores = np.linspace(-2, 2, 100)

hinge_vals = np.maximum(0, 1 - scores)
squared_hinge_vals = np.maximum(0, 1 - scores) ** 2

ax3.plot(scores, hinge_vals, label='Hinge Loss', linewidth=2)
ax3.plot(scores, squared_hinge_vals, label='Squared Hinge Loss', linewidth=2)
ax3.set_xlabel('Predicted Score (y·ŷ)', fontsize=11)
ax3.set_ylabel('Loss', fontsize=11)
ax3.set_title('Hinge Loss', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='Margin')

# 4. Outlier Effect Comparison
ax4 = axes[1, 0]
outlier_factors = np.array([1, 2, 5, 10, 20, 50])
mse_vals_outlier = []
mae_vals_outlier = []
huber_vals_outlier = []

for factor in outlier_factors:
    y_test = np.array([10, 20, 30, 40, 50 * factor])
    y_pred_test = np.array([11, 19, 31, 39, 51])
    
    mse_vals_outlier.append(mse(y_test, y_pred_test))
    mae_vals_outlier.append(mae(y_test, y_pred_test))
    huber_vals_outlier.append(huber_loss(y_test, y_pred_test))

ax4.semilogy(outlier_factors, mse_vals_outlier, 'o-', label='MSE', linewidth=2, markersize=8)
ax4.semilogy(outlier_factors, mae_vals_outlier, 's-', label='MAE', linewidth=2, markersize=8)
ax4.semilogy(outlier_factors, huber_vals_outlier, '^-', label='Huber', linewidth=2, markersize=8)
ax4.set_xlabel('Outlier Magnitude Factor', fontsize=11)
ax4.set_ylabel('Loss (log scale)', fontsize=11)
ax4.set_title('Robustness to Outliers', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, which='both')

# 5. Categorical Cross-Entropy Heatmap
ax5 = axes[1, 1]
pred_confidence = np.linspace(0.1, 0.9, 20)
cce_correct = -np.log(pred_confidence)  # Correct class
cce_wrong = -np.log(1 - pred_confidence)  # Wrong class

x = np.arange(len(pred_confidence))
width = 0.35

ax5.bar(x - width/2, cce_correct, width, label='Correct Class', color='green', alpha=0.7)
ax5.bar(x + width/2, cce_wrong, width, label='Wrong Class', color='red', alpha=0.7)
ax5.set_xlabel('Predicted Probability', fontsize=11)
ax5.set_ylabel('Loss', fontsize=11)
ax5.set_title('Cross-Entropy: Correct vs Wrong', fontsize=12, fontweight='bold')
ax5.set_xticks(x[::4])
ax5.set_xticklabels([f'{p:.1f}' for p in pred_confidence[::4]], fontsize=9)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, axis='y')

# 6. Focal Loss Effect (Gamma parameter)
ax6 = axes[1, 2]
pt = np.linspace(0.1, 1, 100)
gammas = [0, 0.5, 1, 2, 5]
colors_focal = plt.cm.viridis(np.linspace(0, 1, len(gammas)))

for gamma, color in zip(gammas, colors_focal):
    focal_weights = (1 - pt) ** gamma
    ax6.plot(pt, focal_weights, label=f'γ={gamma}', linewidth=2.5, color=color)

ax6.set_xlabel('Probability of True Class (pₜ)', fontsize=11)
ax6.set_ylabel('Focal Weight (1-pₜ)^γ', fontsize=11)
ax6.set_title('Focal Loss: Gamma Parameter Effect', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/cost_functions_plots.png', dpi=300, bbox_inches='tight')
print("✓ Plots saved as 'cost_functions_plots.png'")

# ============================================================================
# PART 5: PRACTICAL RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: PRACTICAL RECOMMENDATIONS")
print("=" * 80)

print("""
REGRESSION PROBLEMS:
  1. Start with MSE (most common)
     - Good for well-behaved data
     - Easy to interpret
  
  2. Use MAE if you have outliers
     - More robust
     - Linear penalty for all errors
  
  3. Use Huber Loss for safety
     - Balanced approach
     - Robust to outliers
     - Good gradients
  
  4. Use Log-Cosh for deep learning
     - Smooth function
     - Good numerical properties
     - Implicit regularization

BINARY CLASSIFICATION:
  1. Always use Binary Cross-Entropy
     - Standard choice
     - Works with sigmoid activation
  
  2. Use Focal Loss for imbalanced data
     - Down-weights easy examples
     - Focuses on hard examples

MULTI-CLASS CLASSIFICATION:
  1. Use Categorical Cross-Entropy
     - Works with softmax activation
     - Standard choice
  
  2. Use Sparse Categorical Cross-Entropy
     - If labels are integers
     - Saves memory
  
  3. Use Focal Loss for imbalance
     - Handles class imbalance

SPECIAL CASES:
  - SVM: Hinge Loss
  - Face Recognition: Triplet Loss
  - GANs: Wasserstein Loss
  - Metric Learning: Contrastive Loss
  - Uncertainty: Quantile Loss
""")

print("=" * 80)
print("Analysis Complete!")
print("=" * 80)
