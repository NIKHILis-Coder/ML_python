"""
GRADIENT DESCENT METHODS: BATCH vs SGD vs MINI-BATCH
======================================================

Three approaches to update model weights during training.
Learn when to use each, how to implement, and performance implications.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

# ============================================================================
# SECTION 1: UNDERSTANDING THE THREE APPROACHES
# ============================================================================

"""
BATCH GRADIENT DESCENT (BGD)
----------------------------
✓ Updates weights ONCE per epoch using ENTIRE dataset
✓ All training samples processed → single gradient calculated → weights updated
✓ Deterministic, stable convergence path
✗ Slow for large datasets (must load all data in memory)
✗ May get stuck in local minima (no noise to escape)

STOCHASTIC GRADIENT DESCENT (SGD)
----------------------------------
✓ Updates weights AFTER EACH sample (or few samples)
✓ Very fast iteration, can escape local minima with noisy updates
✓ Good for online learning / streaming data
✗ Noisy convergence - overshoots optimal point
✗ Requires tuning learning rate (more sensitive)

MINI-BATCH GRADIENT DESCENT (MBGD)
-----------------------------------
✓ Best of both worlds: updates using BATCH of samples (e.g., 32, 64, 128)
✓ Stable + Fast: less noisy than SGD, faster than BGD
✓ Parallelizable (multiple batches processed simultaneously)
✓ Default in modern deep learning (PyTorch, TensorFlow)
✗ Need to tune batch size (another hyperparameter)

MATHEMATICAL DIFFERENCE:
------------------------
Gradient computation per epoch:

BGD:    gradient = (1/n) * Σ(i=1 to n) ∇L(y_i, ŷ_i)
        weights = weights - learning_rate * gradient
        (ONE update per epoch, uses all n samples)

SGD:    for each sample i:
          gradient = ∇L(y_i, ŷ_i)
          weights = weights - learning_rate * gradient
        (n updates per epoch, one sample each)

MBGD:   for each batch of size b:
          gradient = (1/b) * Σ(i in batch) ∇L(y_i, ŷ_i)
          weights = weights - learning_rate * gradient
        (n/b updates per epoch, b samples each)
"""

# ============================================================================
# SECTION 2: SCIKIT-LEARN IMPLEMENTATIONS
# ============================================================================

print("=" * 80)
print("GRADIENT DESCENT IMPLEMENTATIONS WITH SCIKIT-LEARN")
print("=" * 80)

# Generate synthetic dataset for demonstration
np.random.seed(42)
X, y = make_regression(n_samples=10000, n_features=20, noise=50, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (IMPORTANT for gradient descent!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------
# 1. BATCH GRADIENT DESCENT (using sklearn)
# -------------------------------------------------------
print("\n1️⃣  BATCH GRADIENT DESCENT")
print("-" * 80)

"""
In scikit-learn:
- Use SGDRegressor with warm_start=True
- Set partial_fit() to manually iterate through full batches
- Or use LinearRegression (closed-form solution, equivalent to BGD optimal)
"""

# Approach A: Using LinearRegression (solves in one go, equivalent to BGD)
start_time = time.time()
model_bgd_lr = LinearRegression()
model_bgd_lr.fit(X_train_scaled, y_train)
bgd_lr_time = time.time() - start_time

y_pred_bgd = model_bgd_lr.predict(X_test_scaled)
bgd_mse = mean_squared_error(y_test, y_pred_bgd)
bgd_r2 = r2_score(y_test, y_pred_bgd)

print(f"Method: LinearRegression (closed-form solution)")
print(f"Training time: {bgd_lr_time:.4f}s")
print(f"Test MSE: {bgd_mse:.2f}")
print(f"Test R²: {bgd_r2:.4f}")

# Approach B: Using SGDRegressor with manual epoch control (pedagogical)
print(f"\nMethod: SGDRegressor (simulating BGD with warm_start)")
model_bgd_sgd = SGDRegressor(
    loss='squared_error',           # L2 loss
    learning_rate='constant',       # Fixed learning rate
    eta0=0.01,                      # Initial learning rate
    max_iter=1,                     # 1 iteration per fit call
    warm_start=True,                # Keep weights between fits
    random_state=42,
    verbose=0
)

start_time = time.time()
# Manually iterate through entire dataset multiple times (simulating epochs)
n_epochs = 100
mse_history_bgd = []

for epoch in range(n_epochs):
    # Shuffle data each epoch (good practice)
    indices = np.random.permutation(len(X_train_scaled))
    X_shuffled = X_train_scaled[indices]
    y_shuffled = y_train[indices]
    
    # Update with entire dataset
    model_bgd_sgd.partial_fit(X_shuffled, y_shuffled)
    
    # Track MSE on training set
    y_pred_train = model_bgd_sgd.predict(X_train_scaled)
    mse = mean_squared_error(y_train, y_pred_train)
    mse_history_bgd.append(mse)
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}: Train MSE = {mse:.2f}")

bgd_sgd_time = time.time() - start_time
y_pred_bgd_sgd = model_bgd_sgd.predict(X_test_scaled)
bgd_sgd_mse = mean_squared_error(y_test, y_pred_bgd_sgd)
bgd_sgd_r2 = r2_score(y_test, y_pred_bgd_sgd)

print(f"Training time: {bgd_sgd_time:.4f}s")
print(f"Test MSE: {bgd_sgd_mse:.2f}")
print(f"Test R²: {bgd_sgd_r2:.4f}")

# -------------------------------------------------------
# 2. STOCHASTIC GRADIENT DESCENT (SGD)
# -------------------------------------------------------
print(f"\n2️⃣  STOCHASTIC GRADIENT DESCENT")
print("-" * 80)

"""
In scikit-learn:
- SGDRegressor with n_iter_no_change OR max_iter
- Each call to fit() or partial_fit() updates weights
- By default, processes ONE sample per gradient step internally
  (unless you batch them, which contradicts SGD definition)
"""

model_sgd = SGDRegressor(
    loss='squared_error',
    learning_rate='constant',
    eta0=0.01,
    max_iter=100,                   # Total epochs
    random_state=42,
    verbose=0
)

start_time = time.time()
model_sgd.fit(X_train_scaled, y_train)
sgd_time = time.time() - start_time

y_pred_sgd = model_sgd.predict(X_test_scaled)
sgd_mse = mean_squared_error(y_test, y_pred_sgd)
sgd_r2 = r2_score(y_test, y_pred_sgd)

print(f"Method: SGDRegressor (built-in SGD)")
print(f"Training time: {sgd_time:.4f}s")
print(f"Test MSE: {sgd_mse:.2f}")
print(f"Test R²: {sgd_r2:.4f}")

# -------------------------------------------------------
# 3. MINI-BATCH GRADIENT DESCENT (MBGD)
# -------------------------------------------------------
print(f"\n3️⃣  MINI-BATCH GRADIENT DESCENT")
print("-" * 80)

"""
In scikit-learn:
- Use SGDRegressor
- Manually control batching with partial_fit(batch_size)
- Process dataset in chunks, updating weights per batch
"""

batch_size = 32  # Common choice
model_mbgd = SGDRegressor(
    loss='squared_error',
    learning_rate='constant',
    eta0=0.01,
    warm_start=True,
    max_iter=1,                     # 1 epoch per fit call
    random_state=42,
    verbose=0
)

start_time = time.time()
n_epochs = 100
mse_history_mbgd = []

for epoch in range(n_epochs):
    # Shuffle data
    indices = np.random.permutation(len(X_train_scaled))
    X_shuffled = X_train_scaled[indices]
    y_shuffled = y_train[indices]
    
    # Process in mini-batches
    n_batches = len(X_train_scaled) // batch_size
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        
        model_mbgd.partial_fit(X_batch, y_batch)
    
    # Track MSE
    y_pred_train = model_mbgd.predict(X_train_scaled)
    mse = mean_squared_error(y_train, y_pred_train)
    mse_history_mbgd.append(mse)
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}: Train MSE = {mse:.2f}")

mbgd_time = time.time() - start_time
y_pred_mbgd = model_mbgd.predict(X_test_scaled)
mbgd_mse = mean_squared_error(y_test, y_pred_mbgd)
mbgd_r2 = r2_score(y_test, y_pred_mbgd)

print(f"Training time: {mbgd_time:.4f}s")
print(f"Test MSE: {mbgd_mse:.2f}")
print(f"Test R²: {mbgd_r2:.4f}")

# ============================================================================
# SECTION 3: SIDE-BY-SIDE COMPARISON
# ============================================================================

print(f"\n{'='*80}")
print("COMPARISON SUMMARY")
print(f"{'='*80}")

comparison_data = {
    'Method': ['Batch GD (LR)', 'Batch GD (SGD)', 'SGD', 'Mini-Batch GD'],
    'Training Time (s)': [f"{bgd_lr_time:.4f}", f"{bgd_sgd_time:.4f}", f"{sgd_time:.4f}", f"{mbgd_time:.4f}"],
    'Test MSE': [f"{bgd_mse:.2f}", f"{bgd_sgd_mse:.2f}", f"{sgd_mse:.2f}", f"{mbgd_mse:.2f}"],
    'Test R²': [f"{bgd_r2:.4f}", f"{bgd_sgd_r2:.4f}", f"{sgd_r2:.4f}", f"{mbgd_r2:.4f}"]
}

df_comparison = pd.DataFrame(comparison_data)
print(f"\n{df_comparison.to_string(index=False)}")

# ============================================================================
# SECTION 4: WHEN TO USE WHICH
# ============================================================================

print(f"\n{'='*80}")
print("WHEN TO USE WHICH?")
print(f"{'='*80}")

when_to_use = """
✅ USE BATCH GRADIENT DESCENT (BGD):
   • Small to medium datasets (< 100K samples)
   • Guaranteed smooth convergence path needed
   • Offline learning (all data available upfront)
   • When memory is not a constraint
   → LinearRegression, closed-form solutions

✅ USE STOCHASTIC GRADIENT DESCENT (SGD):
   • HUGE datasets (millions of samples)
   • Online/streaming learning
   • Each sample costly to process (so frequent updates help)
   • Need to escape local minima (noise helps)
   → Online learning systems, real-time data streams

✅ USE MINI-BATCH GRADIENT DESCENT (MBGD) ⭐ MOST COMMON:
   • Default choice for deep learning (PyTorch, TensorFlow)
   • Medium to large datasets
   • GPU/parallel processing needed (batches parallelizable)
   • Good balance: stable + fast
   • Typical batch sizes: 32, 64, 128, 256
   → SGDRegressor with batching, neural networks

LEARNING RATE TUNING:
   • BGD: Can use higher learning rates (stable)
   • SGD: Need smaller learning rates (noisy updates can diverge)
   • MBGD: Middle ground, tune carefully

CONVERGENCE BEHAVIOR:
   • BGD: Smooth, monotonic decrease
   • SGD: Noisy, zigzagging but adaptive
   • MBGD: Balance - relatively smooth, good convergence
"""

print(when_to_use)

# ============================================================================
# SECTION 5: CONVERGENCE VISUALIZATION
# ============================================================================

print(f"\n{'='*80}")
print("VISUALIZING CONVERGENCE...")
print(f"{'='*80}")

plt.figure(figsize=(12, 6))

# Plot convergence curves (normalized epochs for fair comparison)
epochs_range = range(1, len(mse_history_bgd) + 1)
plt.plot(epochs_range, mse_history_bgd, label='Batch GD', linewidth=2, alpha=0.8)
plt.plot(epochs_range, mse_history_mbgd, label='Mini-Batch GD (batch_size=32)', 
         linewidth=2, alpha=0.8, linestyle='--')

# Plot SGD trend (noisier, plot moving average)
sgd_history = []
model_sgd_track = SGDRegressor(
    loss='squared_error', learning_rate='constant', eta0=0.01,
    max_iter=1, warm_start=True, random_state=42
)
for epoch in range(100):
    indices = np.random.permutation(len(X_train_scaled))
    model_sgd_track.partial_fit(X_train_scaled[indices], y_train[indices])
    y_pred = model_sgd_track.predict(X_train_scaled)
    sgd_history.append(mean_squared_error(y_train, y_pred))

plt.plot(epochs_range, sgd_history, label='SGD (noisy)', linewidth=1.5, 
         alpha=0.6, linestyle=':')

# Moving average for SGD (smoother visualization)
window = 5
sgd_ma = pd.Series(sgd_history).rolling(window=window).mean()
plt.plot(epochs_range, sgd_ma, label=f'SGD Moving Avg (window={window})', 
         linewidth=2, alpha=0.8, color='red')

plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Training MSE', fontsize=11)
plt.title('Convergence Behavior: Batch vs Mini-Batch vs SGD', fontsize=13, fontweight='bold')
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/gradient_descent_convergence.png', dpi=300)
print("✅ Saved convergence plot to: gradient_descent_convergence.png")

# ============================================================================
# SECTION 6: KEY TAKEAWAYS & CODE SNIPPETS
# ============================================================================

code_snippets = """
='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='
QUICK CODE REFERENCE
='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='

1️⃣  BATCH GRADIENT DESCENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)  # Solves once using all data
predictions = model.predict(X_test)

# OR manually with SGDRegressor:
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(warm_start=True, max_iter=1)
for epoch in range(n_epochs):
    model.partial_fit(X_train_shuffled, y_train_shuffled)


2️⃣  STOCHASTIC GRADIENT DESCENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(
    loss='squared_error',
    learning_rate='constant',
    eta0=0.01,
    max_iter=100
)
model.fit(X_train, y_train)


3️⃣  MINI-BATCH GRADIENT DESCENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from sklearn.linear_model import SGDRegressor

batch_size = 32
model = SGDRegressor(warm_start=True, max_iter=1)

for epoch in range(n_epochs):
    # Shuffle
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    # Process in batches
    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        model.partial_fit(X_batch, y_batch)

predictions = model.predict(X_test)


🔑 KEY HYPERPARAMETERS TO TUNE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• learning_rate (eta0): Start with 0.001 - 0.1, then reduce if diverging
• max_iter: Number of epochs (passes through data)
• warm_start: True for manual epoch control, False for automatic
• loss: 'squared_error' for regression, others for classification
• random_state: For reproducibility
• batch_size: For mini-batch (typical: 32, 64, 128)
• shuffle: Whether to shuffle data between epochs (default: True)


⚠️  CRITICAL TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ALWAYS standardize/normalize features before gradient descent:
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)

2. Use warm_start=True if you want to control epochs manually:
   model = SGDRegressor(warm_start=True, max_iter=1)
   for epoch in range(100):
       model.partial_fit(X, y)

3. Shuffle data between epochs (SGDRegressor does this by default)

4. Learning rate scheduling: Start high, decrease over time
   SGDRegressor has 'learning_rate' parameter for this

5. Mini-batch size depends on GPU/CPU:
   • GPU: 32-256 (memory allows parallel processing)
   • CPU: 16-64 (balance between vectorization & memory)

6. For convergence tracking, keep loss history:
   for epoch in range(n_epochs):
       model.partial_fit(X_batch, y_batch)
       loss = model.loss_
       losses.append(loss)
"""

print(code_snippets)

print("\n" + "="*80)
print("✅ FULL GUIDE COMPLETE - Check the Python file for runnable code!")
print("="*80)









## STOCHASTIC GRADIENT DESCENTTT
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(
    loss='squared_error',
    penalty='l2',
    alpha=0.0001,
    max_iter=1000,
    tol=1e-3,
    learning_rate='invscaling',
    eta0=0.01,
    power_t=0.25,
    shuffle=True,
    random_state=42,
    fit_intercept=True,
    early_stopping=False
)