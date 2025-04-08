# **Kernel-based estimators for Functional Causal Effects**

This repository implements **kernel ridge regression**‐based estimators for **causal effects** when outcomes and/or covariates are **functional** (i.e., observations in an $L_2$ space over a $T$-dimensional grid). Our motivating application is **estimating causal effects** (e.g., average treatment effects, dose-response curves, heterogeneous effects) when the outcome variable is a **time series** or **curve**.

In particular, the **operator-valued kernel** approach (proposed in our manuscript [arXiv:2503.05024](https://arxiv.org/abs/2503.05024)) allows the outcome itself to be an entire **function**. Meanwhile, for **scalar** outcomes, we leverage existing **kernel ridge regression** techniques (Singh, Xu, & Gretton, 2024) extended to causal inference. We also include an **elastic registration** module (using SRVF transforms) to optionally align functional data prior to estimation. The methods and theoretical guarantees are described in detail in [arXiv:2503.05024](https://arxiv.org/abs/2503.05024).

---

## **Features**

- **Kernel Ridge Regression** for causal effect estimation on **functional** or **scalar** outcomes.
- **Operator-valued kernels** to model outcome curves as elements in function spaces.
- **Binary or continuous treatment** options:
  - Set `treatment="discrete"` for binary treatment
  - Otherwise uses an RBF kernel to model continuous (dose) response
- **Elastic registration** using Square Root Velocity Functions (SRVF)
- **Confidence intervals** and **significance testing** under mild asymptotic assumptions
- **Digital monitoring** data from the PD@Home validation study has been included and made publically available with this work in the folder ```data - PDHome```

---

## **Supported Methods and References**

- **Operator-valued kernel for functional outcomes:**
  - Proposed in our manuscript: [arXiv:2503.05024](https://arxiv.org/abs/2503.05024)
- **Scalar-output kernel ridge regression:**
  - Singh, R., Xu, L., & Gretton, A. (2024). *Kernel methods for causal functions: dose, heterogeneous and incremental response curves.* Biometrika, 111(2), 497–516.
- **Inverse Probability Weighting (IPW):**
  - Imai, K., & Van Dyk, D. (2004). *Causal inference with general treatment regimes.* JASA.
- **Doubly Robust Estimators:**
  - Funk, M.J., et al. (2011). *Doubly robust estimation of causal effects.* American Journal of Epidemiology, 173(7).

---

## **Installation**

Clone the repository and install the dependencies:

```bash
git clone https://github.com/JordanRaykov/Kernel-based estimators for Functional Causal Effects.git
cd Kernel-based estimators for Functional Causal Effects

```
Further install ```fdasrsf```
via 
```bash
pip install fdasrsf
```
## **Quick Start Example**

We provide a demo comparing operator-valued kernel estimators **with and without** SRVF registration. Below is a minimal working example:

```python

import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split

from utils import (
    generate_synthetic_curves,
    compute_median_interpoint_distance,
    evaluate_func_kernel_causal_effects,
)
from KernelRidgeRegressionCausalEstimator import KernelRidgeRegressionCausalEstimator

# ---------------------------------------------------------
# 1) Generate a single simulated dataset (n=50) 
#    (binary treatment, functional outcome)
# ---------------------------------------------------------
n_samples = 50
n_timepoints = 48
frac_train = 0.80

Y, D, V, X, theta_1_true = generate_synthetic_curves(n_samples, n_timepoints)  
# shape(Y) = (n_samples, n_timepoints)

# Flatten the treatment and V for easier splitting
D = D.ravel() 
V = V.ravel()

# Train/test split
X_train, X_test, Y_train, Y_test, D_train, D_test, V_train, V_test, theta_1_true_train, theta_1_true_test = \
    train_test_split(X, Y, D, V, theta_1_true, test_size=(1-frac_train), random_state=42)

# ---------------------------------------------------------
# 2) Compute kernel parameter (sigma) from training features
# ---------------------------------------------------------
kernel_sigma = compute_median_interpoint_distance(X_train)
print("Median-based kernel_sigma:", kernel_sigma)

# ---------------------------------------------------------
# 3) Hyperparameter search for operator-valued kernel 
#    WITHOUT registration
# ---------------------------------------------------------
lambd1_vals = np.logspace(-3, 3, 3)
lambd2_vals = np.logspace(-3, 3, 3)
lambd3_vals = np.logspace(-3, 3, 3)
hyperparameter_grid = list(product(lambd1_vals, lambd2_vals, lambd3_vals))

best_score_no_reg = float('inf')
best_params_no_reg = None

for lam1, lam2, lam3 in hyperparameter_grid:
    score = evaluate_func_kernel_causal_effects(
        X_train, V_train.reshape(-1,1), D_train.reshape(-1,1), Y_train,
        X_test,  V_test.reshape(-1,1),  D_test.reshape(-1,1),  Y_test,
        lam1, lam2, lam3, kernel_sigma,
        apply_srfv_Y=False,  # <--- No SRVF registration
        srfv_Y_groups=None, 
        apply_srfv_X=False
    )
    if score < best_score_no_reg:
        best_score_no_reg = score
        best_params_no_reg = (lam1, lam2, lam3)

print(f"\n[No Registration] Best hyperparams: lambda1={best_params_no_reg[0]}, "
      f"lambda2={best_params_no_reg[1]}, lambda3={best_params_no_reg[2]}, "
      f"MSE={best_score_no_reg:.4f}")

# Fit final operator-valued estimator (no registration) on the entire training set
kernel_causal_estimator_no_reg = KernelRidgeRegressionCausalEstimator(
    lambd1=best_params_no_reg[0], 
    lambd2=best_params_no_reg[1], 
    lambd3=best_params_no_reg[2],
    kernel_sigma=kernel_sigma, 
    treatment='discrete',
    use_operator_valued_kernel=True,
    apply_srfv_Y=False,
    srfv_Y_groups=None,
    apply_srfv_X=False
)

kernel_causal_estimator_no_reg.fit(X_train, V_train.reshape(-1,1), D_train.reshape(-1,1), Y_train)
# For the test set, we get predicted ATE for each subject (i.e., E[Y(1)-Y(0)|X_test])
predictions_ate_no_reg, predictions_cate_no_reg = kernel_causal_estimator_no_reg.predict(
    X_test, V_test.reshape(-1,1), D_test.reshape(-1,1)
)

# Evaluate average integrated error
# (comparing predicted Y(1) - Y(0) vs true theta_1)
test_truth = theta_1_true_test  # shape: (n_test, n_timepoints)
est_diff_no_reg = predictions_cate_no_reg[:,1,:] - predictions_cate_no_reg[:,0,:]
mse_no_reg = np.mean((est_diff_no_reg - test_truth)**2)
print(f"Operator-valued Kernel WITHOUT SRVF registration, MSE on test set: {mse_no_reg:.4f}")

# ---------------------------------------------------------
# 4) Hyperparameter search for operator-valued kernel 
#    WITH SRVF registration
# ---------------------------------------------------------
best_score_srvf = float('inf')
best_params_srvf = None

for lam1, lam2, lam3 in hyperparameter_grid:
    score = evaluate_func_kernel_causal_effects(
        X_train, V_train.reshape(-1,1), D_train.reshape(-1,1), Y_train,
        X_test,  V_test.reshape(-1,1),  D_test.reshape(-1,1),  Y_test,
        lam1, lam2, lam3, kernel_sigma,
        apply_srfv_Y=True,   # <--- Turn on SRVF registration
        srfv_Y_groups=[1],  # (We have only one outcome group, so group=[1])
        apply_srfv_X=False
    )
    if score < best_score_srvf:
        best_score_srvf = score
        best_params_srvf = (lam1, lam2, lam3)

print(f"\n[With SRVF] Best hyperparams: lambda1={best_params_srvf[0]}, "
      f"lambda2={best_params_srvf[1]}, lambda3={best_params_srvf[2]}, "
      f"MSE={best_score_srvf:.4f}")

# Fit final operator-valued estimator (with registration) on the entire training set
kernel_causal_estimator_srvf = KernelRidgeRegressionCausalEstimator(
    lambd1=best_params_srvf[0], 
    lambd2=best_params_srvf[1], 
    lambd3=best_params_srvf[2],
    kernel_sigma=kernel_sigma, 
    treatment='discrete',
    use_operator_valued_kernel=True,
    apply_srfv_Y=True, 
    srfv_Y_groups=[1],
    apply_srfv_X=False
)

kernel_causal_estimator_srvf.fit(X_train, V_train.reshape(-1,1), D_train.reshape(-1,1), Y_train)
predictions_ate_srvf, predictions_cate_srvf = kernel_causal_estimator_srvf.predict(
    X_test, V_test.reshape(-1,1), D_test.reshape(-1,1)
)

# Evaluate average integrated error for the SRVF approach
est_diff_srvf = predictions_cate_srvf[:,1,:] - predictions_cate_srvf[:,0,:]
mse_srvf = np.mean((est_diff_srvf - test_truth)**2)
print(f"Operator-valued Kernel WITH SRVF registration, MSE on test set: {mse_srvf:.4f}")
```
## Usage Notes

### Binary Treatments
The default option (`treatment="discrete"`) estimates average and conditional effects for treated vs. untreated groups.

### Continuous Treatments
When `treatment != "discrete"`, the estimator uses an RBF kernel over treatment levels to estimate dose-response functions.

### Elastic Registration
To reduce phase variability, set `apply_srfv_Y=True` (or `apply_srfv_X=True`) to apply SRVF alignment on outcomes or covariates, respectively.

## **Confidence intervals**
We use built-in utilities (e.g., `delta_method_ci`, `test_zero_effect`) to construct asymptotic normal or chi-squared based intervals for  
$\|\Delta\|_2$.
```python

# ---------------------------------------------------------
# 5) Confidence intervals and significance of the effects
# ---------------------------------------------------------
from utils import compute_estimated_effects, estimate_covariance, delta_method_ci, test_zero_effect

# 1) Compute estimated effect (subject-level and average)
Delta_hat_i, Delta_hat = compute_estimated_effects(predictions_cate_srvf)

# 2) Estimate covariance of the subject-level effect
S = estimate_covariance(Delta_hat_i)
n_test = Delta_hat_i.shape[0]

# 3) Build a delta-method CI for ||Delta||^2 (assuming the true norm is not zero):
ci_lower, ci_upper = delta_method_ci(Delta_hat, S, n_test, alpha=0.05)
print(f"Estimated ||Delta||^2 = {np.sum(Delta_hat**2):.4f}")
print(f"95% CI (non-zero effect assumption) = [{ci_lower:.4f}, {ci_upper:.4f}]")

# 4) (Optional) Test for zero effect vs. positive effect
test_stat, chi2_crit, p_value, reject = test_zero_effect(Delta_hat, S, n_test, alpha=0.05)
print(f"Test statistic (naive chi^2) = {test_stat:.4f}")
print(f"Chi^2 critical @ 95%        = {chi2_crit:.4f}")
print(f"p-value                      = {p_value:.4f}")
print(f"Reject H0 (zero effect)?     = {reject}")

# ---------------------
# Plotting
# ---------------------

n_test, T = predictions_ate_srvf.shape
time_grid = np.arange(T)

# 1) Mean ATE across subjects at each timepoint
mean_ate = np.mean(predictions_ate_srvf, axis=0)  # shape (T,)

# 2) Standard error (assuming independence across test subjects)
std_ate = np.std(predictions_ate_srvf, axis=0, ddof=1)  # sample stdev across i
sem_ate = std_ate / np.sqrt(n_test)                     # standard error

# 3) For a 95% confidence interval, use 1.96 as the normal critical value
z_val = 1.96
ci_lower = mean_ate - z_val * sem_ate
ci_upper = mean_ate + z_val * sem_ate

import matplotlib.pyplot as plt

plt.plot(time_grid, mean_ate, label="Mean SRVF ATE")
plt.fill_between(time_grid, ci_lower, ci_upper, alpha=0.2, label="95% CI")

plt.xlabel("Time index")
plt.ylabel("Predicted ATE")
plt.title("SRVF Operator‐Valued Kernel – ATE over Time")
plt.legend()
plt.show()
```

## Citing This Work

If you use this package for academic research, please cite:

### Operator-Valued Kernel Estimators
*Kernel-based estimators for functional causal effects*. arXiv:2503.05024
### Parkinson@Home Validation study
Evers, L.J., Raykov, Y.P., Krijthe, J.H., Silva de Lima, A.L., Badawy, R., Claes, K., Heskes, T.M., Little, M.A., Meinders, M.J. and Bloem, B.R. (2020). *Real-life gait performance as a digital biomarker for motor fluctuations: the Parkinson@ Home validation study*. **Journal of medical Internet research**, 22(10), p.e19068.

### Scalar Kernel Ridge Regression for Causal Effects
Singh, R., Xu, L., & Gretton, A. (2024). *Kernel methods for causal functions: dose, heterogeneous and incremental response curves*. **Biometrika**, 111(2), 497–516.

### Classical Methods
Imai, K., & Van Dyk, D. (2004). *Causal inference with general treatment regimes*. **Journal of the American Statistical Association**.

Funk, M. J., et al. (2011). *Doubly robust estimation of causal effects*. **American Journal of Epidemiology**, 173(7).
