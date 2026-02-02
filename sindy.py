"""
SINDy on UDE-exported data — narrow state-space robust version.

The UDE-smoothed trajectories explore a small region of state space,
making standard SINDy ill-conditioned. This script uses strategies
that work within that constraint.

Expected ground truth (missing interaction terms):
    f1(x, y) = -β * x * y   (β = 2.239)
    f2(x, y) = +γ * x * y   (γ = 2.57)

Requirements:
    pip install pysindy pandas numpy matplotlib scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

import pysindy as ps
from pysindy.optimizers import STLSQ

# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────

df = pd.read_csv("ude_data_sindy_clean.csv")

t = df["t"].values
X = df[["x_hat", "y_hat"]].values
F_hat = df[["f1_hat", "f2_hat"]].values
F_true = df[["f1_true", "f2_true"]].values

print(f"Loaded {len(t)} points, t ∈ [{t[0]:.2f}, {t[-1]:.2f}]")
print(f"State ranges: x ∈ [{X[:,0].min():.4f}, {X[:,0].max():.4f}], "
      f"y ∈ [{X[:,1].min():.4f}, {X[:,1].max():.4f}]\n")


# ──────────────────────────────────────────────
# 2. Centered + normalized SINDy
#    Subtracting the mean decorrelates features
#    significantly when data lives in a small region
# ──────────────────────────────────────────────

print("=" * 65)
print("STRATEGY 1: Centered + normalized features (degree 2)")
print("=" * 65)

x_mean = X.mean(axis=0)
X_centered = X - x_mean
print(f"Centering at x̄={x_mean[0]:.4f}, ȳ={x_mean[1]:.4f}")

lib_c = ps.PolynomialLibrary(degree=2, include_bias=True)
lib_c.fit(X_centered)
Theta_c = lib_c.transform(X_centered)
names_c = lib_c.get_feature_names(input_features=["dx", "dy"])

col_norms = np.linalg.norm(Theta_c, axis=0, keepdims=True)
col_norms[col_norms < 1e-12] = 1.0
Theta_c_norm = Theta_c / col_norms

cond_raw = np.linalg.cond(lib_c.transform(X))
cond_c = np.linalg.cond(Theta_c_norm)
print(f"Condition number (raw): {cond_raw:.1f}")
print(f"Condition number (centered+normalized): {cond_c:.1f}")

for thr in [0.01, 0.05, 0.1, 0.5]:
    opt = STLSQ(threshold=thr, alpha=1e-5, max_iter=200)
    try:
        opt.fit(Theta_c_norm, y=F_hat)
        Xi_norm = opt.coef_
        Xi_orig = Xi_norm / col_norms
        n_terms = np.sum(np.abs(Xi_norm) > 1e-6)
        pred = Theta_c_norm @ Xi_norm.T
        mse = np.mean((F_hat - pred) ** 2)
        print(f"\n  threshold={thr}, {n_terms} terms, MSE={mse:.2e}")
        for i, label in enumerate(["f1", "f2"]):
            terms = [f"{c:+.4f}·{n}" for c, n in zip(Xi_orig[i], names_c) if abs(c) > 1e-6]
            print(f"    {label} = {' '.join(terms) if terms else '0'}")
    except Exception as e:
        print(f"  threshold={thr}: {e}")


# ──────────────────────────────────────────────
# 3. Ensemble SINDy (bootstrap)
#    Subsample many times, keep only robust terms
# ──────────────────────────────────────────────

print("\n" + "=" * 65)
print("STRATEGY 2: Ensemble SINDy (100 bootstrap samples, degree 2)")
print("=" * 65)

lib_d2 = ps.PolynomialLibrary(degree=2, include_bias=True)
lib_d2.fit(X)
Theta_d2 = lib_d2.transform(X)
names_d2 = lib_d2.get_feature_names(input_features=["x", "y"])

n_ensembles = 100
subsample_fraction = 0.7
threshold = 0.1

n_samples = Theta_d2.shape[0]
n_features = Theta_d2.shape[1]
coef_accumulator = np.zeros((2, n_features))
inclusion_count = np.zeros((2, n_features))

rng = np.random.default_rng(42)

for _ in range(n_ensembles):
    idx = rng.choice(n_samples, size=int(n_samples * subsample_fraction), replace=False)
    opt = STLSQ(threshold=threshold, alpha=1e-5, max_iter=200)
    try:
        opt.fit(Theta_d2[idx], y=F_hat[idx])
        Xi = opt.coef_
        mask = np.abs(Xi) > 1e-6
        coef_accumulator += Xi * mask
        inclusion_count += mask
    except:
        continue

with np.errstate(divide="ignore", invalid="ignore"):
    Xi_ensemble = np.where(inclusion_count > 0, coef_accumulator / inclusion_count, 0)

inclusion_freq = inclusion_count / n_ensembles
Xi_ensemble[inclusion_freq < 0.5] = 0

print(f"Terms included in >50% of {n_ensembles} ensembles:")
for i, label in enumerate(["f1", "f2"]):
    terms = []
    for j, name in enumerate(names_d2):
        if abs(Xi_ensemble[i, j]) > 1e-6:
            freq = inclusion_freq[i, j]
            terms.append(f"{Xi_ensemble[i,j]:+.4f}·{name} ({freq:.0%})")
    print(f"  {label} = {' '.join(terms) if terms else '0'}")

pred_ens = Theta_d2 @ Xi_ensemble.T
print(f"  MSE: {np.mean((F_hat - pred_ens) ** 2):.2e}")


# ──────────────────────────────────────────────
# 4. Targeted regression on candidate terms
#    Exhaustive model selection on a small,
#    physically motivated library
# ──────────────────────────────────────────────

print("\n" + "=" * 65)
print("STRATEGY 3: Targeted regression — exhaustive model selection")
print("=" * 65)

candidates = {
    "1":     np.ones(len(t)),
    "x":     X[:, 0],
    "y":     X[:, 1],
    "x*y":   X[:, 0] * X[:, 1],
    "x^2":   X[:, 0] ** 2,
    "y^2":   X[:, 1] ** 2,
}

cand_names = list(candidates.keys())
Theta_minimal = np.column_stack(list(candidates.values()))

# --- Single-term models ---
print(f"\n{'Term':>10} | {'Coef f1':>14} | {'Coef f2':>14} | {'MSE':>10}")
print("-" * 60)

results = []
for i, name in enumerate(cand_names):
    A = Theta_minimal[:, [i]]
    c1, _, _, _ = np.linalg.lstsq(A, F_hat[:, 0], rcond=None)
    c2, _, _, _ = np.linalg.lstsq(A, F_hat[:, 1], rcond=None)
    pred = np.column_stack([A @ c1, A @ c2])
    mse = np.mean((F_hat - pred) ** 2)
    results.append((name, c1, c2, mse))
    print(f"{name:>10} | {c1[0]:>+14.6f} | {c2[0]:>+14.6f} | {mse:>10.2e}")

# --- Two-term models (top 5) ---
two_term = []
for (i, n1), (j, n2) in combinations_with_replacement(enumerate(cand_names), 2):
    if i == j:
        continue
    A = Theta_minimal[:, [i, j]]
    c1, _, _, _ = np.linalg.lstsq(A, F_hat[:, 0], rcond=None)
    c2, _, _, _ = np.linalg.lstsq(A, F_hat[:, 1], rcond=None)
    pred = np.column_stack([A @ c1, A @ c2])
    mse = np.mean((F_hat - pred) ** 2)
    two_term.append((f"{n1}, {n2}", c1, c2, mse))

two_term.sort(key=lambda x: x[3])

print(f"\nTop 5 two-term models:")
for name, c1, c2, mse in two_term[:5]:
    c1_str = ", ".join(f"{v:+.4f}" for v in c1)
    c2_str = ", ".join(f"{v:+.4f}" for v in c2)
    print(f"  {name:>15} | f1=[{c1_str}] | f2=[{c2_str}] | MSE={mse:.2e}")

# Highlight x*y
print("\n" + "─" * 65)
xy_result = [r for r in results if r[0] == "x*y"][0]
print(f"★  SINGLE-TERM x·y MODEL:")
print(f"   f1 = {xy_result[1][0]:+.6f} · x·y   (true: -2.239000)")
print(f"   f2 = {xy_result[2][0]:+.6f} · x·y   (true: +2.570000)")
print(f"   MSE = {xy_result[3]:.2e}")
print("─" * 65)


# ──────────────────────────────────────────────
# 5. Plots
# ──────────────────────────────────────────────

# Reconstruct using x*y single-term model
xy_idx = cand_names.index("x*y")
A_xy = Theta_minimal[:, [xy_idx]]
c1_xy, _, _, _ = np.linalg.lstsq(A_xy, F_hat[:, 0], rcond=None)
c2_xy, _, _, _ = np.linalg.lstsq(A_xy, F_hat[:, 1], rcond=None)
F_sindy_xy = np.column_stack([A_xy * c1_xy, A_xy * c2_xy])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Trajectory
axes[0, 0].plot(t, X[:, 0], "b-", label="x (prey)")
axes[0, 0].plot(t, X[:, 1], "r-", label="y (predator)")
axes[0, 0].set_xlabel("t")
axes[0, 0].set_ylabel("State")
axes[0, 0].set_title("UDE-Smoothed Trajectories")
axes[0, 0].legend()

# (b) f1
axes[0, 1].plot(t, F_true[:, 0], "k-", lw=2, label="True f₁")
axes[0, 1].plot(t, F_hat[:, 0], "r--", alpha=0.7, label="NN f₁")
axes[0, 1].plot(t, F_sindy_xy[:, 0], "b:", lw=2, label=f"SINDy: {c1_xy[0]:+.4f}·xy")
axes[0, 1].set_xlabel("t")
axes[0, 1].set_title("Missing Term f₁ (prey interaction)")
axes[0, 1].legend()

# (c) f2
axes[1, 0].plot(t, F_true[:, 1], "k-", lw=2, label="True f₂")
axes[1, 0].plot(t, F_hat[:, 1], "r--", alpha=0.7, label="NN f₂")
axes[1, 0].plot(t, F_sindy_xy[:, 1], "b:", lw=2, label=f"SINDy: {c2_xy[0]:+.4f}·xy")
axes[1, 0].set_xlabel("t")
axes[1, 0].set_title("Missing Term f₂ (predator interaction)")
axes[1, 0].legend()

# (d) Single-term MSE comparison
single_names_list = [r[0] for r in results]
single_mses = [r[3] for r in results]
colors = ["green" if n == "x*y" else "gray" for n in single_names_list]
axes[1, 1].bar(single_names_list, single_mses, color=colors)
axes[1, 1].set_ylabel("MSE")
axes[1, 1].set_title("Single-Term Model Comparison")
axes[1, 1].set_yscale("log")

plt.tight_layout()
plt.savefig("sindy_results_improved.png", dpi=150)
plt.show()
print("\nSaved sindy_results_improved.png")
