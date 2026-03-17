import csv
import math

# Load data
x, y = [], []
with open('LR_armyTraining.csv') as f:
    for row in csv.reader(f):
        if len(row) == 2:
            try:
                x.append(float(row[0]))
                y.append(float(row[1]))
            except ValueError:
                pass

n = len(x)
print(f"Data: {n} points")
print(f"Height range: [{min(x)}, {max(x)}]")
print(f"Weight range: [{min(y)}, {max(y)}]")
print()

# Precompute sums for fast SSE(beta, alpha) = beta^2*Sxx + alpha^2*n + Syy + 2*beta*alpha*Sx - 2*beta*Sxy - 2*alpha*Sy
Sx  = sum(x)
Sy  = sum(y)
Sxx = sum(xi**2 for xi in x)
Syy = sum(yi**2 for yi in y)
Sxy = sum(x[i]*y[i] for i in range(n))

def sse(beta, alpha):
    return beta*beta*Sxx + alpha*alpha*n + Syy + 2*beta*alpha*Sx - 2*beta*Sxy - 2*alpha*Sy

def logspace(lo, hi, num):
    """Return num values log-spaced between lo and hi (both positive)."""
    log_lo = math.log10(lo)
    log_hi = math.log10(hi)
    step = (log_hi - log_lo) / (num - 1)
    return [10 ** (log_lo + i * step) for i in range(num)]

def search_grid(beta_lo, beta_hi, alpha_lo, alpha_hi, N):
    """Grid search over [beta_lo,beta_hi] x [alpha_lo,alpha_hi] using log-spaced offsets from center."""
    offsets = logspace(1e-6, 1, N // 2)

    beta_span = (beta_hi - beta_lo) / 2
    beta_c = (beta_hi + beta_lo) / 2
    beta_grid = sorted(set(
        [beta_c + v * beta_span for v in offsets] +
        [beta_c - v * beta_span for v in offsets]
    ))

    alpha_span = (alpha_hi - alpha_lo) / 2
    alpha_c = (alpha_hi + alpha_lo) / 2
    alpha_grid = sorted(set(
        [alpha_c + v * alpha_span for v in offsets] +
        [alpha_c - v * alpha_span for v in offsets]
    ))

    best_sse = float('inf')
    best_beta, best_alpha = beta_c, alpha_c

    for beta in beta_grid:
        for alpha in alpha_grid:
            s = sse(beta, alpha)
            if s < best_sse:
                best_sse = s
                best_beta, best_alpha = beta, alpha

    return best_beta, best_alpha, best_sse

N_GRID    = 200    # grid points per axis per zoom
TOL       = 5e-4   # stop when both beta and alpha change by less than 0.0005 (3 decimal places)
MAX_ZOOMS = 50     # hard cap to prevent infinite loops

# Wide initial range
beta_lo,  beta_hi  = -500.0,   500.0
alpha_lo, alpha_hi = -10000.0, 10000.0

print(f"{'Zoom':>5}  {'beta':>12}  {'alpha':>14}  {'SSE':>14}  {'d_beta':>10}  {'d_alpha':>10}")
print("-" * 75)

best_beta, best_alpha, best_sse_val = 0.0, 0.0, float('inf')
prev_beta, prev_alpha = None, None

for zoom in range(MAX_ZOOMS):
    best_beta, best_alpha, best_sse_val = search_grid(beta_lo, beta_hi, alpha_lo, alpha_hi, N_GRID)
    d_beta  = abs(best_beta  - prev_beta)  if prev_beta  is not None else float('inf')
    d_alpha = abs(best_alpha - prev_alpha) if prev_alpha is not None else float('inf')
    print(f"{zoom+1:>5}  {best_beta:>12.6f}  {best_alpha:>14.4f}  {best_sse_val:>14.4f}  {d_beta:>10.2e}  {d_alpha:>10.2e}")
    if d_beta < TOL and d_alpha < TOL:
        print(f"\n  Converged after {zoom+1} iterations (beta and alpha stable to 3 decimal places)")
        break
    prev_beta, prev_alpha = best_beta, best_alpha

    # Zoom in: center on best, shrink range by 10x
    beta_span  = (beta_hi  - beta_lo)  / 10
    alpha_span = (alpha_hi - alpha_lo) / 10
    beta_lo  = best_beta  - beta_span  / 2
    beta_hi  = best_beta  + beta_span  / 2
    alpha_lo = best_alpha - alpha_span / 2
    alpha_hi = best_alpha + alpha_span / 2

print()
print(f"Result:     weight = {best_beta:.6f} * height + {best_alpha:.4f}")
print(f"Result SSE: {best_sse_val:.4f}")

# Analytical solution for comparison
beta_exact  = (n * Sxy - Sx * Sy) / (n * Sxx - Sx**2)
alpha_exact = (Sy - beta_exact * Sx) / n
sse_exact   = sse(beta_exact, alpha_exact)
print()
print(f"Analytical: weight = {beta_exact:.6f} * height + {alpha_exact:.4f}")
print(f"Analytical SSE: {sse_exact:.4f}")

# --- Predict on test set ---
test_heights = []
with open('LR_armyTesting.csv') as f:
    for row in csv.reader(f):
        if row:
            try:
                test_heights.append(float(row[0]))
            except ValueError:
                pass

print()
print(f"{'#':>3}  {'Height':>7}  {'Predicted Weight':>16}")
print("-" * 30)
for i, h in enumerate(test_heights, 1):
    pred = best_beta * h + best_alpha
    print(f"{i:>3}  {h:>7.1f}  {pred:>16.2f}")
