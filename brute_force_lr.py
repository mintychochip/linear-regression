import csv
import math

def load_csv(filename, num_columns):
    rows = []
    with open(filename) as f:
        for row in csv.reader(f):
            if len(row) == num_columns:
                try:
                    rows.append([float(v) for v in row])
                except ValueError:
                    pass
    return rows

def logspace(lo, hi, num):
    log_lo = math.log10(lo)
    step = (math.log10(hi) - log_lo) / (num - 1)
    return [10 ** (log_lo + i * step) for i in range(num)]

def build_grid(lo, hi, offsets):
    center = (hi + lo) / 2
    span   = (hi - lo) / 2
    return sorted([center + v * span for v in offsets] +
                  [center - v * span for v in offsets])

def search_grid(beta_lo, beta_hi, alpha_lo, alpha_hi, offsets):
    beta_grid  = build_grid(beta_lo,  beta_hi,  offsets)
    alpha_grid = build_grid(alpha_lo, alpha_hi, offsets)
    best_sse = float('inf')
    best_beta, best_alpha = (beta_lo + beta_hi) / 2, (alpha_lo + alpha_hi) / 2
    for beta in beta_grid:
        for alpha in alpha_grid:
            s = sse(beta, alpha)
            if s < best_sse:
                best_sse = s
                best_beta, best_alpha = beta, alpha
    return best_beta, best_alpha, best_sse

rows = load_csv('LR_armyTraining.csv', num_columns=2)
x = [r[0] for r in rows]
y = [r[1] for r in rows]

n   = len(x)
Sx  = sum(x)
Sy  = sum(y)
Sxx = sum(xi**2 for xi in x)
Syy = sum(yi**2 for yi in y)
Sxy = sum(xi * yi for xi, yi in zip(x, y))

def sse(beta, alpha):
    return beta*beta*Sxx + alpha*alpha*n + Syy + 2*beta*alpha*Sx - 2*beta*Sxy - 2*alpha*Sy

N_GRID    = 200
TOL       = 5e-4
MAX_ZOOMS = 50

offsets = logspace(1e-6, 1, N_GRID // 2)

beta_lo,  beta_hi  = -5000.0,   5000.0
alpha_lo, alpha_hi = -100000.0, 100000.0

print(f"Data: {n} points  |  Height [{min(x)}, {max(x)}]  |  Weight [{min(y)}, {max(y)}]")
print()
print(f"{'Zoom':>5}  {'beta':>12}  {'alpha':>14}  {'SSE':>14}  {'d_beta':>10}  {'d_alpha':>10}")
print("-" * 75)

prev_beta, prev_alpha = None, None

for zoom in range(MAX_ZOOMS):
    best_beta, best_alpha, best_sse_val = search_grid(beta_lo, beta_hi, alpha_lo, alpha_hi, offsets)
    d_beta  = abs(best_beta  - prev_beta)  if prev_beta  is not None else float('inf')
    d_alpha = abs(best_alpha - prev_alpha) if prev_alpha is not None else float('inf')
    print(f"{zoom+1:>5}  {best_beta:>12.6f}  {best_alpha:>14.4f}  {best_sse_val:>14.4f}  {d_beta:>10.2e}  {d_alpha:>10.2e}")
    if d_beta < TOL and d_alpha < TOL:
        print(f"\n  Converged after {zoom+1} iterations (beta and alpha stable to 3 decimal places)")
        break
    prev_beta, prev_alpha = best_beta, best_alpha
    beta_half  = (beta_hi  - beta_lo)  / 2
    alpha_half = (alpha_hi - alpha_lo) / 2
    beta_lo,  beta_hi  = best_beta  - beta_half,  best_beta  + beta_half
    alpha_lo, alpha_hi = best_alpha - alpha_half, best_alpha + alpha_half

beta_exact  = (n * Sxy - Sx * Sy) / (n * Sxx - Sx**2)
alpha_exact = (Sy - beta_exact * Sx) / n

print()
print(f"Result:     weight = {best_beta:.6f} * height + {best_alpha:.4f}  (SSE {best_sse_val:.4f})")
print(f"Analytical: weight = {beta_exact:.6f} * height + {alpha_exact:.4f}  (SSE {sse(beta_exact, alpha_exact):.4f})")

test_heights = [r[0] for r in load_csv('LR_armyTesting.csv', num_columns=1)]

actual_weight = [x[0] for x in load_csv('LR_armyResults.csv', num_columns=1)]

print()
print(f"{'#':>3}  {'Height':>7}  {'Predicted Weight':>16}  {'Actual Weight':>15}.")
print("-" * 45)
predicted_weight = []
for i, h in enumerate(test_heights, 0):
    w = best_beta * h + best_alpha
    print(f"{i+1:>3}  {h:>7.1f}  {w:>10.2f} {actual_weight[i]:>18.2f}")
    predicted_weight.append(round(w,2))

sse_test = sum((p - a) ** 2 for p, a in zip(predicted_weight, actual_weight))
print(f"\nSSE (predicted vs actual weight): {sse_test:.4f}")