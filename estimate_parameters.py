import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = 48

df = pd.read_csv("data/input/historic_demand_year_2024.csv", parse_dates=["settlement_date"])
df = df.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)
df["day_of_week"] = df["settlement_date"].dt.dayofweek  # 0=Mon, 6=Sun

# Trend m_t: moving average over one full day (d=48) centered
df["trend"] = df["nd"].rolling(window=d, center=True, min_periods=d).mean()

# W_{d,k}: mean detrended value per (day_of_week, settlement_period) — 7×48 = 336 bins
df["detrended"] = df["nd"] - df["trend"]
W = df.groupby(["day_of_week", "settlement_period"])["detrended"].mean()

# s_{d,k} = W_{d,k} - mean(W)
s = W - W.mean()
s_df = s.reset_index(name="seasonality")
df = df.merge(s_df, on=["day_of_week", "settlement_period"], how="left")

# Deseasonalized and residual
df["deseasonalized"] = df["nd"] - df["seasonality"]
df["residual"] = df["nd"] - df["trend"] - df["seasonality"]

# --- Polynomial fit (degree 3) to deseasonalized data ---
N = len(df)
t = np.arange(N) / (N - 1)
mask = df["deseasonalized"].notna()
poly_coeffs = np.polyfit(t[mask], df["deseasonalized"][mask], deg=3)
poly_fit = np.polyval(poly_coeffs, t)

# --- Harmonic fit (k=1) to seasonal component: daily period d=48 and weekly period 7 ---
sp = df["settlement_period"].values
dow = df["day_of_week"].values
X = np.column_stack([
    np.ones(len(sp)),
    np.cos(2 * np.pi * sp / d),
    np.sin(2 * np.pi * sp / d),
    np.cos(2 * np.pi * dow / 7),
    np.sin(2 * np.pi * dow / 7),
])
harm_coeffs, _, _, _ = np.linalg.lstsq(X, df["seasonality"].values, rcond=None)
harm_fit = X @ harm_coeffs

print("Seasonal component (harmonic fit, k=1, daily period=48, weekly period=7):")
print(f"  a0     = {harm_coeffs[0]:.4f}")
print(f"  a1     = {harm_coeffs[1]:.4f}  (cos daily)")
print(f"  b1     = {harm_coeffs[2]:.4f}  (sin daily)")
print(f"  c1     = {harm_coeffs[3]:.4f}  (cos weekly)")
print(f"  e1     = {harm_coeffs[4]:.4f}  (sin weekly)")

# Format polynomial label with coefficients
a3, a2, a1, a0 = [round(c) for c in poly_coeffs]
def signed(val):
    return f"+ {val}" if val >= 0 else f"- {abs(val)}"
poly_label = rf"$\hat{{m}}_t = {a0} {signed(a1)} t {signed(a2)} t^2 {signed(a3)} t^3$"

# Format harmonic label with coefficients
h0, h1, hb1, hc1, he1 = [round(c) for c in harm_coeffs]
harm_label = (
    rf"$\hat{{s}}_{{d,k}} = {h0} {signed(h1)} \cdot \cos\!\left(\frac{{2\pi k}}{{48}}\right)"
    rf" {signed(hb1)} \cdot \sin\!\left(\frac{{2\pi k}}{{48}}\right)"
    rf" {signed(hc1)} \cdot \cos\!\left(\frac{{2\pi d}}{{7}}\right)"
    rf" {signed(he1)} \cdot \sin\!\left(\frac{{2\pi d}}{{7}}\right)$"
)

dates = df["settlement_date"]
month_ticks = pd.date_range("2024-01-01", periods=12, freq="MS")
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# Deseasonalized with polynomial fit
plt.figure(figsize=(14, 5))
plt.plot(dates, df["deseasonalized"], linewidth=0.3, alpha=0.5, label="Deseasonalized data")
plt.plot(dates, poly_fit, linewidth=1.5, color="tab:red", label=poly_label)
plt.xticks(ticks=month_ticks, labels=month_labels)
plt.xlabel("Month")
plt.ylabel("Consumption (MW)")
plt.title("UK Electricity Consumption 2024 (Deseasonalized)")
leg = plt.legend()
leg.legend_handles[0].set_linewidth(1.5)
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/deseasonalized.png", dpi=150)
plt.close()
print("Saved to data/output/deseasonalized.png")

# Residual
plt.figure(figsize=(14, 5))
plt.plot(dates, df["residual"], linewidth=0.3, alpha=0.5)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xticks(ticks=month_ticks, labels=month_labels)
plt.xlabel("Month")
plt.ylabel("Consumption (MW)")
plt.title("UK Electricity Consumption 2024 (Residual)")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/residual.png", dpi=150)
plt.close()
print("Saved to data/output/residual.png")

# Seasonal differencing at lag d=48: Y_t = residual_t - residual_{t-48}
max_lag = 5 * d  # 5 days of lags
_res = df["residual"].dropna().values
x = _res[d:] - _res[:-d]
n = len(x)
x_centered = x - x.mean()
lags = np.arange(0, max_lag + 1)
gamma = np.array([
    np.sum(x_centered[:n - h] * x_centered[h:]) / n
    for h in lags
])

rho = gamma / gamma[0]

plt.figure(figsize=(14, 5))
plt.plot(lags, rho, linewidth=0.8, color="tab:blue")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
for k in range(1, max_lag // d + 1):
    plt.axvline(k * d, color="tab:red", linewidth=0.6, linestyle=":",
                label="Daily period (48)" if k == 1 else None)
plt.xlabel("h")
plt.ylabel(r"$\hat{\rho}(h)$")
plt.title(r"Sample ACF of Seasonally Differenced Residuals ($\nabla_{48}$)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/sample_acf.png", dpi=150)
plt.close()
print("Saved to data/output/sample_acf.png")

# --- Yule-Walker AR(m) order selection via AICc ---

def fit_ar_yule_walker(gamma, m):
    """Solve Yule-Walker equations for AR(m). Returns (phi, sigma2)."""
    Gamma = np.array([[gamma[abs(i - j)] for j in range(m)] for i in range(m)])
    gamma_vec = gamma[1 : m + 1]
    phi = np.linalg.solve(Gamma, gamma_vec)
    sigma2 = gamma[0] - phi @ gamma_vec
    return phi, sigma2

def aicc(sigma2, n, m):
    k = m + 1  # m AR coefficients + noise variance
    aic = n * np.log(sigma2) + 2 * k
    return aic + 2 * k * (k + 1) / (n - k - 1)

def best_ar_yule_walker(gamma, n, m_max):
    """Return the AR order in 1..m_max that minimises AICc."""
    scores = []
    for m in range(1, m_max + 1):
        phi, sigma2 = fit_ar_yule_walker(gamma, m)
        scores.append((m, phi, sigma2, aicc(sigma2, n, m)))
    return min(scores, key=lambda r: r[3]), scores

best, all_scores = best_ar_yule_walker(gamma, n, m_max=10)
m_best, phi_best, sigma2_best, aicc_best = best

print("\nYule-Walker AR(m) order selection (AICc):")
print(f"  {'m':>3}  {'AICc':>14}")
for m, _, _, score in all_scores:
    marker = " <-- best" if m == m_best else ""
    print(f"  {m:>3}  {score:>14.2f}{marker}")
print(f"\nBest order: AR({m_best})")
print(f"  phi    = {phi_best}")
print(f"  sigma2 = {sigma2_best:.2f} MW^2")
print(f"  sigma  = {np.sqrt(sigma2_best):.2f} MW")

# Plot 1: AICc vs AR order
orders = [r[0] for r in all_scores]
aicc_vals = [r[3] for r in all_scores]

plt.figure(figsize=(7, 4))
plt.plot(orders, aicc_vals, marker="o", linewidth=1.2, color="tab:blue")
plt.axvline(m_best, color="tab:red", linewidth=1, linestyle="--",
            label=f"Best: AR({m_best})")
plt.xlabel("AR order $m$")
plt.ylabel("AICc")
plt.title("AICc by AR Order (Yule-Walker)")
plt.xticks(orders)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/yule_walker_aicc.png", dpi=150)
plt.close()
print("Saved to data/output/yule_walker_aicc.png")

# Plot 2: Sample ACF vs theoretical ACF of best AR(m)
# Theoretical ACF via Yule-Walker recursion: rho(h) = sum_j phi_j * rho(h-j)
rho_theory = np.zeros(max_lag + 1)
rho_theory[0] = 1.0
for h in range(1, max_lag + 1):
    rho_theory[h] = sum(
        phi_best[j] * rho_theory[abs(h - (j + 1))] for j in range(m_best)
    )

phi_str = ", ".join(f"{p:.3f}" for p in phi_best)
plt.figure(figsize=(14, 5))
plt.plot(lags, rho, linewidth=0.8, color="tab:blue", label="Sample ACF")
plt.plot(lags, rho_theory, linewidth=1.4, color="tab:red", linestyle="--",
         label=rf"AR({m_best}) Yule-Walker: $\hat{{\phi}}$ = [{phi_str}]")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
for k in range(1, max_lag // d + 1):
    plt.axvline(k * d, color="grey", linewidth=0.5, linestyle=":",
                label="Daily period (48)" if k == 1 else None)
plt.xlabel("Lag $h$")
plt.ylabel(r"$\hat{\rho}(h)$")
plt.title(rf"Sample ACF of $\nabla_{{48}}$ Residuals vs AR({m_best}) Yule-Walker Fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/yule_walker_acf.png", dpi=150)
plt.close()
print("Saved to data/output/yule_walker_acf.png")

# ACF without seasonal removal (detrended only)
x2 = df["detrended"].dropna().values
n2 = len(x2)
x2_centered = x2 - x2.mean()
gamma2 = np.array([
    np.sum(x2_centered[:n2 - h] * x2_centered[h:]) / n2
    for h in lags
])
rho2 = gamma2 / gamma2[0]

plt.figure(figsize=(14, 5))
plt.plot(lags, rho2, linewidth=0.8, color="tab:blue")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
for k in range(1, max_lag // d + 1):
    plt.axvline(k * d, color="tab:red", linewidth=0.6, linestyle=":",
                label="Daily period (48)" if k == 1 else None)
plt.xlabel("h")
plt.ylabel(r"$\hat{\rho}(h)$")
plt.title("Sample Autocorrelation Function (No Seasonal Removal)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/sample_acf_no_seasonal.png", dpi=150)
plt.close()
print("Saved to data/output/sample_acf_no_seasonal.png")
