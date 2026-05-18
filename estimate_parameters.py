import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/input/historic_demand_2009_2024.csv", parse_dates=["settlement_date"])
daily_df = df.groupby("settlement_date")["nd"].mean().reset_index()
daily_df.columns = ["date", "nd"]
daily_df = daily_df.sort_values("date").reset_index(drop=True)

train_df = daily_df[daily_df["date"].dt.year < 2024].copy().reset_index(drop=True)
test_df  = daily_df[daily_df["date"].dt.year >= 2024].copy().reset_index(drop=True)

# Trend m_t: yearly moving average (1.5.5 in the book)
train_df["trend"] = train_df["nd"].rolling(window=365, center=True, min_periods=365).mean()

# W_k: mean detrended value per day-of-year (annual seasonality)
train_df["detrended"] = train_df["nd"] - train_df["trend"]
train_df["doy"] = train_df["date"].dt.day_of_year
W = train_df.groupby("doy")["detrended"].mean()
s = W - W.mean()
train_df["seasonality"] = train_df["doy"].map(s)

# Weekly (day-of-week) seasonality from annually-deseasonalized residual
train_df["after_annual"] = train_df["detrended"] - train_df["seasonality"]
train_df["dow"] = train_df["date"].dt.dayofweek
W_weekly = train_df.dropna(subset=["after_annual"]).groupby("dow")["after_annual"].mean()
W_weekly = W_weekly - W_weekly.mean()
train_df["weekly_seasonal"] = train_df["dow"].map(W_weekly)

# Deseasonalized and residual
train_df["deseasonalized"] = train_df["nd"] - train_df["seasonality"] - train_df["weekly_seasonal"]
train_df["residual"] = train_df["nd"] - train_df["trend"] - train_df["seasonality"] - train_df["weekly_seasonal"]

# --- Polynomial fit (degree 3) to deseasonalized data ---
N = len(train_df)
t = np.arange(N) / (N - 1)
mask = train_df["deseasonalized"].notna()
poly_coeffs = np.polyfit(t[mask], train_df["deseasonalized"][mask], deg=3)
poly_fit = np.polyval(poly_coeffs, t)

# --- Harmonic fit (k=1) to annual seasonal component ---
doy = train_df["doy"].values
X_ann = np.column_stack([
    np.ones(len(doy)),
    np.cos(2 * np.pi * doy / 365),
    np.sin(2 * np.pi * doy / 365),
])
harm_ann_coeffs, _, _, _ = np.linalg.lstsq(X_ann, train_df["seasonality"].values, rcond=None)

# --- Harmonic fit (k=2) to weekly seasonal component ---
dow_vals = train_df["dow"].values
X_week = np.column_stack([
    np.ones(len(dow_vals)),
    np.cos(2 * np.pi * dow_vals / 7),
    np.sin(2 * np.pi * dow_vals / 7),
    np.cos(4 * np.pi * dow_vals / 7),
    np.sin(4 * np.pi * dow_vals / 7),
])
harm_week_coeffs, _, _, _ = np.linalg.lstsq(X_week, train_df["weekly_seasonal"].values, rcond=None)

def signed(val):
    return f"+ {val}" if val >= 0 else f"- {abs(val)}"

# Format polynomial label
a3, a2, a1, a0 = [round(c) for c in poly_coeffs]
poly_label = rf"$\hat{{m}}_t = {a0} {signed(a1)} t {signed(a2)} t^2 {signed(a3)} t^3$"

# Format annual harmonic label
h0, h1, hb1 = [round(c) for c in harm_ann_coeffs]
harm_ann_label = (
    rf"$\hat{{s}}_t = {h0} {signed(h1)} \cdot \cos\!\left(\frac{{2\pi t}}{{365}}\right)"
    rf" {signed(hb1)} \cdot \sin\!\left(\frac{{2\pi t}}{{365}}\right)$"
)

# Format weekly harmonic label
w0, w1, wb1, w2, wb2 = [round(c) for c in harm_week_coeffs]
harm_week_label = (
    rf"$\hat{{w}}_t = {w0} {signed(w1)} \cos\!\frac{{2\pi t}}{{7}}"
    rf" {signed(wb1)} \sin\!\frac{{2\pi t}}{{7}}"
    rf" {signed(w2)} \cos\!\frac{{4\pi t}}{{7}}"
    rf" {signed(wb2)} \sin\!\frac{{4\pi t}}{{7}}$"
)

# Deseasonalized with polynomial fit
plt.figure(figsize=(14, 5))
plt.plot(train_df["date"], train_df["deseasonalized"], linewidth=0.5, alpha=0.7, label="Deseasonalized data")
plt.plot(train_df["date"], poly_fit, linewidth=1.5, color="tab:red", label=poly_label)
plt.xlabel("Year")
plt.ylabel("Consumption (MW)")
plt.title("UK Daily Electricity Consumption 2009–2023 (Deseasonalized)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/deseasonalized.png", dpi=150)
plt.close()
print("Saved to data/output/deseasonalized.png")

# Annual seasonal component — one period (doy 1–365)
doy_one = np.arange(1, 366)
X_ann_one = np.column_stack([
    np.ones(365),
    np.cos(2 * np.pi * doy_one / 365),
    np.sin(2 * np.pi * doy_one / 365),
])
harm_ann_one = X_ann_one @ harm_ann_coeffs

plt.figure(figsize=(10, 4))
plt.plot(doy_one, s.reindex(doy_one), linewidth=1.0, alpha=0.8, label="Annual seasonal component")
plt.plot(doy_one, harm_ann_one, linewidth=1.5, color="tab:red", label=harm_ann_label)
plt.xlabel("Day of year")
plt.ylabel("Consumption (MW)")
plt.title("Annual Seasonal Component (One Period)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/seasonality.png", dpi=150)
plt.close()
print("Saved to data/output/seasonality.png")

# Weekly seasonal component — one period (Mon–Sun)
dow_one = np.arange(7)
X_week_one = np.column_stack([
    np.ones(7),
    np.cos(2 * np.pi * dow_one / 7),
    np.sin(2 * np.pi * dow_one / 7),
    np.cos(4 * np.pi * dow_one / 7),
    np.sin(4 * np.pi * dow_one / 7),
])
harm_week_one = X_week_one @ harm_week_coeffs

day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
plt.figure(figsize=(8, 4))
plt.plot(dow_one, W_weekly.reindex(dow_one), "o-", linewidth=1.0, alpha=0.8, label="Weekly seasonal component")
plt.plot(dow_one, harm_week_one, linewidth=1.5, color="tab:red", label=harm_week_label)
plt.xticks(dow_one, day_names)
plt.xlabel("Day of week")
plt.ylabel("Consumption (MW)")
plt.title("Weekly Seasonal Component (One Period)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/weekly_seasonality.png", dpi=150)
plt.close()
print("Saved to data/output/weekly_seasonality.png")

# Residual
plt.figure(figsize=(14, 5))
plt.plot(train_df["date"], train_df["residual"], linewidth=0.5, alpha=0.7)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xlabel("Year")
plt.ylabel("Consumption (MW)")
plt.title("UK Daily Electricity Consumption 2009–2023 (Residual)")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/residual.png", dpi=150)
plt.close()
print("Saved to data/output/residual.png")

# ACF of residuals
resid = train_df["residual"].dropna().values
n = len(resid)
max_lag = 100
resid_c = resid - resid.mean()
acf_vals = np.array([
    np.dot(resid_c[:n - k], resid_c[k:]) / np.dot(resid_c, resid_c)
    for k in range(max_lag + 1)
])
lags = np.arange(max_lag + 1)

plt.figure(figsize=(12, 4))
plt.bar(lags, acf_vals, width=0.5, color="tab:blue")
plt.axhline(0, color="black", linewidth=0.6)
plt.xlabel("Lag (days)")
plt.ylabel("ACF")
plt.title("ACF of Residuals")
plt.grid(True, axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("data/output/acf_residuals.png", dpi=150)
plt.close()
print("Saved to data/output/acf_residuals.png")

# Yule-Walker estimation with AICC model selection
p_max = 10
gamma = np.array([np.dot(resid_c[:n - k], resid_c[k:]) / n for k in range(p_max + 1)])

aicc_vals = np.full(p_max + 1, np.nan)
# p=0: white noise, 1 parameter (sigma^2)
aicc_vals[0] = n * np.log(gamma[0]) + 2 * n / (n - 2)

for p in range(1, p_max + 1):
    if n - p - 2 <= 0:
        break
    Gamma = np.array([[gamma[abs(i - j)] for j in range(p)] for i in range(p)])
    gamma_vec = gamma[1:p + 1]
    phi = np.linalg.solve(Gamma, gamma_vec)
    sigma2 = gamma[0] - phi @ gamma_vec
    if sigma2 <= 0:
        break
    aicc_vals[p] = n * np.log(sigma2) + 2 * (p + 1) * n / (n - p - 2)

p_opt = int(np.nanargmin(aicc_vals))
print(f"Optimal AR order: p = {p_opt},  AICC = {aicc_vals[p_opt]:.2f}")

# Refit at optimal order and report coefficients
if p_opt > 0:
    Gamma_opt = np.array([[gamma[abs(i - j)] for j in range(p_opt)] for i in range(p_opt)])
    gamma_vec_opt = gamma[1:p_opt + 1]
    phi_opt = np.linalg.solve(Gamma_opt, gamma_vec_opt)
    sigma2_opt = gamma[0] - phi_opt @ gamma_vec_opt
    print(f"AR({p_opt}) coefficients: {np.round(phi_opt, 4)}")
    print(f"Noise variance: {sigma2_opt:.2f}")

# Plot AICC vs p
plt.figure(figsize=(10, 4))
plt.plot(np.arange(p_max + 1), aicc_vals, marker="o", markersize=4)
plt.axvline(p_opt, color="tab:red", linestyle="--", linewidth=0.8, label=f"p = {p_opt}")
plt.xlabel("AR order p")
plt.ylabel("AICC")
plt.title("AICC vs AR Order (Yule-Walker)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/aicc.png", dpi=150)
plt.close()
print("Saved to data/output/aicc.png")
