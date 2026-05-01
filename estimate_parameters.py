import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = 48

df = pd.read_csv("data/input/historic_demand_year_2024.csv", parse_dates=["settlement_date"])
df = df.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)

# Trend m_t: moving average over one full day (d=48) centered
df["trend"] = df["nd"].rolling(window=d, center=True, min_periods=d).mean()

# W_k: mean detrended value per settlement period (1–48)
df["detrended"] = df["nd"] - df["trend"]
W = df.groupby("settlement_period")["detrended"].mean()

# s_k = W_k - mean(W)
s = W - W.mean()
df["seasonality"] = df["settlement_period"].map(s)

# Deseasonalized and residual
df["deseasonalized"] = df["nd"] - df["seasonality"]
df["residual"] = df["nd"] - df["trend"] - df["seasonality"]

# --- Polynomial fit (degree 3) to deseasonalized data ---
N = len(df)
t = np.arange(N) / (N - 1)
mask = df["deseasonalized"].notna()
poly_coeffs = np.polyfit(t[mask], df["deseasonalized"][mask], deg=3)
poly_fit = np.polyval(poly_coeffs, t)

# --- Harmonic fit (k=1) to seasonal component, period d=48 ---
sp = df["settlement_period"].values
X = np.column_stack([
    np.ones(len(sp)),
    np.cos(2 * np.pi * sp / d),
    np.sin(2 * np.pi * sp / d),
])
harm_coeffs, _, _, _ = np.linalg.lstsq(X, df["seasonality"].values, rcond=None)
harm_fit = X @ harm_coeffs

# Format polynomial label with coefficients
a3, a2, a1, a0 = [round(c) for c in poly_coeffs]
def signed(val):
    return f"+ {val}" if val >= 0 else f"- {abs(val)}"
poly_label = rf"$\hat{{m}}_t = {a0} {signed(a1)} t {signed(a2)} t^2 {signed(a3)} t^3$"

# Format harmonic label with coefficients
h0, h1, b1 = [round(c) for c in harm_coeffs]
harm_label = (
    rf"$\hat{{s}}_t = {h0} {signed(h1)} \cdot \cos\!\left(\frac{{2\pi t}}{{{d}}}\right)"
    rf" {signed(b1)} \cdot \sin\!\left(\frac{{2\pi t}}{{{d}}}\right)$"
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
