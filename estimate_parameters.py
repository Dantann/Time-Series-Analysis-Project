import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/input/historic_demand_2009_2024.csv", parse_dates=["settlement_date"])
daily_df = df.groupby("settlement_date")["nd"].mean().reset_index()
daily_df.columns = ["date", "nd"]
daily_df = daily_df.sort_values("date").reset_index(drop=True)

# Trend m_t: yearly moving average (1.5.5 in the book)
daily_df["trend"] = daily_df["nd"].rolling(window=365, center=True, min_periods=365).mean()

# W_k: mean detrended value per day-of-year
daily_df["detrended"] = daily_df["nd"] - daily_df["trend"]
daily_df["doy"] = daily_df["date"].dt.day_of_year
W = daily_df.groupby("doy")["detrended"].mean()

# s_k = W_k - mean(W)
s = W - W.mean()
daily_df["seasonality"] = daily_df["doy"].map(s)

# Deseasonalized and residual
daily_df["deseasonalized"] = daily_df["nd"] - daily_df["seasonality"]
daily_df["residual"] = daily_df["nd"] - daily_df["trend"] - daily_df["seasonality"]

# --- Polynomial fit (degree 3) to deseasonalized data ---
N = len(daily_df)
t = np.arange(N) / (N - 1)
mask = daily_df["deseasonalized"].notna()
poly_coeffs = np.polyfit(t[mask], daily_df["deseasonalized"][mask], deg=3)
poly_fit = np.polyval(poly_coeffs, t)

# --- Harmonic fit (k=1) to seasonal component ---
# s_t = a0 + a1*cos(2*pi*t/365) + b1*sin(2*pi*t/365)
doy = daily_df["doy"].values
X = np.column_stack([
    np.ones(len(doy)),
    np.cos(2 * np.pi * doy / 365),
    np.sin(2 * np.pi * doy / 365),
])
harm_coeffs, _, _, _ = np.linalg.lstsq(X, daily_df["seasonality"].values, rcond=None)
harm_fit = X @ harm_coeffs

# Format polynomial label with coefficients
a3, a2, a1, a0 = [round(c) for c in poly_coeffs]
def signed(val):
    return f"+ {val}" if val >= 0 else f"- {abs(val)}"
poly_label = rf"$\hat{{m}}_t = {a0} {signed(a1)} t {signed(a2)} t^2 {signed(a3)} t^3$"

# Format harmonic label with coefficients
h0, h1, b1 = [round(c) for c in harm_coeffs]
harm_label = (
    rf"$\hat{{s}}_t = {h0} {signed(h1)} \cdot \cos\!\left(\frac{{2\pi t}}{{365}}\right)"
    rf" {signed(b1)} \cdot \sin\!\left(\frac{{2\pi t}}{{365}}\right)$"
)

# Deseasonalized with polynomial fit
plt.figure(figsize=(14, 5))
plt.plot(daily_df["date"], daily_df["deseasonalized"], linewidth=0.5, alpha=0.7, label="Deseasonalized data")
plt.plot(daily_df["date"], poly_fit, linewidth=1.5, color="tab:red", label=poly_label)
plt.xlabel("Year")
plt.ylabel("Consumption (MW)")
plt.title("UK Daily Electricity Consumption 2009–2024 (Deseasonalized)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/deseasonalized.png", dpi=150)
plt.close()
print("Saved to data/output/deseasonalized.png")

# Seasonal component with harmonic fit
plt.figure(figsize=(14, 5))
plt.plot(daily_df["date"], daily_df["seasonality"], linewidth=0.5, alpha=0.7, label="Seasonal data")
plt.plot(daily_df["date"], harm_fit, linewidth=1.5, color="tab:red", label=harm_label)
plt.xlabel("Year")
plt.ylabel("Consumption (MW)")
plt.title("UK Daily Electricity Consumption 2009–2024 (Seasonal Component)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/seasonality.png", dpi=150)
plt.close()
print("Saved to data/output/seasonality.png")

# Residual
plt.figure(figsize=(14, 5))
plt.plot(daily_df["date"], daily_df["residual"], linewidth=0.5, alpha=0.7)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xlabel("Year")
plt.ylabel("Consumption (MW)")
plt.title("UK Daily Electricity Consumption 2009–2024 (Residual)")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/residual.png", dpi=150)
plt.close()
print("Saved to data/output/residual.png")
