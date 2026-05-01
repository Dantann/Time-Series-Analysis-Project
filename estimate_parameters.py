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

# Deseasonalized
plt.figure(figsize=(14, 5))
plt.plot(daily_df["date"], daily_df["deseasonalized"], linewidth=0.5, alpha=0.7)
plt.xlabel("Year")
plt.ylabel("Consumption (MW)")
plt.title("UK Daily Electricity Consumption 2009–2024 (Deseasonalized)")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/deseasonalized.png", dpi=150)
plt.close()
print("Saved to data/output/deseasonalized.png")

# Seasonality
plt.figure(figsize=(14, 5))
plt.plot(daily_df["date"], daily_df["seasonality"], linewidth=0.5, alpha=0.7)
plt.xlabel("Year")
plt.ylabel("Consumption (MW)")
plt.title("UK Daily Electricity Consumption 2009–2024 (Seasonal Component)")
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

