import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/input/historic_demand_2009_2024.csv", parse_dates=["settlement_date"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(df["settlement_date"], df["nd"], linewidth=0.3, alpha=0.7)
ax1.set_ylabel("UK Electricity Consumption (MW)")
ax1.set_title("UK Electricity Consumption 2009–2024")
ax1.grid(True)

ax2.plot(df["settlement_date"], np.log(df["nd"]), linewidth=0.3, alpha=0.7)
ax2.set_xlabel("Year")
ax2.set_ylabel("UK Electricity Consumption (MW)")
ax2.set_title("UK Electricity Consumption 2009–2024 (Log Transformed)")
ax2.grid(True)

plt.tight_layout()
plt.savefig("data/output/demand_by_year.png", dpi=150)
print("Saved to data/output/demand_by_year.png")

df2024 = pd.read_csv("data/input/historic_demand_year_2024.csv", parse_dates=["settlement_date"])

plt.figure(figsize=(12, 5))
plt.plot(df2024["settlement_date"], df2024["nd"], linewidth=0.3, alpha=0.7)
plt.xlabel("Month")
plt.ylabel("UK Electricity Consumption (MW)")
plt.title("UK Electricity Consumption 2024s")
plt.xticks(
    ticks=pd.date_range("2024-01-01", periods=12, freq="MS"),
    labels=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
)
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/demand_2024_by_month.png", dpi=150)
print("Saved to data/output/demand_2024_by_month.png")
