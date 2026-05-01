import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/input/historic_demand_2009_2024.csv", parse_dates=["settlement_date"])

daily_df = df.groupby("settlement_date")["nd"].mean().reset_index()
daily_df.columns = ["date", "nd"]

fig, ax1 = plt.subplots(1, 1, figsize=(14, 5))

ax1.plot(daily_df["date"], daily_df["nd"], linewidth=0.5, alpha=0.7)
ax1.set_xlabel("Year")
ax1.set_ylabel("Consumption (MW)")
ax1.set_title("UK Daily Electricity Consumption 2009–2024")
ax1.grid(True)

# ax2.plot(daily_df["date"], np.log(daily_df["nd"]), linewidth=0.5, alpha=0.7)
# ax2.set_xlabel("Year")
# ax2.set_ylabel("Consumption (MW)")
# ax2.set_title("UK Daily Electricity Consumption 2009–2024 (Log Transformed)")
# ax2.grid(True)

plt.tight_layout()
plt.savefig("data/output/demand_by_year.png", dpi=150)
print("Saved to data/output/demand_by_year.png")

df2024 = pd.read_csv("data/input/historic_demand_year_2024.csv", parse_dates=["settlement_date"])
daily_df_2024 = df2024.groupby("settlement_date")["nd"].mean().reset_index()

plt.figure(figsize=(12, 5))
plt.plot(daily_df_2024["settlement_date"], daily_df_2024["nd"], linewidth=0.3, alpha=0.7)
plt.xlabel("Month")
plt.ylabel("Consumption (MW)")
plt.title("UK Daily Electricity Consumption 2024")
plt.xticks(
    ticks=pd.date_range("2024-01-01", periods=12, freq="MS"),
    labels=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
)
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/demand_2024_by_month.png", dpi=150)
print("Saved to data/output/demand_2024_by_month.png")
