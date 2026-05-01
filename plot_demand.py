import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df2024 = pd.read_csv("data/input/historic_demand_year_2024.csv", parse_dates=["settlement_date"])
df2024 = df2024.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)

plt.figure(figsize=(12, 5))
plt.plot(df2024["settlement_date"], df2024["nd"], linewidth=0.3, alpha=0.5)
plt.xlabel("Month")
plt.ylabel("Consumption (MW)")
plt.title("UK Electricity Consumption 2024")
plt.xticks(
    ticks=pd.date_range("2024-01-01", periods=12, freq="MS"),
    labels=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
)
plt.grid(True)
plt.tight_layout()
plt.savefig("data/output/demand_2024_by_month.png", dpi=150)
print("Saved to data/output/demand_2024_by_month.png")
