import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --------------------
# 1. LOAD DATA (Simulated)
# --------------------
np.random.seed(42)  # For reproducibility
training_data = pd.DataFrame({
    "year": pd.date_range(2010, periods=14, freq="YE"),
    "gdp_per_capita_usd": [115000, 118000, 121000, 123000, 125000, 128000,
                           130000, 132000, 135000, 137000, 140000, 142000,
                           145000, 148000],
    "military_pct_gdp": [4.2, 3.9, 3.7, 3.5, 3.4, 4.8, 4.9, 5.1, 5.2,
                         5.3, 5.5, 5.6, 5.8, 5.9],
    "military_spending_usd": [950, 1020, 1080, 1120, 1150, 1450, 1500,
                              1580, 1620, 1650, 1700, 1750, 1800, 1850],
    "military_troops": [1_000_000, 1_050_000, 1_100_000, 1_150_000, 1_200_000,
                        1_250_000, 1_300_000, 1_350_000, 1_400_000, 1_450_000,
                        1_500_000, 1_550_000, 1_650_000, 1_870_000],
    "fighter_active": [150, 160, 170, 180, 190, 210, 230, 250, 270, 290,
                       310, 330, 350, 360],
    "nuclear_submarines": [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
                           34, 36, 38],
    "aircraft_carriers": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                          19]
})

# --------------------
# 2. MODEL SETUP & EVALUATION
# --------------------
features = ["gdp_per_capita_usd", "military_pct_gdp"]
target = "military_spending_usd"

model = LinearRegression()
model.fit(training_data[features], training_data[target])

y_pred = model.predict(training_data[features])
print(f"Model R² Score: {r2_score(training_data[target], y_pred):.4f}")

# Plot residuals
plt.figure(figsize=(10, 5))
plt.scatter(training_data["year"], training_data[target] - y_pred, color="red")
plt.axhline(y=0, color="blue", linestyle="--")
plt.title("Residuals Plot")
plt.xlabel("Year")
plt.ylabel("Residuals")
plt.grid(True)
plt.savefig("residuals_plot.png")

# --------------------
# 3. PROJECTIONS (2024 - 1 year ahead)
# --------------------
years = len(training_data) - 1
latest_data = training_data.iloc[-1]

def project_future_value(col_name, years_ahead=1):
    """Calculates CAGR and projects the value forward."""
    start, end = training_data[col_name].iloc[0], latest_data[col_name]
    cagr = (end / start) ** (1 / years) - 1
    return end * (1 + cagr) ** years_ahead

# Project values dynamically
projected = {col: project_future_value(col) for col in [
    "gdp_per_capita_usd", "military_pct_gdp", "military_troops", 
    "fighter_active", "nuclear_submarines", "aircraft_carriers"
]}

# Predict spending for 2024
future_data = pd.DataFrame([{
    "gdp_per_capita_usd": projected["gdp_per_capita_usd"], 
    "military_pct_gdp": projected["military_pct_gdp"]
}])
predicted_spending = model.predict(future_data)[0]

army_strength = int(projected["military_troops"])

# --------------------
# 4. ARMY ATTRITION MODEL (Projected Reductions)
# --------------------
RETIREMENT_RATE = 0.05       # ~5% standard retirement/separation
NATURAL_DEATH_RATE = 0.001   # ~0.1% expected natural mortality
TRAINING_DEATH_RATE = 0.0002 # ~0.02% expected training exercise fatalities

retirements = int(army_strength * RETIREMENT_RATE)
natural_deaths = int(army_strength * NATURAL_DEATH_RATE)
training_deaths = int(army_strength * TRAINING_DEATH_RATE)

net_army_strength = army_strength - (retirements + natural_deaths + training_deaths)

# --------------------
# 5. SIMULATION OUTPUT (2024)
# --------------------
print("\n=== 2024 MILITARY SIMULATION RESULTS ===")
print(f"Predicted military spending (USD) = ${predicted_spending:,.0f}")
print(f"Projected Gross Army Strength (before attrition) = {army_strength:,}")
print(f"  - Less Retirements: {retirements:,}")
print(f"  - Less Natural Deaths: {natural_deaths:,}")
print(f"  - Less Training Casualties: {training_deaths:,}")
print(f"Net Army Strength (Active Personnel) = {net_army_strength:,}")
print(f"Nuclear submarines = {int(projected['nuclear_submarines']):,}")
print(f"Fighter sorties (annual) = {int(projected['fighter_active']):,}")
print(f"Aircraft carriers = {int(projected['aircraft_carriers']):,}")
print(f"\nDerived Metrics:")
print(f"Military efficiency (USD per active soldier): ${predicted_spending / net_army_strength * 1000:,.2f}")