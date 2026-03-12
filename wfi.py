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
    "gdp_per_capita_usd": [11500, 11800, 121000, 123000, 125000, 128000,
                           130000, 132000, 135000, 137000, 140000, 142000,
                           145000, 148000],          # <-- usd, not usb
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

# Evaluate model performance
y_pred = model.predict(training_data[features])
r2 = r2_score(training_data[target], y_pred)
print(f"Model R² Score: {r2:.4f}")

# Plot residuals (to check for patterns)
residuals = training_data[target] - y_pred
plt.figure(figsize=(10, 5))
plt.scatter(training_data["year"], residuals, color="red")
plt.axhline(y=0, color="blue", linestyle="--")
plt.title("Residuals Plot (Model Fit Check)")
plt.xlabel("Year")
plt.ylabel("Residuals")
plt.grid(True)
plt.savefig("residuals_plot.png")

# --------------------
# 3. compute genuine annual growth rates (CAGR) for the historical series
# --------------------
years = len(training_data) - 1  # 14 points span 13 full year‑on‑year intervals

def compound_rate(start, end, periods):
    """compound annual growth rate between `start` and `end` over `periods` years"""
    return (end / start) ** (1 / periods) - 1

latest_year_data = training_data.iloc[-1]  # Get last row

gdp_growth_rate = compound_rate(
    training_data["gdp_per_capita_usd"].iloc[0],
    latest_year_data["gdp_per_capita_usd"],
    years
)

military_pct_rate = compound_rate(
    training_data["military_pct_gdp"].iloc[0],
    latest_year_data["military_pct_gdp"],
    years
)

military_troops_rate = compound_rate(
    training_data["military_troops"].iloc[0],
    latest_year_data["military_troops"],
    years
)

fighter_rate = compound_rate(
    training_data["fighter_active"].iloc[0],
    latest_year_data["fighter_active"],
    years
)

nuclear_sub_rate = compound_rate(
    training_data["nuclear_submarines"].iloc[0],
    latest_year_data["nuclear_submarines"],
    years
)

carriers_rate = compound_rate(
    training_data["aircraft_carriers"].iloc[0],
    latest_year_data["aircraft_carriers"],
    years
)

# --------------------
# 4. SIMULATION PARAMETERS (2024 - 1 year ahead)
# --------------------

# Project future values based on those rates
def project_future_value(current_value, growth_rate, years_ahead=1):
    return current_value * (1 + growth_rate) ** years_ahead

future_gdp = project_future_value(latest_year_data["gdp_per_capita_usd"], gdp_growth_rate, years_ahead=1)
future_pct = project_future_value(latest_year_data["military_pct_gdp"], military_pct_rate, years_ahead=1)

# Create future data DataFrame for prediction
future_data = pd.DataFrame({
    "gdp_per_capita_usd": [future_gdp],
    "military_pct_gdp": [future_pct]
})

# Predict military spending using the trained model
predicted_spending = model.predict(future_data)[0]

army_strength = int(project_future_value(
    latest_year_data["military_troops"],
    military_troops_rate,
    years_ahead=1
))

fighter_sorties = int(project_future_value(
    latest_year_data["fighter_active"],
    fighter_rate,
    years_ahead=1
))

nuclear_subs = int(project_future_value(
    latest_year_data["nuclear_submarines"],
    nuclear_sub_rate,
    years_ahead=1
))

carriers = int(project_future_value(
    latest_year_data["aircraft_carriers"],
    carriers_rate,
    years_ahead=1
))

# --------------------
# 5. ARMY ATTRITION MODEL (Projected Reductions)
# --------------------
# Annual estimated attrition rates (as percentages of total forces)
RETIREMENT_RATE = 0.05       # ~5% standard retirement/separation
NATURAL_DEATH_RATE = 0.001   # ~0.1% expected natural mortality
TRAINING_DEATH_RATE = 0.0002 # ~0.02% expected training exercise fatalities

# Calculate reductions
retirements = int(army_strength * RETIREMENT_RATE)
natural_deaths = int(army_strength * NATURAL_DEATH_RATE)
training_deaths = int(army_strength * TRAINING_DEATH_RATE)

total_attrition = retirements + natural_deaths + training_deaths
net_army_strength = army_strength - total_attrition

# --------------------
# 6. SIMULATION OUTPUT (2024)
# --------------------
print("\n=== 2024 MILITARY SIMULATION RESULTS ===")
print(f"Predicted military spending (USD) = ${predicted_spending:,.0f}")
print(f"Projected Gross Army Strength (before attrition) = {army_strength:,}")
print(f"  - Less Retirements: {retirements:,}")
print(f"  - Less Natural Deaths: {natural_deaths:,}")
print(f"  - Less Training Casualties: {training_deaths:,}")
print(f"Net Army Strength (Active Personnel) = {net_army_strength:,}")
print(f"Nuclear submarines = {nuclear_subs:,}")
print(f"Fighter sorties (annual) = {fighter_sorties:,}")
print(f"Aircraft carriers = {carriers:,}")
# Calculate some derived metrics
military_efficiency = predicted_spending / net_army_strength * 1000
print(f"\nDerived Metrics:")
print(f"Military efficiency (USD per active soldier): ${military_efficiency:,.2f}")