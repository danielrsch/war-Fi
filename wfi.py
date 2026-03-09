import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data (simulated)
training_data = pd.DataFrame({
    "year": pd.date_range(2010, periods=14, freq='Y'),
    "gdp_per_capita_usb": [11500, 11800, 121000, 123000, 125000, 128000, 13000, 132000, 135000, 137000, 14000, 142000, 145000, 148000],
    "military_pct_gdp": [4.2, 3.9, 3.7, 3.5, 3.4, 4.8, 4.9, 5.1, 5.2, 5.3, 5.5, 5.6, 5.8, 5.9],
    "military_spending_usb": [950, 1020, 1080, 1120, 1150, 1450, 1500, 1580, 1620, 1650, 1700, 1750, 1800, 1850],
    "military_troops": [500_000, 520_000, 540_000, 560_000, 580_000, 620_000, 650_000, 680_000, 700_000, 720_000, 750_000, 780_000, 800_000, 820_000],
    "fighter_active": [150, 160, 170, 180, 190, 210, 230, 250, 270, 290, 310, 330, 350, 360],
    "nuclear_submarines": [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
    "aircraft_carriers": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
})

# Model setup
X = training_data[["gdp_per_capita_usb", "military_pct_gdp"]]
y = training_data["military_spending_usb"]

model = LinearRegression()
model.fit(X, y)

# Simulation parameters (2024)
future_gdp = 149_000
future_pct = 5.2

# Predict using the trained model
future_data = pd.DataFrame({
    'gdp_per_capita_usb': [future_gdp],
    'military_pct_gdp': [future_pct]
})
predicted_spending = model.predict(future_data)[0]

# Extract latest values from training data
air_force_troops = training_data["military_troops"].iloc[-1]  # Corrected column name
fighter_sorties = training_data["fighter_active"].iloc[-1]   # Corrected column name
nuclear_submarines = training_data["nuclear_submarines"].iloc[-1]
aircraft_carriers = training_data["aircraft_carriers"].iloc[-1]

# Simulation output (2024)
print("=== 2024 MILITARY SIMULATION RESULTS ===")
print(f"Predicted military spending (USD-B) = ${predicted_spending:,.0f}")
print(f"Army strength (personnel) = {air_force_troops:,}")  # Note: This reflects total military personnel
print(f"Nuclear submarines = {nuclear_submarines:,}")      # Shows submarine count (6B note removed)
print(f"Fighter sorties (annual) = {fighter_sorties:,}")   # Uses fighter_active as proxy
print(f"Air Force battle groups = {aircraft_carriers:,}")  # Aircraft carrier count
