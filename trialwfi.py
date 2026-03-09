import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --------------------
# 1. LOAD DATA (Simulated)
# --------------------
# Create a sample DataFrame representing military data over time.
training_data = pd.DataFrame({
    "year": pd.date_range(2010, periods=14, freq='YE'),
    "gdp_per_capita_usb": [11500, 11800, 121000, 123000, 125000, 128000, 130000, 132000, 135000, 137000, 140000, 142000, 145000, 148000],
    "military_pct_gdp": [4.2, 3.9, 3.7, 3.5, 3.4, 4.8, 4.9, 5.1, 5.2, 5.3, 5.5, 5.6, 5.8, 5.9],
    "military_spending_usb": [950, 1020, 1080, 1120, 1150, 1450, 1500, 1580, 1620, 1650, 1700, 1750, 1800, 1850],
    "military_troops": [500_000, 520_000, 540_000, 560_000, 580_000, 620_000, 650_000, 680_000, 700_000, 720_000, 750_000, 780_000, 800_000, 820_000],
    "fighter_active": [150, 160, 170, 180, 190, 210, 230, 250, 270, 290, 310, 330, 350, 360],
    "nuclear_submarines": [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
    "aircraft_carriers": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
})

# Add dummy columns to avoid errors when accessing non-existent columns.
training_data["military_titans"] = np.random.randint(100, 200, size=len(training_data))
training_data["fighter_aircraft"] = np.random.randint(100, 200, size=len(training_data))


# --------------------
# 2. MODEL SETUP
# --------------------
# Define features (independent variables) and target variable (dependent variable).
features = ["gdp_per_capita_usb", "military_pct_gdp"]
target = "military_spending_usb"

# Create a linear regression model.
model = LinearRegression()

# Train the model using the training data.
model.fit(training_data[features], training_data[target])

# --------------------
# 3. SIMULATION PARAMETERS (2024)
# --------------------
# Set parameters for the simulation year 2024.
future_gdp = 149_000  # Predicted GDP per capita in USD
future_pct = 5.2  # Predicted military percentage of GDP

# Create a DataFrame for the future year to make prediction.
future_data = pd.DataFrame({
    "gdp_per_capita_usb": [future_gdp],
    "military_pct_gdp": [future_pct]
})

# Predict military spending using the trained model.
predicted_spending = model.predict(future_data)[0]

# Extract latest values from training data for other simulation elements.
army_strength = training_data["military_troops"].iloc[-1]  # Total army personnel
fighter_sorties = training_data["fighter_active"].iloc[-1]  # Annual fighter sorties
nuclear_subs = training_data["nuclear_submarines"].iloc[-1]  # Number of nuclear submarines
carriers = training_data["aircraft_carriers"].iloc[-1]  # Number of aircraft carriers


# --------------------
# 4. SIMULATION OUTPUT (2024)
# --------------------
print("=== 2024 MILITARY SIMULATION RESULTS ===")
print(f"Predicted military spending (USD) = ${predicted_spending:,.0f}")
print(f"Army strength (personnel) = {army_strength:,}")
print(f"Nuclear submarines = {int(nuclear_subs):,}")  # Display as integer
print(f"Fighter sorties (annual) = {int(fighter_sorties):,}")
print(f"Air Force battle groups = {int(carriers):,}")  # Display as integer
