# war Fi 🛡️

**war Fi** is a Python-based military simulation and prediction tool. It uses historical (simulated) data to predict future military spending and other strategic metrics using Linear Regression.

## 🚀 Features

- **Military Spending Prediction**: Uses `scikit-learn`'s Linear Regression to forecast spending based on GDP per capita and military budget as a percentage of GDP.
- **Strategic Asset Tracking**: Simulates and predicts metrics for:
  - Total Army Strength (Personnel)
  - Nuclear Submarine Count
  - Annual Fighter Sorties
  - Aircraft Carrier/Battle Group Counts
- **Data Simulation**: Includes built-in historical data simulation for training and testing.

## 🛠️ Prerequisites

To run the simulation, you'll need Python installed along with the following libraries:

- `pandas`
- `numpy`
- `scikit-learn`

You can install the dependencies using pip:

```bash
pip install pandas numpy scikit-learn
```

## 📂 Project Structure

- `wfi.py`: The core simulation script for military spending prediction.
- `trialwfi.py`: An experimental version with additional dummy data and expanded simulation elements.
- `README.md`: Project documentation (you are here).

## 🚦 How to Run

Simply run either of the Python scripts:

```bash
python wfi.py
```
or
```bash
python trialwfi.py
```

## 📊 Sample Output

When running the simulation, you'll see output similar to this:

```text
=== 2024 MILITARY SIMULATION RESULTS ===
Predicted military spending (USD) = $1,850
Army strength (personnel) = 820,000
Nuclear submarines = 38
Fighter sorties (annual) = 360
Air Force battle groups = 19
```

## 📝 Note

This project currently uses simulated data for demonstration purposes. It is designed to be extensible, allowing for real-world historical data to be plugged in for more accurate geopolitical forecasting.
