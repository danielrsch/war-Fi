# war Fi

This repository contains a small Python simulation of
military‑spending data.  It builds a very simple linear‑regression
model from a handful of synthetic time‑series and then uses that model
together with a few other projections to produce a 2024 “military
simulation” report.

The Final primary script is `wfi.py` ; the file is
self‑contained and does not read or write any external data.

---

## What it does

1. **Load / simulate data**  
   Creates a `pandas.DataFrame` with 14 years of made‑up values for:
  --This values can be customised for personal uses--
   * GDP per capita (USD),
   * military spending as a percentage of GDP,
   * total military spending (USD),
   * number of troops, active fighters, nuclear submarines, carriers, etc.

   Two extra dummy columns are added for robustness.

2. **Fit a model**  
   A `sklearn.linear_model.LinearRegression` is trained to predict
   `military_spending_usb` from `gdp_per_capita_usb` and
   `military_pct_gdp`.  The R² score is printed and a residuals plot is
   displayed to check for systematic errors.

3. **Set up 2024 parameters**  
   Hypothetical 2024 values for GDP per capita and military‑%‑of‑GDP are
   used to generate a one‑row `DataFrame` which is fed to the trained
   model to obtain a spending forecast.

4. **Project other quantities**  
   Compound annual growth rates (CAGR) are computed for troops, fighters,
   subs and carriers based on the historical series; those rates are then
   applied to the latest values to project four years ahead.

5. **Print results**  
   The script prints the predicted spending, projected counts and a
   derived “military efficiency” metric.

---

## Requirements

The script was developed with Python 3.7+ and depends on the following
libraries:

* `pandas`
* `numpy`
* `matplotlib`
* `scikit-learn`

Install them into your chosen interpreter/venv before running the
program:

```bash
pip install --upgrade pandas numpy matplotlib scikit-learn
# or, on macOS with the system python:
# pip3 install --upgrade pandas numpy matplotlib scikit-learn