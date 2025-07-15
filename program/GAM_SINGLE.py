import pandas as pd
import numpy as np
from pygam import LinearGAM, s
import matplotlib.pyplot as plt
from scipy.stats import f

# Load data
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
df.set_index('Date', inplace=True)

# Handle outliers (interpolate visits after March 2023)
df['Visits'].loc['2023-03':] = np.nan
df['Visits'] = df['Visits'].interpolate(method='linear')

# Extract meteorological variables (exclude date and visits)
features = [col for col in df.columns if col not in ['Date', 'Visits']]

# List to store results
results = []

for feature in features:
    # Extract single variable data and remove missing values
    X = df[[feature]].dropna().values.reshape(-1, 1)  # Ensure X is 2D array
    y = df.loc[df[feature].dropna().index, 'Visits'].values

    # Build GAM model (default spline degrees of freedom=5)
    gam = LinearGAM(s(0, n_splines=5)).fit(X, y)

    # Manually calculate F-statistic and p-value
    y_pred = gam.predict(X)
    residuals = y - y_pred
    rss = np.sum(residuals ** 2)  # Residual sum of squares
    tss = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
    n = len(y)  # Sample size
    p = gam.statistics_['edof']  # Effective degrees of freedom

    # Calculate F-statistic and p-value
    f_statistic = ((tss - rss) / p) / (rss / (n - p - 1))
    p_value = 1 - f.cdf(f_statistic, p, n - p - 1)

    # Store results
    results.append({
        'Variable': feature,
        'Effective_DoF': round(p, 2),
        'F-statistic': round(f_statistic, 2),
        'P-value': round(p_value, 4)
    })

# Convert to DataFrame and sort by p-value
results_df = pd.DataFrame(results).sort_values(by='P-value')
print(results_df.to_string(index=False))
