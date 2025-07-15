import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Load data
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
df.set_index('Date', inplace=True)

# Extract target and feature columns
target = 'Visits'
features = ['AvgPressure', 'AvgWindSpeed', 'AvgTemperature', 'AvgHumidity', 'TotalPrecipitation', 'AvgSunshineHours']
data = df[[target] + features].copy()

# Initialize results storage
results = pd.DataFrame(columns=['Feature', 'Lag', 'Spearman_R', 'P-value'])

# Calculate lagged correlations
for feature in features:
    for lag in range(0, 5):  # lag0 to lag4
        # Create lagged feature
        shifted_feature = data[feature].shift(lag)  # Positive lag means feature lags behind visits
        combined = pd.concat([data[target], shifted_feature], axis=1).dropna()

        # Calculate Spearman correlation
        r, p = spearmanr(combined[target], combined[feature])

        # Store results
        results.loc[len(results)] = {
            'Feature': feature,
            'Lag': f'lag{lag}',
            'Spearman_R': round(r, 3),
            'P-value': round(p, 4)
        }

# Sort results by feature and lag
results = results.sort_values(by=['Feature', 'Lag'])

# Print results
print(results)
