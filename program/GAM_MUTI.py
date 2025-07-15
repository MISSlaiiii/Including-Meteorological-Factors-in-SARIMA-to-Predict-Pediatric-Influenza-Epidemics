import pandas as pd
import numpy as np
from pygam import LinearGAM, s
import matplotlib.pyplot as plt
from scipy import stats

# Load data
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
df.set_index('Date', inplace=True)

# Handle outliers (interpolate visits data after March 2023)
df['Visits'].loc['2023-03':] = np.nan
df['Visits'] = df['Visits'].interpolate(method='linear')

# Select significant variables (based on univariate analysis results)
significant_features = ['Average Humidity']  #
X = df[significant_features].dropna()
y = df.loc[X.index, 'Visits']

# Build multivariate GAM model (one smooth term per variable)
gam_multi = LinearGAM(
    s(0, n_splines=10),  # Corresponds to variables in significant_features
    fit_intercept=True
).fit(X, y)

# Model summary
print("Multivariate GAM Model Summary:")
print(gam_multi.summary())

# Calculate F-statistic and P-value for each variable
results_multi = []
for i, feature in enumerate(significant_features):
    # Extract prediction contribution of current variable
    XX = gam_multi.generate_X_grid(term=i)
    partial_dep = gam_multi.partial_dependence(term=i, X=XX)

    # Calculate degrees of freedom, F and P values (approximate)
    edof = gam_multi.statistics_['edof']
    f_statistic = (np.var(partial_dep) / edof) / (
                np.var(gam_multi.deviance_residuals(X, y)) / (len(y) - np.sum(gam_multi.statistics_['edof']) - 1))
    p_value = 1 - stats.f.cdf(f_statistic, edof, len(y) - np.sum(gam_multi.statistics_['edof']) - 1)

    results_multi.append({
        'Variable': feature,
        'Effective_DoF': round(edof, 2),
        'F-statistic': round(f_statistic, 2),
        'P-value': round(p_value, 4)
    })

# Output results
results_multi_df = pd.DataFrame(results_multi).sort_values(by='P-value')
print("\nMultivariate Significance Test:")
print(results_multi_df.to_string(index=False))
