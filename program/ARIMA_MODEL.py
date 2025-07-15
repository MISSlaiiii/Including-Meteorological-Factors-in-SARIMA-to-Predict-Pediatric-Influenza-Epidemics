import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro, normaltest
from statsmodels.graphics.gofplots import qqplot
import warnings

# Set display options
pd.set_option('display.float_format', lambda x: '%.2f' % x)
plt.rcParams['font.sans-serif'] = ['SimHei']  # Display Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Display negative signs
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
df.set_index('Date', inplace=True)
ts = df['Visits']

# Handle outliers (data after March 2023 is significantly abnormal, use linear interpolation)
ts.loc['2023-03':] = np.nan
ts = ts.interpolate(method='linear')

# ADF test for stationarity
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("The series is stationary")
    else:
        print("The series is non-stationary, differencing is needed")

# Check stationarity
print("Stationarity Test:")
adf_test(ts)

# First-order differencing
ts_diff = ts.diff().dropna()
adf_test(ts_diff)

# Time series decomposition (additive model)
decomposition = seasonal_decompose(ts, model='additive', period=12)

# Auto-tune SARIMA parameters
model = auto_arima(
    ts,
    seasonal=True,
    m=12,
    stepwise=True,
    suppress_warnings=True,
    trace=True
)

order = model.order
seasonal_order = model.seasonal_order
print(f"Optimal Parameters: SARIMA{order}{seasonal_order}")

# Train the SARIMA model
model_fit = SARIMAX(
    ts,
    order=order,
    seasonal_order=seasonal_order
).fit(disp=False)

# Residual diagnostics
residuals = model_fit.resid

# Ljung-Box test for autocorrelation
lb_test = acorr_ljungbox(residuals, lags=12)
print(f"Ljung-Box Test p-value: {lb_test.iloc[-1, 1]}")

# Residual normality tests
print("\nResidual Normality Tests:")
print(f"Shapiro-Wilk Test: statistic={shapiro(residuals)[0]:.3f}, p-value={shapiro(residuals)[1]:.3f}")
print(f"Normality Test: statistic={normaltest(residuals)[0]:.3f}, p-value={normaltest(residuals)[1]:.3f}")

# Get fitted values and confidence intervals
fitted_values = model_fit.fittedvalues
prediction = model_fit.get_prediction(start=ts.index[0], end=ts.index[-1])
fitted_conf_int = prediction.conf_int()

# Forecast future 19 months (July 2023 to January 2025)
forecast = model_fit.get_forecast(steps=19)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Combine fitted values and forecast
full_index = ts.index.union(forecast_mean.index)
combined_mean = pd.concat([fitted_values, forecast_mean])
combined_conf = pd.concat([fitted_conf_int, conf_int])

# Visualize the results
plt.figure(figsize=(14, 7))
plt.plot(ts, label='Actual Values', color='blue', alpha=0.8)
plt.plot(combined_mean, label='Fitted/Forecast Values', color='red', linestyle='--', linewidth=2)

# Fill confidence interval
plt.fill_between(
    combined_conf.index,
    combined_conf.iloc[:, 0],
    combined_conf.iloc[:, 1],
    color='pink',
    alpha=0.3,
    label='95% Confidence Interval'
)

# Add a vertical line to separate training and forecasting periods
split_date = pd.to_datetime('2023-06-01')
plt.axvline(x=split_date, color='gray', linewidth=2, linestyle='--', label='Forecast Start')

plt.title('Tianjin City Medical Visits Forecast (SARIMA Model)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Visits', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Output forecast table
forecast_dates = pd.date_range(start='2023-07-01', periods=19, freq='MS')
forecast_df = pd.DataFrame({
    'Forecast': forecast_mean.values,
    'Lower 95% CI': conf_int.iloc[:, 0],
    'Upper 95% CI': conf_int.iloc[:, 1]
}, index=forecast_dates)

print("\nForecast Results:")
print(forecast_df)
