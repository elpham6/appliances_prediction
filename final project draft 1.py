#%% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import seaborn as sns
from sktime.forecasting.arima import AutoARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_process import ArmaProcess
from tools import ADF_Cal, Cal_rolling_mean_var, ACF_PACF_Plot, generate_arma,create_gpac_table, kpss_test
from scipy.stats import chi2
############## EDA #################
#%%
# upload this to github later for hardcoding
# url =
df = pd.read_csv("energydata_complete.csv")
print(df.head())

# %%
# count missing values
clean_df = df.dropna()
print(clean_df.info)
# data has no missing values

# %%
# Setting a time column as the index
clean_df['date'] = pd.to_datetime(clean_df['date'])
clean_df.set_index('date', inplace=True)

# separate X and Y
Y = clean_df['Appliances']
X = clean_df.drop('Appliances', axis=1)

# %%
# plot dependent against time
plt.figure()
clean_df['Appliances'].plot()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Energy vs Time')
plt.show()
# %%
# Plot ACF/PACF
ACF_PACF_Plot(Y, 50)

# %%
# Plotting Heatmap
corr = clean_df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# %%
from sklearn.model_selection import train_test_split
# Split 80/20 train/test
# Splitting the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=False, random_state=6313)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# %%
from statsmodels.tsa.stattools import adfuller, kpss
######## Test stationarity #############
adf_result = ADF_Cal(clean_df['Appliances'])
print(adf_result)
# KPSS Test
kpss_result = kpss_test(clean_df['Appliances'])
print(kpss_result)
# Rolling Mean and Variance
Cal_rolling_mean_var(clean_df, "Appliances")
# Data is stationary

# %%
####### STL Decomp ########
from statsmodels.tsa.seasonal import STL
energy = clean_df["Appliances"]
if energy.isna().any():
    energy.interpolate(inplace=True)
stl = STL(energy, 144)
stl_result = stl.fit()
# plotting the components
plt.figure()
stl_result.plot()
plt.show()

# Calculate strength of trend and seasonal components
T = stl_result.trend
S = stl_result.seasonal
R = stl_result.resid

def str_trend_seasonal(T, S, R):
    F = np.maximum(0 ,1- np.var(np.array(R))/np.var(np.array(T+R)) )
    print(f'Strength of trend for the raw data is {100*F:.3f}%')

    FS = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(S + R)))
    print(f'Strength of seasonality for the raw data is {100*FS:.3f}%')

str_trend_seasonal(T,S,R)
# no strong trend or seasonality

# %%
##### Holt-Winters method #####
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
# Fit the Holt-Winters model
hw_model = ExponentialSmoothing(Y_train, 
                             trend='mul', seasonal = "mul",
                             seasonal_periods = 144,
                             damped=True).fit() # 144 10-min intervals a day
# Make predictions
hw_pred = hw_model.forecast(len(Y_test))

# Evaluate the model
hw_mse = mean_squared_error(Y_test, hw_pred)
print('Mean Squared Error:', hw_mse)

# Plotting the results
plt.figure()
Y_train.plot(label='Train')
Y_test.plot(label='Test')
hw_pred.plot(label='Holt-Winters Prediction')
plt.legend()
plt.show()

# %%
###### Feature Selection #######
# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the data
X_std = StandardScaler().fit_transform(X)  

# Apply PCA
pca = PCA(n_components=2)  # need to fix n_components
principalComponents = pca.fit_transform(X_std)

# To view explained variance
pca.explained_variance_ratio_
# SVD
U, s, VT = np.linalg.svd(X_std)
# Cond. number
cond_number = np.linalg.cond(X_std)

#%% 
# Backwards Stepwise
import statsmodels.api as sm

def backward_elimination(data, target, significance_level = 0.05):
    features = data.columns.tolist()
    while(len(features) > 0):
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
        
        max_p_value = p_values.max()
        if max_p_value >= significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    
    return features

selected_features = backward_elimination(X, Y) 
print(selected_features) 

# %%
###### Baseline Model ######
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# Drift forecast
def drift_forecast(train, h):
    T = len(train)
    y1 = train[0] # get first value
    yT = train[-1] # get last training value
    forecasts = [yT + (i / (T-1)) * (yT - y1) for i in range(1, h+1)]
    return forecasts

# Average forecast
def average_forecast(train, h):
    return [np.mean(train)] * h

# Naive forecast
def naive_forecast(train, h):
    return [train[-1]] * h

# SES
def ses_forecast(train, alpha, h):
    forecasts = [train[0]]  # Initial forecast is the first training data point
    for i in range(1, h+1):
        forecast = alpha * train[i-1] + (1 - alpha) * forecasts[-1]
        forecasts.append(forecast)
    return forecasts[1:]  # Skip the initial value

h = len(Y_test)
alpha = 0.5
# Average Model
avg = average_forecast(Y_train, h)

# Naïve Model
naive = naive_forecast(Y_train, h)

# Drift Model
drift = drift_forecast(Y_train, h)

# Simple Smoothing
simple_model = SimpleExpSmoothing(Y_train[:-h]).fit(smoothing_level=0.2)
simple_forecast = simple_model.forecast(h)

# Exponential Smoothing
exp_model = ExponentialSmoothing(Y_train[:-h], trend='add', seasonal='add', seasonal_periods=144).fit()
exp_forecast = exp_model.forecast(h)

#%%
def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error between the true and predicted values.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    float: The mean squared error.
    """
    # Convert inputs to numpy arrays 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the mean squared error
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

actual = Y_test

# Calculate MSE for each model
mse_avg = mean_squared_error(actual, avg)
mse_naive = mean_squared_error(actual, naive)
mse_drift = mean_squared_error(actual, drift)
mse_simple = mean_squared_error(actual, simple_forecast)
mse_exp = mean_squared_error(actual, exp_forecast)

# Print MSE values
print("MSE - Average:", mse_avg)
print("MSE - Naive:", mse_naive)
print("MSE - Drift:", mse_drift)
print("MSE - Simple Smoothing:", mse_simple)
print("MSE - Exponential Smoothing:", mse_exp)

#%%
###### Regression ######
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.stattools import durbin_watson
# Remove features by backwards selection
X_selected_train = X_train[selected_features]
X_selected_test = X_test[selected_features]
# Fit the model
model = sm.OLS(Y_train, sm.add_constant(X_selected_train)).fit()
# One-step ahead prediction
predictions = model.predict(sm.add_constant(X_selected_test))
# Model Evaluation
mse = mean_squared_error(Y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, predictions)
print(model.summary())
print(f"RMSE: {rmse}, R-squared: {r2}")

# ACF of Residuals and Q-Value
residuals = Y_test - predictions
ACF_PACF_Plot(residuals, 50)
plt.show()

# Durbin-Watson Test for autocorrelation (approximate Q-value)
dw = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw}")

# Calculate Q and Q-critical
acf_res = acf(residuals, nlags=30)
Q = len(Y_train)*np.sum(np.square(acf_res[30:]))
DOF = 30
alfa = 0.05
from scipy.stats import chi2
chi_critical = chi2.ppf(1-alfa, DOF)
print("Q = ", Q)
print("Q* = ", chi_critical)
if Q< chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white ")
# Variance and Mean of the Residuals
var_res = np.var(residuals)
mean_res = np.mean(residuals)
print(f"Variance of Residuals: {var_res}, Mean of Residuals: {mean_res}")

# Plot Train, Test, and Predicted Values
plt.figure(figsize=(10, 6))
plt.plot(Y_train, label='Train')
plt.plot(Y_test.index, Y_test, label='Test')
plt.plot(Y_test.index, predictions, label='Predictions')
plt.legend()
plt.show()
#%%
##### ARMA/ARIMA/SARIMA/Multiplicative ######

### Order Determination ###
# GPAC Table #
acf1 = acf(Y_train, nlags=50)
gpac_prelim = create_gpac_table(acf1, 12, 12)
# Possibly AR(1) or ARMA(1, 10) or AR(8) or ARMA(8, 10)

# ACF/PACF
ACF_PACF_Plot(Y_train, lags=50)
# ACF/PACF shows AR model (tails off, cuts off after lag 1)
# So probably AR(1), but we can test a couple of models
#%%
### Estimate Parameters ###
p = 1 # AR order
q = 0  # MA order
arma = sm.tsa.ARIMA(Y_train, order=(p, 0, q)).fit()
print(arma.summary())
res_arma = arma.resid
ACF_PACF_Plot(res_arma, 50)

#%%
# Choose best model by AIC
# import itertools

# # Define the p and q ranges to test
# p = range(1, 11)  
# q = range(0, 11)  

# # Generate all different combinations of p, q
# pq = list(itertools.product(p, q))

# best_aic = float("inf")
# best_order = None
# best_model = None

# for order in pq:
#     try:
#         model = sm.tsa.ARIMA(Y_train, order=(order[0], 0, order[1])).fit()
#         if model.aic < best_aic:
#             best_aic = model.aic
#             best_order = order
#             best_model = model
#     except:  # Handle the case where the model fails to converge
#         continue

# print(f'Best ARMA{best_order} AIC: {best_aic}')

#%%
##### Forecast Function ######
# def forecast_function(history, order, seasonal_order, forecast_periods):
#     """
#     Forecast future values using a SARIMA model.
    
#     Parameters:
#     history (array-like): The time series data to fit the SARIMA model.
#     order (tuple): The (p, d, q) order of the model for the number of AR parameters, differences, and MA parameters.
#     seasonal_order (tuple): The (P, D, Q, s) seasonal order of the model.
#     forecast_periods (int): Number of periods to forecast forward.
    
#     Returns:
#     array: Forecasted values.
#     """
#     # Fit the SARIMA model
#     model = sm.tsa.SARIMAX(history, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
#     model_fit = model.fit(disp=False)
    
#     # Make forecast
#     forecast = model_fit.forecast(steps=forecast_periods)
    
#     return forecast

# import statsmodels.api as sm

# def forecast_arima(history, const, ar1_coeff, periods):
#     """
#     Forecast future values using an ARIMA(1, 0, 0) model.

#     Parameters:
#     history (list or array-like): The historical time series data.
#     const (float): The constant term of the ARIMA model.
#     ar1_coeff (float): The coefficient for the AR(1) term.
#     periods (int): The number of future periods to forecast.

#     Returns:
#     list: Forecasted values for the specified number of periods.
#     """
#     # Initialize the forecast list with historical data
#     forecast = list(history)

#     # Generate forecasts
#     for _ in range(periods):
#         # The forecast for the next period is based on the previous value and the model's parameters
#         next_forecast = const + ar1_coeff * forecast[-1]
#         forecast.append(next_forecast)

#     # Return only the forecasted values, not the historical data
#     return forecast[-periods:]


#%%
##### Residual Analysis ######
## Whiteness Chi-square Test ##
from statsmodels.stats.diagnostic import acorr_ljungbox
ljung_box_result = acorr_ljungbox(res_arma, lags=[50])  
print(ljung_box_result)
# Calculate Q and Q-critical
acf_res1 = acf(res_arma, nlags=30)
Q = len(Y_train)*np.sum(np.square(acf_res1[30:]))
DOF = 30 - 1
alfa = 0.05
from scipy.stats import chi2
chi_critical = chi2.ppf(1-alfa, DOF)
print("Q = ", Q)
print("Q* = ", chi_critical)
if Q< chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white ")
    
#%%
# Variance of error
print("Variance of the errors:", arma.resid.var())

# Covariance matrix of the parameters
print("Covariance matrix of the parameters:")
print(arma.cov_params())

#%%
## Mean of Residuals ##
mean_residuals = res_arma.mean()
print("Mean of the residuals:", mean_residuals)
# unbiased, close to 0
#%%
## Confidence Interval ##
conf_int = arma.conf_int()
print(conf_int)
#%%
##### LSTM ######

#%%
