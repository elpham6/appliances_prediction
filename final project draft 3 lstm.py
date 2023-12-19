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
# Cross-Validation
from sklearn.model_selection import TimeSeriesSplit

# Fit the model
model = sm.OLS(Y_train, sm.add_constant(X_selected_train)).fit()
# One-step ahead prediction
predictions = model.predict(sm.add_constant(X_selected_test))
# Model Evaluation
mse = mean_squared_error(Y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, predictions)
print(f"MSE: {mse}, RMSE: {rmse}, R-squared: {r2}")
print(model.summary())


# ACF of Residuals and Q-Value
residuals = Y_test - predictions
ACF_PACF_Plot(residuals, 50)
plt.show()

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
#### Regressions w/ Cross-Validation ####
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)  

# Remove features by backwards selection
X_selected = X[selected_features]

# Lists to store results of each fold
mse_scores = []
r2_scores = []
rmse_scores = []
for train_index, test_index in tscv.split(X_selected):
    # Splitting the data
    X_train_cv, X_test_cv = X_selected.iloc[train_index], X_selected.iloc[test_index]
    Y_train_cv, Y_test_cv = Y.iloc[train_index], Y.iloc[test_index]
    
    # Fit the model
    model_cv = sm.OLS(Y_train_cv, sm.add_constant(X_train_cv)).fit()
    predictions_cv = model_cv.predict(sm.add_constant(X_test_cv))
    
    # Evaluate the model
    mse_cv = mean_squared_error(Y_test_cv, predictions_cv)
    r2_cv = r2_score(Y_test_cv, predictions_cv)
    rmse_cv = np.sqrt(mse_cv)
    # Append scores
    mse_scores.append(mse_cv)
    r2_scores.append(r2_cv)
    rmse_scores.append(rmse_cv)

# Calculate average MSE and R2 across all folds
ols_mse = np.mean(mse_scores)
ols_r2 = np.mean(r2_scores)
ols_rmse = np.mean(rmse_scores)
print(f"Average MSE: {ols_mse}, Average R-squared: {ols_r2}, Average RMSE: {ols_rmse}")


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
# So probably AR(1)
#%%
### Estimate Parameters ###
p = 1 # AR order
q = 0  # MA order
arma = sm.tsa.ARIMA(Y_train, order=(p, 0, q)).fit()
print(arma.summary())
res_arma = arma.resid
ACF_PACF_Plot(res_arma, 50)

#%%
##### Forecast Function ######
# Using SARIMAX
def forecast_function(history, order, seasonal_order, forecast_periods):
    """
    """
    # Fit the SARIMA model
    model = sm.tsa.SARIMAX(history, order=order, seasonal_order=seasonal_order, 
                           enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    
    # Make forecast
    forecast = model_fit.forecast(steps=forecast_periods)
    
    return forecast

order = (1, 0, 0)
seasonal_order = (0, 0, 0, 0)
forecast_periods = len(Y_test)

sarima_forecast_values = forecast_function(Y_train, order, seasonal_order, forecast_periods)
print(sarima_forecast_values)
res_fc = Y_test - sarima_forecast_values
res_fc_var = np.var(res_fc)

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
# Residuals Variance
# Forecast Variance (get forcast errors, then calc variance)
print("Forecast Variance: ", res_fc_var)
#%%
## Model Simplification ##
root_check = np.roots([1, 0.7569])
print("Final coefficient(s): ", root_check[0])

#%%
#### LSTM ####
#%%
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import EarlyStopping

# early_stopping = EarlyStopping(monitor='val_loss', patience=10, 
#                                min_delta=0.001, mode='min', verbose=1)
# # Scaling the features
# scaler = MinMaxScaler(feature_range=(0, 1))
# Y_scaled = scaler.fit_transform(Y.values.reshape(-1,1))

# # Function to create a dataset for LSTM
# def create_dataset(data, time_step=1):
#     X, Y = [], []
#     for i in range(len(data) - time_step - 1):
#         a = data[i:(i + time_step), 0]
#         X.append(a)
#         Y.append(data[i + time_step, 0])
#     return np.array(X), np.array(Y)

# # Reshape into [samples, time steps, features]
# time_step = 200
# X_lstm, Y_lstm = create_dataset(Y_scaled, time_step)
# X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

# # Splitting data into train and test
# train_size = int(len(Y_lstm) * 0.80)
# X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
# Y_train, Y_test = Y_lstm[:train_size], Y_lstm[train_size:]

# model = Sequential()
# model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
# model.add(LSTM(50, return_sequences=False))
# model.add(Dense(1))

# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
#           epochs=30, batch_size=64,
#           callbacks=[early_stopping])
# train_predict = model.predict(X_train)
# test_predict = model.predict(X_test)

# # Inverting back to original scale
# train_predict = scaler.inverse_transform(train_predict)
# test_predict = scaler.inverse_transform(test_predict)
# plt.figure(figsize=(10, 6))
# plt.plot(Y, label='Original Data')
# plt.plot(np.arange(time_step, time_step + len(train_predict)), train_predict, label='Train Predict')
# plt.plot(np.arange(len(Y) - len(test_predict), len(Y)), test_predict, label='Test Predict')
# plt.legend()
# plt.show()

# #%%
# # Assuming you used a simple train-test split
# train_size = len(Y_train)  # The number of data points in the training set

# # Assuming n is the number of time steps used by your LSTM model
# n = time_step

# # Adjust the test index to align with the predictions
# test_index_aligned = clean_df.index[train_size + n:]  # Skip the first 'n' timestamps in the test data
# test_predict_flat = test_predict.flatten()

# # Ensure the lengths are the same
# if len(test_index_aligned) != len(test_predict_flat):
#     # If lengths are different, adjust the length of test_predict_flat to match test_index_aligned
#     test_predict_flat = test_predict_flat[:len(test_index_aligned)]


# plt.figure(figsize=(15, 6))
# plt.plot(clean_df.index, clean_df['Appliances'], label='Original Data')
# plt.plot(clean_df.index[:train_size], train_predict.flatten(), label='Train Predict')
# plt.plot(test_index_aligned, test_predict_flat, label='Test Predict')
# plt.legend()
# plt.show()


# #%%
# # If you've scaled your data, you need to invert the scaling for both the predictions and the actual values
# Y_test_inverted = scaler.inverse_transform(Y_test.reshape(-1, 1))
# test_predict_inverted = scaler.inverse_transform(test_predict)

# # Flatten arrays to 1D if they are in 2D
# Y_test_inverted = Y_test_inverted.flatten()
# test_predict_inverted = test_predict_inverted.flatten()

# from sklearn.metrics import mean_squared_error
# import numpy as np

# rmse = np.sqrt(mean_squared_error(Y_test_inverted, test_predict_inverted))
# print("RMSE: ", rmse)

# from sklearn.metrics import r2_score

# r_squared = r2_score(Y_test_inverted, test_predict_inverted)
# print("R-squared: ", r_squared)

# %%
#### Final Model Selection ####
metrics = {}