from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess

def extract_date(date_str):
    """
    Extract month, date, and year from a date string formatted as m/d/yyyy.
    - date_str (str): date string in m/d/yyyy

    Returns a tuple: (month, date, year)
    """

    # Split the date string using '/'
    parts = date_str.split('/')

    # Extract the individual components
    month = int(parts[0])
    date = int(parts[1])
    year = int(parts[2])

    return month, date, year


def Plot_Rolling_Mean_Var():
    pass

def ADF_Cal(x):
    """
    Calculate ADF test given a list/array of values x

    Returns  ADF stat, p-value and critical values.
    """
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def kpss_test(timeseries):
    """
    Calculate KPSS test given a list/array of timeseries data "timeseries"

    Returns KPSS stat, p-value and critical values
    """
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)



def difference(data, order=1):
    """
    Perform differencing on data
    - data: A pandas Series or DataFrame
    - order: The number of order to difference the data by (default = 1)

    Returns 'differenced_data': pandas Series or DataFrame containing the differenced data
    """
    # for pd arrays
    if isinstance(data, pd.Series):
        # differencing order must be at least 1
        # throw error otherwise
        if order <= 0:
            raise ValueError("Order must be greater than 0.")

        # exclude data points to avoid differencing invalid values
        # depending on the differencing order
        differenced_data = pd.Series(index=data.index[order:])
        for i in range(order, len(data)):
            # loop over each value to perform differencing
            differenced_data[data.index[i]] = data.iloc[i] - data.iloc[i - order]
        return differenced_data
    # for dataframes (in case of differencing for multiple columns)
    elif isinstance(data, pd.DataFrame):
        if order <= 0:
            raise ValueError("Order must be greater than 0.")
        # make a copy of the data
        differenced_data = data.copy()
        for col in data.columns:
            for i in range(order, len(data)):
                differenced_data.at[i, col] = data.at[i, col] - data.at[i - order, col]

        return differenced_data.iloc[order:]
    # give error if data type is not a pd list or df
    else:
        raise ValueError("Input data must be a pandas Series or DataFrame.")

# write test cases later
# for pd lists
# for dataframe

def Cal_rolling_mean_var(data, column_name):
    rolling_means = []
    rolling_variances = []

    for i in range(1, len(data)+1):
        # base case
        # mean = first value, variance = 0
        if i == 1:
            rolling_means.append(data[column_name].head(1))
            rolling_variances.append(0)
        else:
            # load the first 'i' observations
            rolling_data = data[column_name].head(i)

            # calculate mean and variance
            mean = rolling_data.mean()
            variance = rolling_data.var()

            rolling_means.append(mean)
            rolling_variances.append(variance)

    # create a time axis for the x-axis to match no. of values
    time_axis = range(1, len(data) + 1)
    # set y-axis lower bound to 0
    y_min = 0

    # plot the rolling mean and rolling variance
    plt.figure(figsize=(10, 6))
    # rolling mean
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, rolling_means, label='Rolling Mean')
    plt.title(f'Rolling Mean - {column_name}')
    plt.xlabel('Number of Samples')
    plt.ylabel('Magnitude')
    plt.legend(loc = 'lower right')

    # rolling variance
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, rolling_variances, label='Varying Variance')
    plt.title(f'Rolling Variance - {column_name}')
    plt.xlabel('Number of Samples')
    plt.ylabel('Magnitude')
    plt.ylim(y_min)
    plt.legend(loc = 'lower right')

    plt.tight_layout()
    plt.show()

def log_transform_series(data):
    """
    Log transform time series data
    - data: A pandas Series or DataFrame containing the time series data

    Returns
    - log_transformed_data: pandas Series or DataFrame containing the log-transformed data
    """
    if isinstance(data, pd.Series):
        log_transformed_data = np.log(data)
        return log_transformed_data
    elif isinstance(data, pd.DataFrame):
        # Apply the log transformation to each column of the DataFrame
        log_transformed_data = data.apply(np.log)
        return log_transformed_data
    else:
        raise ValueError("Input data must be a pandas Series or DataFrame.")


def compute_and_plot_acf(y, n_lags=50):
    mean_y = np.mean(y)
    T = len(y)

    def autocorr_formula(y, r):
        numerator = sum([(y[t] - mean_y) * (y[t-r] - mean_y) for t in range(r, T)])
        denominator = sum([(yt - mean_y) ** 2 for yt in y])
        return numerator / denominator

    lags = list(range(-n_lags + 1, 0)) + list(range(0, n_lags))
    autocorr_values = [autocorr_formula(y, abs(r)) for r in lags]

    plt.stem(lags, autocorr_values, basefmt=" ", markerfmt='ro')
    confidence_band = 1.96 / np.sqrt(T)
    plt.fill_between([-n_lags + 1, n_lags-1], -confidence_band, confidence_band, color='blue', alpha=0.3, zorder=0)
    plt.axhline(y=0, color='black', linewidth=0.8)
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    

import pandas as pd

def seasonal_difference(data, interval):
    """
    Perform seasonal differencing on data.
    - data: A pandas Series or DataFrame
    - interval: The seasonal period

    Returns 'seasonally_differenced_data': pandas Series or DataFrame containing the seasonally differenced data
    """
    if interval <= 0:
        raise ValueError("Interval must be greater than 0.")
    if len(data) < interval:
        raise ValueError("Length of data must be greater than the seasonal interval.")

    # Perform seasonal differencing
    seasonally_differenced_data = data.diff(periods=interval).dropna()
    return seasonally_differenced_data

def generate_arma():
    """
    Generate ARMA Process
    Takes Input:
    n: number of data samples
    mean = WN mean
    variance = WN variance
    ar_order, ma_order: order of AR, MA
    ar_coeff, ma_coeff: coefficients of AR, MA
    """
    np.random.seed(6313)
    n = int(input("Enter the number of data samples: "))
    mean = float(input("Enter the mean of white noise: "))
    variance = float(input("Enter the variance of the white noise: "))
    ar_order = int(input("Enter AR order: "))
    ma_order = int(input("Enter MA order: "))

    ar_coeff = [float(input(f"Enter the coefficient for AR a{i}: ")) for i in range(1, ar_order + 1)]
    ma_coeff = [float(input(f"Enter the coefficient for MA b{i}: ")) for i in range(1, ma_order + 1)]

    ar = np.r_[1, np.array(ar_coeff)]
    ma = np.r_[1, np.array(ma_coeff)]
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    #y_mean = mean*(1+np.sum(ma_coeff))/(1+np.sum(ar_coeff))
    arma_data = arma_process.generate_sample(nsample=n, scale=np.sqrt(variance)) #+ y_mean

    return arma_data, arma_process

def calc_gpac(acf, j, k):
    """ 
    Calculate the coefficient for ARMA model at AR order k
    """
    # throw error if j = 0 and k = 0
    if k == 0:
        raise ValueError("k must be greater than 0 for GPAC calculations.")
    # create k x k matrices for numerator and denominator
    num = np.zeros((k, k))
    den = np.zeros((k, k))
    last_lag = j # initate lag of last column
    # loop through each row and column of matrix 
    # calculate and append the autocorr. at the respective lag
    for row in range(k):
        for col in range(k):
            # calculate lags for numerator and denominator
            if col == k-1: # if last column
                last_lag+=1
                lag_num = abs(last_lag)
                lag_den = abs(j + row - col)
                # access the acf value
                r1 = acf[lag_den]
                r2 = acf[lag_num]
                # assign the values to the matrices
                den[row][col] = r1
                num[row][col] = r2
            else:
                lag_den = abs(j + row - col)
                r1 = acf[lag_den]
                den[row][col] = r1
                num[row][col] = r1

    # calculate determinants of the matrices
    det_num = np.linalg.det(num)
    det_den = np.linalg.det(den)
    if det_den == 0:
        return float('inf')

    # compute GPAC value
    phi = det_num / det_den
    if abs(phi) < 0.000001:
        phi = 0
    return phi

# Create a GPAC table

def create_gpac_table(acf, max_j=7, max_k=7):
    """
    Creates a GPAC table for given data with specified range of j and k values

    """
    # # Synthesize data and ARMA process according to j and k
    # arma_data, arma_process = generate_arma()
    # # calculate theoretical ACF
    # acf = arma_process.acf(lags=max_j+max_k+1)
    # Initialize an empty table structure
    gpac_table = np.zeros((max_j, max_k-1))

    # Fill the table with GPAC values
    for k in range(1, max_k):
        for j in range(max_j):  # k starts from 1
            gpac_table[j][k - 1] = calc_gpac(acf, j, k)

    # Convert to a pandas DataFrame 
    gpac_df = pd.DataFrame(gpac_table)
    # round to 2 decimals
    gpac_df = gpac_df.round(2)
    # start k column from 1
    gpac_df.columns = range(1, max_k)
    # plot the table
    plt.figure(figsize=(10, 8))
    sns.heatmap(gpac_df, annot=True, fmt=".2f")
    plt.title('Generalized Partial Autocorrelation (GPAC) Table')
    plt.show()

    return gpac_df

def ACF_PACF_Plot(y,lags):
 acf = sm.tsa.stattools.acf(y, nlags=lags)
 pacf = sm.tsa.stattools.pacf(y, nlags=lags)
 fig = plt.figure()
 plt.subplot(211)
 plt.title('ACF/PACF of the raw data')
 plot_acf(y, ax=plt.gca(), lags=lags)
 plt.subplot(212)
 plot_pacf(y, ax=plt.gca(), lags=lags)
 fig.tight_layout(pad=3)
 plt.show()
