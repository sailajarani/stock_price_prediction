import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from pmdarima.arima import auto_arima



df = web.DataReader('AMZN',data_source='yahoo',start='2012-01-01',end='2021-04-26')
print(df.head())
print(df.shape)
plt.figure(figsize=(12,6))
plt.title('closing price')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('close price',fontsize=18)
plt.show()
print(df.info())
df_close = df['Close']
print(df_close.head())


def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12,center=False).mean()
    rolstd = timeseries.rolling(window=12,center=False).std()
     #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')

    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
test_stationarity(df_close)

df_close_log = np.log(df_close)
#print(type(df_close_log))
df_close_log_diff = df_close_log - df_close_log.shift()
plt.plot(df_close_log_diff)
plt.show()
df_close_log_diff = df_close_log_diff.dropna()
test_stationarity(df_close_log_diff)

#ACF and PACF
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import statsmodels.api as sm

lag_acf = acf(df_close_log_diff, nlags=10,) #Autocorrelation plot for 10 lags
lag_pacf = pacf(df_close_log_diff, nlags=10, method='ols') #Partial autocorrelation plot for 10 lags

plt.figure(figsize=(13,6))
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_close_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_close_log_diff)),linestyle='--',color='gray')
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_close_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_close_log_diff)),linestyle='--',color='gray')
plt.show()


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_close_log_diff,lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_close_log_diff,lags=40,ax=ax2)

plt.show()



# Define the p, d and q parameters to take any value between 0 and 3


#Length of df_close is 2343

#model

model = ARIMA(df_close_log,order=(1,1,1))
results_ARIMA = model.fit(disp=-1)
print(results_ARIMA.summary())

plt.plot(df_close_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-df_close_log_diff)**2))
plt.show()


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())

size = int(len(df_close_log) - 10)
train, test = df_close_log[0:size], df_close_log[size:len(df_close_log)]
history = [x for x in train]
predictions = list()
print('Printing Predicted vs Expected Values...')
print('\n')
original = []
predicted = []
for t in range(len(test)):
    model = ARIMA(history, order=(1,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    print(obs)
    history.append(obs)
    original.append(np.exp(obs))
    predicted.append(np.exp(yhat))

print('predicted=%f, expected=%f' % (np.exp(yhat), np.exp(obs)))
final_df = pd.DataFrame(list(zip(original,predicted)),columns=['original','predicted'])
print(final_df)

error = mean_squared_error(test, predictions)
print('\n')
print('Printing Mean Squared Error of Predictions...')
print('Test MSE: %.6f' % error)
predictions_series = pd.Series(predictions, index = test.index)
print(np.exp(predictions_series))
fig, ax = plt.subplots()
ax.set(title='stock prediction', xlabel='Date', ylabel='stock value')
ax.plot(df_close[-60:], 'o', label='observed')
ax.plot(np.exp(predictions_series), 'g', label='rolling one-step out-of-sample forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')
plt.show()
