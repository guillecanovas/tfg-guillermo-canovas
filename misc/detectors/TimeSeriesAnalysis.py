"""
@Author: Simona Bernardi
@Date: Updated on January 2, 2020
    This script follows the guideline "Time Series Analysis in Python - A comprehensive
    Guide with Examples - ML+.pdf" to analyse the consumption behavior of a customer
    provided in input (meterID)
    The input dataset is the one in "data_all_filtered" directory (provided in input)
"""

import sys
import os
import numpy as np
import pandas as pd
# from pandas.plotting import autocorrelation_plot
# import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
# from statsmodels.customer_analysis_results.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.customer_analysis_results.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# import pmdarima as pm

nObs = 336  # number of readings in a week


class TSanalyzer:

    def __init__(self):
        # dataframe with all the readings: columns ['ID','DT','Usage']
        self.df = pd.DataFrame()

    def getDataSetFiles(self, dpath, prefixFilename):
        # This part is very dependent on how the dataset is structured
        # Loads the list of filenames in my_files array and then sort the files
        # according to the number of week
        dirFiles = os.listdir(dpath)
        myfiles = []
        for files in dirFiles:
            if files.startswith(prefixFilename):
                pathfile = (dpath + '/') + files
                myfiles.append(pathfile)
                myfiles.sort(key=lambda x: int(x.split()[-1]))

        return myfiles

    def loadTimeSeries(self, files, firstWeek, lastWeek, meterID):
        i = 0  # No. of week counter
        for file in files[firstWeek:lastWeek + 1]:
            print("Reading File: ", file)
            dset = pd.read_csv(file)  # load as a pd.dataFrame
            dset = dset[dset.ID == meterID]
            # Adding the week in the last column (4) with the no. of week
            dset.insert(2, 'Week', i)
            # print("Observation statistics: ", dset.head())
            self.df = pd.concat([self.df, dset])
            i += 1
        # Set DT column as the index
        # self.df.index = self.df.DT
        self.df.reset_index(inplace=True)
        return

    def visualizeTimeSeries(self, title, xlabel, ylabel):

        plt.figure(figsize=(16, 5), dpi=100)
        plt.plot(self.df.index, self.df.Usage, color='tab:red')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.show()

        return

    def decomposeTimeSeries(self):

        # Multiplicative decomposition
        # Freq: seasonality frequency = frequency of the timestamps
        result_mul = seasonal_decompose(np.asarray(tsa.df['Usage']),
                                        model='multiplicative',
                                        freq=nObs, extrapolate_trend='freq')

        # Additive composition
        result_add = seasonal_decompose(np.asarray(tsa.df['Usage']),
                                        model='additive',
                                        freq=nObs, extrapolate_trend='freq')
        # Plot
        plt.rcParams.update({'figure.figsize': (10, 10)})
        result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
        result_add.plot().suptitle('Additive Decompose', fontsize=22)
        plt.show()

        return result_mul, result_add

    def plotMeansVars(self):
        # Compute the means and vars on week basis
        dfMeanWeek = tsa.df.groupby(tsa.df.Week)['Usage'].mean()
        dfVarWeek = tsa.df.groupby(tsa.df.Week)['Usage'].var()

        # Plot the means/vars
        plt.figure(figsize=(16, 5), dpi=100)
        plt.plot(dfMeanWeek.index, dfMeanWeek, color='tab:red')
        plt.plot(dfVarWeek.index, dfVarWeek, color='tab:blue')

        plt.gca().set(title='Customer consumption stats by week',
                      xlabel='Weeks', ylabel='Mean/Var')
        plt.show()

        return

    def stationarityTests(self, ts):
        print("----> Is the data stationary ?")

        # ADF tests
        # Null Hypothesis (H0): If failed to be rejected, it suggests the
        # time series has a unit root, meaning it is non-stationary.
        # It has some time dependent structure.
        # Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests
        # the time series does not have a unit root, meaning it is stationary.
        # It does not have time-dependent structure.
        result = adfuller(ts, autolag='AIC')
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(
                "\t{}: {} - The data is {} stationary with {}% confidence".format(
                    key, value, "not" if value < result[0] else "",
                    100 - int(key[:-1])))

        # KPSS Test
        # Null Hypothesis (H0): If failed to be rejected, it suggest the
        # time series is stationary
        # Alternate hypothesis (H1): The H0 is rejected, then the time series
        # is not stationary.
        result = kpss(ts, regression='c')
        print('\nKPSS Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[3].items():
            print(
                "\t{}: {} - The data is {} trend-stationary with {}% confidence".format(
                    key, value, "" if value < result[0] else "not",
                    100 - float(key[:-1])))


    def detrend(self, trend_add, trend_mult):
        # Remove the trend components computed from the decompositions
        detrended_add = tsa.df['Usage'].values - trend_add
        detrended_mul = tsa.df['Usage'].values / trend_mult

        # ADF and KPSS Tests of the detrended TS - additive
        plt.figure(figsize=(16, 5), dpi=100)
        plt.plot(detrended_add)
        plt.title("Detrended TS -- additive")
        plt.show()
        print("Tests for the detrended TS - additive")
        tsa.stationarityTests(detrended_add)

        # ADF and KPSS Tests of the detrended TS - multiplicative
        plt.figure(figsize=(16, 5), dpi=100)
        plt.plot(detrended_mul)
        plt.title("Detrended TS -- multiplicative")
        plt.show()
        print("Tests for the detrended TS - multiplicative")
        tsa.stationarityTests(detrended_mul)

        return

    def deseasonalize(self, season_add, season_mul):
        # Remove the seasonal components computed from the decompositions

        deseasonalize_add = tsa.df['Usage'].values - season_add
        deseasonalize_mul = tsa.df['Usage'].values / season_mul

        # ADF and KPSS Tests of the deseasonalized TS - additive
        plt.figure(figsize=(16, 5), dpi=100)
        plt.plot(deseasonalize_add)
        plt.title("Deseasonalized TS -- additive")
        plt.show()
        print("Tests for the deseasonalized TS - additive")
        tsa.stationarityTests(deseasonalize_add)

        # ADF and KPSS Tests of the deseasonalized TS - multiplicative
        plt.figure(figsize=(16, 5), dpi=100)
        plt.plot(deseasonalize_mul)
        plt.title("Deseasonalized TS -- multiplicative")
        plt.show()
        print("Tests for the deseasonalized TS - multiplicative")
        tsa.stationarityTests(deseasonalize_mul)

    def forecast_accuracy(self, forecast, actual):

        mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE

        return mape


if __name__ == '__main__':

    if (len(sys.argv) == 3):
        tsa = TSanalyzer()

        # Load the file names of the dataset files in files given the first
        # param (directory)
        files = tsa.getDataSetFiles(sys.argv[1], "ElectricityDataWeek")

        meterID = int(sys.argv[2])
        firstWeek = 0
        lastWeek = 74
        # Load time series of the meterID into the dataframe attribute "df" of the class
        # A new column is added besides DT,ID,Usage: the week number.
        tsa.loadTimeSeries(files, firstWeek, lastWeek, meterID)

        # Visualization with a plot: all the period
        title = ("Consumption of customer " + str(meterID) +
                 " during the weeks from 0 to " + str(lastWeek))
        tsa.visualizeTimeSeries(title, "Timestamp", "Usage")

        # Time series decomposition (multiplicative/additive)
        res_mult, res_add = tsa.decomposeTimeSeries()

        # Stationarity of time series
        # Strict stationarity: A TS is stationary iff its mean, variance, autocorrelation
        # are constant over time (autocorrelation=correlation of the time series
        # with its previous values)

        # Computing means and vars for each week and plot them
        tsa.plotMeansVars()

        # ADF (linear or difference stationarity) and KPSS (trend stationarity) Tests
        tsa.stationarityTests(tsa.df.Usage.values)

        # Detrend a time series
        # customer_analysis_results.detrend(res_add.trend,res_mult.trend)

        # Deseasonalize a time series
        # customer_analysis_results.deseasonalize(res_add.seasonal,res_mult.seasonal)

        # Autocorrelation and partial autocorrelation
        plt.rcParams.update({'figure.figsize': (16, 5), 'figure.dpi': 120})
        plot_acf(tsa.df.Usage, alpha=0.1, lags=4 * nObs)  # one day
        plt.title("ACF plot -- one day observations")
        plt.show()
        plot_pacf(tsa.df.Usage, lags=4 * nObs)
        plt.title("PACF plot -- one day observations")
        plt.show()

        # Difference the series and see how the autocorrelation
        plt.rcParams.update({'figure.figsize': (16, 5), 'figure.dpi': 120})
        # order Differencing
        order = 1
        plot_acf(tsa.df.Usage.diff(order).dropna(), lags=nObs)
        plt.title("ACF -- " + str(
            order) + " order differencing -- one day observations")
        plt.show()
        plot_pacf(tsa.df.Usage.diff(order).dropna(), lags=nObs)
        plt.title("PACF -- " + str(
            order) + " order differencing -- one day observations")
        plt.show()

        # Training set / testing set

        training_period = round((lastWeek - firstWeek + 1) * 0.8) * nObs

        train = tsa.df.Usage[:training_period]
        test = tsa.df.Usage[training_period:]

        # Build SARIMA model (p,d,q)x(P,D,Q)period
        # Possible time expensive
        model = SARIMAX(train, order=(1, 0, 0), seasonal_order=(2, 1, 0, 12))
        model_fit = model.fit(disp=0)

        # Print the information about the fitted model
        print(model_fit.summary())

        # Diagnostic on the residuals
        model_fit.plot_diagnostics(figsize=(12, 5))
        plt.show()

        # Forecast
        pred = model_fit.get_prediction(start=test.index[0],
                                        end=test.index[-1],
                                        dynamic=False)

        conf = pred.conf_int()  # confidence intervals

        # Plotting the training set, the testing set, the forecasting and conf.levels
        plt.figure(figsize=(12, 5), dpi=100)
        # plt.plot(train, label='training')
        plt.plot(test, label='actual')
        plt.plot(pred.predicted_mean, label='forecast')
        plt.fill_between(conf.index, conf['lower Usage'], conf['upper Usage'],
                         color='k', alpha=.05)
        plt.title('CustomerID ' + str(meterID) + '-- Forecast vs/ Actual')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()

        # Compute MAPE - Mean Absolute Percentage Error - metric
        mape = tsa.forecast_accuracy(pred.predicted_mean.values, test.values)
        print("Forecast accuracy: -- Mape: ", mape)
'''               
        # Looking for the best params with auto_arima -- time expensive
        model = pm.auto_arima(tsa.df.Usage, 
                              tests='adf', #type of tests to determine d
                              start_p=1, #initial p value
                              start_q=0, #initial q value
                              max_p=3, #max p value
                              max_q=3, #max q value
                              m=1, #period for seasonal differencing
                              d=None, #order of differencing determined by the tests
                              seasonal=False, #seasonal tests on
                              start_P=2,
                              start_Q=1,
                              #D=0,
                              trace=True,
                              error_action='warn',
                              suppress_warnings=True,
                              stepwise=True )
        print(model.summary())
        model.plot_diagnostics(figsize=(7,5))
        plt.show()

'''