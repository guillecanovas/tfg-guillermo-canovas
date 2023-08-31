"""
@Author: Simona Bernardi
@Date: updated 04/03/2020
"""

# Library of statistics: descriptive statistics
from scipy.stats import describe

# Library of statistics: confidence interval computation
import statsmodels.stats.api as sms
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tools.sm_exceptions import MissingDataError
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

# Modules
from misc.detectors.TEG.discretization import DataSetLoader, EventLogExtractor, AttackInjector
from misc.detectors.TEG.graph_discovery import GraphGenerator
from misc.detectors.TEG.graph_comparison import GraphComparator, GraphHammingDissimilarity, \
    GraphCosineDissimilarity, GraphKLDissimilarity


class TSanalyzer:

    def __init__(self, gts):
        # Graph time series: index Week Week+1 Metric
        self.ts = gts

    def visualizeGraphTS(self, title, xlabel, ylabel):
        plt.figure(figsize=(16, 5), dpi=100)
        plt.plot(self.ts.index, self.ts.Metric, color='tab:red')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.show()

    def storeGraphTS(self, direct, filename):
        # Same file name but in different directory
        filef = direct + filename
        self.ts.to_csv(filef, index=True)

    def decomposeTimeSeries(self, f, title):

        # Multiplicative decomposition
        # Freq: frequency of the timestamps
        result_mul = seasonal_decompose(np.asarray(self.ts['Metric']),
                                        model='multiplicative',
                                        freq=f, extrapolate_trend='freq')

        # Additive composition
        result_add = seasonal_decompose(np.asarray(self.ts['Metric']),
                                        model='additive',
                                        freq=f, extrapolate_trend='freq')
        # Plot
        plt.rcParams.update({'figure.figsize': (10, 10)})
        result_mul.plot().suptitle(title + ' Multiplicative Decompose',
                                   fontsize=22)
        result_add.plot().suptitle(title + 'Additive Decompose', fontsize=22)
        plt.show()

        return

    def testStationarity(self):
        print("----> Is the data stationary ?")

        # ADF tests Null Hypothesis (H0): If failed to be rejected,
        # it suggests the time series has a unit root, meaning it is
        # non-stationary. It has some time dependent structure. Alternate
        # Hypothesis (H1): The null hypothesis is rejected; it suggests the
        # time series does not have a unit root, meaning it is stationary. It
        # does not have time-dependent structure.
        try:
            result_adf = adfuller(self.ts.Metric.values, autolag='AIC')
        except MissingDataError:    # some are inf or nan
            return

        # Normalizing to 11 decimal points
        list_result = list(result_adf)
        list_result[0] = float("{:.11f}".format(result_adf[0]))  # 11 decimal points
        result_adf = tuple(list_result)

        print(f'ADF Statistic: {result_adf[0]}')
        print(f'p-value: {result_adf[1]}')
        print('Critical Values:')
        for key, value in result_adf[4].items():
            print(
                "\t{}: {} - The data is {} stationary with {}% "
                "confidence".format(
                    key, value, "not" if value < result_adf[0] else "",
                    100 - int(key[:-1])))

        # KPSS Test
        # Null Hypothesis (H0): If failed to be rejected, it suggest the
        # time series is stationary
        # Alternate hypothesis (H1): The H0 is rejected, then the time series
        # is not stationary.
        result_kpss = kpss(self.ts.Metric.values, regression='c', nlags='auto')

        # Normalizing to 11 decimal points
        list_result = list(result_kpss)
        list_result[0] = float("{:.11f}".format(result_kpss[0]))  # 11 decimal points
        result_kpss = tuple(list_result)

        print('\nKPSS Statistic: %f' % result_kpss[0])
        print('p-value: %f' % result_kpss[1])
        print('Critical Values:')
        for key, value in result_kpss[3].items():
            print(
                "\t{}: {} - The data is {} trend-stationary with {}% "
                "confidence".format(
                    key, value, "" if value < result_kpss[0] else "not",
                    100 - float(key[:-1])))

        # Easier to tests
        return result_adf, result_kpss

    def plotsPACF(self, lags, title):
        # Autocorrelation and partial autocorrelation
        plt.rcParams.update({'figure.figsize': (16, 5), 'figure.dpi': 120})
        plot_acf(self.ts.Metric, alpha=0.1, lags=lags)
        plt.title("ACF plot --" + title)
        plt.show()
        plot_pacf(self.ts.Metric, lags=lags)
        plt.title("PACF plot --" + title)
        plt.show()
        return

    def differencesTS(self, order, nlags):
        # Difference the series and see how the autocorrelation
        plt.rcParams.update({'figure.figsize': (16, 5), 'figure.dpi': 120})
        plot_acf(self.ts.Metric.diff(order).dropna(), lags=nlags)
        plt.title("ACF -- " + str(order) + " order differencing --")
        plt.show()
        plot_pacf(self.ts.Metric.diff(order).dropna(), lags=nlags)
        plt.title("PACF -- " + str(order) + " order differencing --")
        plt.show()
        return

    def buildAutoARIMA(self, nItems, title):
        # Training set
        training_period = round(nItems * 0.8)
        train = self.ts.Metric[:training_period]

        # Looking for the best params with auto_arima -- time expensive
        model = pm.auto_arima(train,
                              test='adf',  # type of tests to determine d
                              start_p=1,  # initial p value
                              start_q=0,  # initial q value
                              max_p=5,  # max p value
                              max_q=5,  # max q value
                              m=1,  # period for seasonal differencing
                              d=None,
                              # order of differencing determined by the tests
                              seasonal=False,  # seasonal tests on
                              start_P=0,
                              start_Q=0,
                              D=0,
                              trace=True,
                              error_action='warn',
                              suppress_warnings=True,
                              stepwise=True)

        # Print the information about the fitted model
        print(model.summary())

        # Diagnostic on the residuals
        model.plot_diagnostics(figsize=(12, 5))
        plt.title("Model diagnostic:" + title)
        plt.show()

        return model, training_period

    def makePrediction(self, model, training_period, title):
        train = self.ts.Metric[:training_period]
        test = self.ts.Metric[training_period:]

        pred, conf = model.predict(n_periods=test.size, return_conf_int=True)

        # Plotting the training set, the testing set, the forecasting and
        # conf.levels
        plt.figure(figsize=(12, 5), dpi=100)
        plt.plot(train, label='training')
        plt.plot(test, label='actual')
        plt.plot(test.index, pred, label='forecast')
        plt.fill_between(test.index, conf[:, 0], conf[:, 1],
                         color='k', alpha=.05)
        plt.title(title + '-- Forecast vs/ Actual')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()
        '''
        #Compute MAPE - Mean Absolute Percentage Error - metric
        mape=tsa.forecast_accuracy(pred.predicted_mean.values, tests.values)
        print("Forecast accuracy: -- Mape: ", mape)  
        '''


class StatChecker:
    # Collects the statistical checkers needed to detect outliers
    # Confidence interval for the mean difference and relative error

    def __init__(self):
        # Precondition: just two experiments
        self.experiment = []
        # Confidence interval
        self.ci = []
        # Relative error
        self.re = None

    def addExperiment(self, exp):
        # Precondition: experiment 1 and 2 need to have the same caseID, metric
        self.experiment.append(exp)

    def getConfInt(self, conflevel):
        # Precondition: experiment 1 and 2 need to have the same number of items
        nItems = len(self.experiment[0].bset)
        first = []
        second = []

        for i in range(nItems):
            first.append(self.experiment[0].bset[i].metricValue)
            second.append(self.experiment[1].bset[i].metricValue)

        # Class for two sample comparison
        cm = sms.CompareMeans(sms.DescrStatsW(first), sms.DescrStatsW(second))
        # Computes the confidence level: conf. level for the confidence
        # interval, coverage is 1-alpha, Different standard deviation:  Welsh
        # ttest with Satterthwait degrees of freedom is used
        self.ci = cm.tconfint_diff(alpha=conflevel, alternative='two-sided',
                                   usevar='unequal')

    def isStatSignificant(self):
        # In case zero does not belong to the interval
        if 0.0 < self.ci[0] or 0.0 > self.ci[1]:
            return True
        else:
            return False

    def getRelErr(self):
        # Pre-condition: the time-evolving graphs of the normal behavior
        # correspond to the first experiment
        meanNormBehavior = float(self.experiment[0].statQualifiers.mean)
        if meanNormBehavior != 0.0:
            self.re = min(abs(self.ci[0] / meanNormBehavior),
                          abs(self.ci[1] / meanNormBehavior))


class Experiment:
    # Experiment container: caseID and (distance, similarity) metric
    def __init__(self, idmeter, metric):
        self.bset = []  # to be a list of basic experiments
        self.caseID = idmeter
        self.metric = metric
        # Experiment with integrity attack common to all basic experiments
        self.attackType = None
        self.paramA = None
        self.paramB = None
        # Statistics: nsample,min,max,variance,skewness,kurtosis
        self.statQualifiers = None

    def setAttack(self, kind, paramA, paramB):
        self.attackType = kind
        self.paramA = paramA
        self.paramB = paramB

    def addBasicExperiment(self, filename1, filename2):
        self.bset.append(BasicExperiment(filename1, filename2, self))

    def getStats(self):
        # array collecting the metric values
        a = []
        for i in range(len(self.bset)):
            a.append(self.bset[i].metricValue)

        self.statQualifiers = describe(a)


class BasicExperiment:
    # Basic experiment: two input data sets
    def __init__(self, filename1, filename2, experiment):
        self.fname = []  # original dataset files
        self.fname.append(filename1)
        self.fname.append(filename2)
        # self.fnameEL = [] #event log files
        # self.fnameG = [] #graph files
        self.exp = experiment
        self.dsetloader = []
        self.extr = []
        self.gr = []
        self.grcomp = None
        self.metricValue = None

        for i in range(2):
            self.dsetloader.append(
                DataSetLoader())  # dataset at time period T_i,T_{i+1}
            self.dsetloader[i].load(self.fname[i], self.exp.caseID)

    def generateEventLogs(self, minValue, kw, n_bins):
        for i in range(2):
            self.extr.append(EventLogExtractor(minValue, kw, n_bins))
            dataset = self.dsetloader[i].dataset
            # caseID = self.exp.caseID
            self.extr[i].extractEL(dataset)

        return 0

    def generateManipulatedEventLogs(self, minValue, kw, n_bins):
        # First: normal data
        self.extr.append(EventLogExtractor(minValue, kw, n_bins))
        dataset = self.dsetloader[0].dataset
        # caseID = self.exp.caseID
        self.extr[0].extractEL(dataset)
        # Second: manipulated data
        self.extr.append(AttackInjector(minValue, kw, n_bins))
        kind = self.exp.attackType
        para = self.exp.paramA
        parb = self.exp.paramB
        self.extr[1].getSyntheticEL(dataset, kind, para, parb)

    def generateGraphs(self):
        for i in range(2):
            self.gr.append(GraphGenerator())
            eventlog = self.extr[i].eventLog
            # caseID = self.exp.caseID
            self.gr[i].generateGraph(eventlog)
        return 0

    def computeGraphsMetric(self):

        # Generate a new graph comparator
        self.grcomp = GraphComparator(self.gr[0].graph,self.gr[1].graph)
        #self.grcomp.graph1 = 
        #self.grcomp.graph2 = 
        # Graph normalization
        self.grcomp.normalizeGraphs()
        # Compute the distance or similarity corresponding to the "metric"
        metric = self.exp.metric
        if metric == "Hamming":
            self.grcomp.__class__ = GraphHammingDissimilarity
            self.metricValue = self.grcomp.compareGraphs()
        elif metric == "Cosine":
            self.grcomp.__class__ = GraphCosineDissimilarity
            self.metricValue = self.grcomp.compareGraphs()
        elif metric == "KLD":
            self.grcomp.__class__ = GraphKLDissimilarity
            self.metricValue = self.grcomp.compareGraphs()


class Scenario:
    # Create a new experiment and execute it by generating the time-evolving
    # graphs from the input dataset by selecting a given idMeter. Produces a
    # set of graph pairs and the corresponding metricValue

    def __init__(self, idmeter, metric, kind=None, paramA=None, paramB=None):
        # Initialization and creation of a new experiment
        self.experiment = Experiment(idmeter, metric)
        if kind is not None:
            self.experiment.setAttack(kind, paramA, paramB)

    def executeSteps(self, i, minValue, kw=5, n_bins=24):
        # Classification step:
        # Extract the two event logs
        if self.experiment.attackType is None:
            self.experiment.bset[i].generateEventLogs(minValue, kw, n_bins)
        else:
            self.experiment.bset[i].generateManipulatedEventLogs(minValue, kw, n_bins)

        # Generation of the two graphs
        self.experiment.bset[i].generateGraphs()

        # Compute the metric by comparing the two graphs
        self.experiment.bset[i].computeGraphsMetric()

        return self.experiment.bset[i].metricValue
