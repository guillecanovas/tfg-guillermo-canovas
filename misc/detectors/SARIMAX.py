'''
Created on Aug 7, 2020 - Updated on Aug 12, 2020

@author: Simona Bernardi
Analysis of the CER Electricity dataset according to the Badrinath thesis:
- Dataset: data_all_filtered
- The analysis is performed on a single meterID, in a period [initial,final] week
- Workflow of this script:
-- Visualization of the series in the overall period
-- Partition of the training and tests sets (8:2)
-- Autocorrelation plot (to see the seasonality)
-- Build the SARIMAX model with auto_arima (exogenous variables are derived from Fourier transformation
to model the seasonality, since the high frequency of seasonality make the procedure not scalable
(see here: https://robjhyndman.com/hyndsight/longseasonality/)
-- Make prediction on the tests set
-- Print and plot the results of the fitting (state space model results and model summary)
-- Visualization of the predicted series/confidence intervals and tests series
-- Computation of number of false alarms (one of the metrics to be considered in the comparative analysis)

'''

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf #, plot_pacf
import pmdarima as pm
from pmdarima.arima.utils import ndiffs #, nsdiffs
from pmdarima.preprocessing import FourierFeaturizer

nObs = 336 #number of readings in a week
freq = 48  #number of readings in a seasonal cycle: every half-an-hour in a day

class TSanalyzer:
    
    def __init__(self):
        #dataframe with all the readings: columns ['ID','DT','Usage']
        self.df = pd.DataFrame()
    
    def getDataSetFiles(self,dpath,prefixFilename):
    
        #This part is very dependent on how the dataset is structured
        #Loads the list of filenames in myfiles array and then sort the files
        #according to the number of week
        dirFiles = os.listdir(dpath)
        myfiles = []
        for files in dirFiles:
            if files.startswith(prefixFilename):
                pathfile = (dpath + '/') + files
                myfiles.append(pathfile)
                myfiles.sort(key=lambda x:int(x.split()[-1]))
    
        return myfiles
 
    def loadTimeSeries(self, files, firstWeek, lastWeek, meterID):
        
        i=0 #No. of week counter
        for file in files[firstWeek:lastWeek+1]:
            print ("Reading File: ", file)
            dset=pd.read_csv(file) #load as a pd.dataFrame
            dset=dset[dset.ID == meterID]
            #Adding the week in the last column (4) with the no. of week
            dset.insert(2,'Week',i)
            #print("Observation statistics: ", dset.head())
            self.df = pd.concat([self.df,dset])
            i+=1
        #Set DT column as the index
        #self.df.index = self.df.DT 
        self.df.reset_index(inplace=True)   
        return 
    
    def visualizeTimeSeries(self,title,xlabel,ylabel):
        
        plt.figure(figsize=(16,5), dpi=100)
        plt.plot(self.df.index, self.df.Usage, color='tab:red')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.show()
        
        return
       
    def plotMovingAverage(self,terms):

        df_ma = self.df.Usage.rolling(terms, center=True).mean()
          
        #Plot the moving average        
        plt.figure(figsize=(16,5), dpi=100)
        plt.plot(df_ma, color='tab:red')
        plt.gca().set(title='Moving average '+ str(terms), xlabel='obs', ylabel='Mean')
        plt.show()
    
    def plotACF(self,train): 
        
        plt.rcParams.update({'figure.figsize':(16,5), 'figure.dpi':120})
        plot_acf(train, lags=nObs) #one week
        plt.title("ACF plot -- one week observations")
        plt.show()
   
    def buildModel(self,train):
        
        # Precomputing d
        # Estimate the number of differences using an ADF tests
        # Other possibilities: kpss, pp
        n_adf = ndiffs(train, test='adf')  
            
        #Use of exogenous variables to capture seasonality: 
        #the use of seasonality feature is too much time-expensive (not scalable)
        #The parameter values corresponds to the seasonality of the time series: in
        #our case every day (half-an-hour observations: frequency of observations = 2x24)
        
        trans = FourierFeaturizer(freq)
        train,exog=trans.fit_transform(train)

       
        # Looking for the best params with auto_arima -- time expensive!!
        # model(p,d,q)x(P,D,Q)
        model = pm.auto_arima(train, #the time series to be fitted
                              exogenous=exog, #exogenous variables (default None)
                              #start_p=5, #initial p value (default 2)
                              d=n_adf, #estimated a priori (default None)
                              #start_q=0, #initial q value (default 2)
                              #max_p=5, #max p value (default 5)
                              #max_d=2, #max d value (default 2)
                              #max_q=5, #max q value (default 5)
                              #start_P=1, #initial P value (default 1)
                              #D=n_ch, #(default None)
                              #start_Q=1, #initial Q value (default 1)
                              #max_P = 2, #max P value (default 2)
                              #max_D = 1, #max D value (default 1)
                              #max_Q = 2, #max Q value (default 2)
                              #max_order=5, #p+q+P+Q if stepwise=False (default 5)
                              #m=48, #period for seasonal differencing (default 1)
                              seasonal=False, #whether to fit a seasonal ARIMA (default True)
                              #stationary=False, #the time-series is stationary (default False)
                              information_criterion = 'bic', #default 'aic'
                              #alpha=0.05, #level of the tests for testing significance (default 0.05)
                              #tests='adf', #type of tests to determine d (d=None); default 'kpss'
                              #seasonal_test ='ch', #type of tests to determine D (D=None); default 'ocsb'
                              stepwise=True, #follows the Hyndman and Khandakar approach (default True)
                              #n_jobs=1, #number of models to fit in parallel in case stepwise=False (default 1)
                              #start_params=None, #starting params for ARMA(p,q); default None
                              #method='nm', #optimization method (default 'lbfgs')
                              #trend=None, #trend parameter (default None)
                              maxiter=20, #max no. of function evaluations (default 50) 
                              suppress_warnings=True,
                              error_action='warn', #error-handling behavior in case of unable to fit (default 'warn')
                              trace=True #print the status of the fit (default False)
                              #random=False, #random search over a hyper-parameter space (default False)
                              #random_state=None, #the PRNG when random=True
                              #n_fits=10, #number of ARIMA models to be fit, when random=True
                              #return_valid_fits=False, #return all valid ARIMA fits if True (default False)
                              #out_of_sample_size=0, #portion of the data to be hold out and use as validation (default 0)
                              #scoring='mse', #if out_of_sample_size>0, metric to use for validation purposes (default 'mse')
                              #scoring_args=None, #dictionary to be passed to the scoring metric (default None)
                              #with_intercept='auto', #whether to include an intercept term (default 'auto')
                              #sarimax_kwargs=None #arguments to pase to the ARIMA constructor (default None)
                            )
        return model
    
    def predictConsumption(self,test):
               
        #The exogenous variables need to be used also in the forecast
        trans = FourierFeaturizer(freq)
        test, exog = trans.fit_transform(test)
        
         
        pred, conf= model.predict(n_periods=exog.shape[0], exogenous=exog, 
                                  return_conf_int=True, alpha=0.05)      
        
        return pred, conf
    
    def plotPredictions(self,training_period,pred,conf,test):
        
        index_of_pred = tsa.df.index[training_period:]

        #Make series for plotting purpose
        fitted_series = pd.Series(pred, index=index_of_pred)
        lower_series = pd.Series(conf[:, 0], index=index_of_pred)
        upper_series = pd.Series(conf[:, 1], index=index_of_pred)
                 
        plt.figure(figsize=(12,5),dpi=100)
        plt.plot(test, label='actual')
        plt.plot(fitted_series,label='forecast')
        plt.fill_between(lower_series.index, lower_series, upper_series, color='k',alpha=.05)
        plt.title('CustomerID ' + str(meterID)  + '-- Forecast vs/ Actual')
        plt.legend(loc='upper left',fontsize=8)
        plt.show()

   
    def computeOutliers(self,pred,conf,actual):
        #Pre: pred.size = actual.size = conf.shape[0]
        #Setting a counter vector to zero
        n_out = np.zeros(pred.size)
        #Count if actual is outside of the confidence interval
        n_out = np.where(((actual < conf[:,0]) | (actual > conf[:,1])) 
                          ,n_out+1,n_out)

        return np.sum(n_out)


if __name__ == '__main__':
    
    '''
    args: 
    sys.argv[1]: data_all_filtered folder
    sys.argv[2]: meterID
    sys.argv[3]: first week    
    sys.argv[4]: last week
    '''
     
    if (len(sys.argv) == 5):
        
        tsa = TSanalyzer()
        
        #Load the file names of the dataset files in files given the first 
        #param1 (directory): electricity/data_all_filtered/
        files = tsa.getDataSetFiles(sys.argv[1],"ElectricityDataWeek")
        
        #param2: meterID 
        meterID = int(sys.argv[2])
        
        #param3 and param4: first and last weeks
        firstWeek = int(sys.argv[3])
        lastWeek = int(sys.argv[4])
        
        #Load time series of the meterID in memory
        #A new column is added besides DT,ID,Usage: the week number.
        tsa.loadTimeSeries(files, firstWeek, lastWeek, meterID)
        
        
        #Visualization with a plot: all the period
        title = ("Consumption of customer " + str(meterID) + 
                 " during the weeks from " + str(firstWeek) + " to " + str(lastWeek))
        tsa.visualizeTimeSeries(title,"Timestamp","Usage")

        #Training set / testing set
        percentage_training = 0.8
        #Simona: the partition the overall period on week basis 
        training_period = round((lastWeek - firstWeek) * percentage_training) * nObs
        
        #Partition of the Usage column of the dataframe      
        train, test = tsa.df.Usage[:training_period], tsa.df.Usage[training_period:]

        #Autocorrelation plot         
        tsa.plotACF(train)

        
        #(S)ARIMA(X) model from the time series      
        model = tsa.buildModel(train)
         
        #Model summary (important info): 
        #Information criterion: the lower is the better
        #p-value of the coeffs: good model if \forall coeff: p-value(coeff) << alpha (=0.05)
        print(model.summary())
        model.plot_diagnostics(figsize=(7,5))
        plt.show()
        
        #Prediction  
        pred, conf = tsa.predictConsumption(test)
        
        #Plotting the testing set, the predictions and conf.levels  
        tsa.plotPredictions(training_period,pred,conf,test)  
           

        #Number of false alarms
        n_false = tsa.computeOutliers(pred,conf,test)
        
        print("ACCURACY RESULTS -----------------------")
        print("n. false alarms: ", n_false)
        print("n. observations: ", pred.size)
        print("% of false alarms: ", n_false/pred.size)
