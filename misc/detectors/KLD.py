'''
Created on Sept 29, 2020 

@author: Simona Bernardi
Analysis of the CER Electricity dataset according to the Badrinath thesis:
- Dataset: data_all_filtered
- The analysis is performed on a single meterID, in a period [initial,final] week
- Workflow of this script:
-- Loads the consumption data of a meterID for the whole period [firstWeek,lastWeek]
-- Separates the training set and the testing set
-- Builds the model based on the training set and the number of bins
-- Makes prediction based on the model, the significance level and the testing set
-- Computes the outliers (number of false alarms)

'''

import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import entropy #it is used to compute the KLD measure
import matplotlib.pyplot as plt

nObs = 336 #number of readings in a week

class loader:
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
            self.df = pd.concat([self.df,dset])
            i+=1
    
    
class KLDmodel:
    
    def __init__(self,min_v,max_v):
        self.m = min_v
        self.M = max_v
        #self.PX initialized in setXdist
        #self.K initialized in getKdist
        
    def setXdist(self,P_Xi):
        nWeeks = P_Xi.shape[0]
        nBins = P_Xi.shape[1]
        
        PT = np.transpose(P_Xi)
        #PX[j] = number of values of X that belong to bin_j
        self.PX = np.zeros(nBins)
        for j in range(nBins):
            self.PX[j] = np.sum(PT[j])
        
        self.PX = self.PX / nWeeks     
        print("X distribution:", self.PX)
     
    def setKdist(self,P_Xi):
        nWeeks = P_Xi.shape[0]
        self.K = np.zeros(nWeeks)
        
        for i in range(nWeeks):
            self.K[i] = entropy(P_Xi[i],self.PX,base=2)  
        
        print("KLD distribution:", self.K)
   
class KLDdetector:
    
    def __init__(self,bins,signLevel):
        self.nbins = bins
        self.signLevel = signLevel
        
    def getXiDist(self,ds,model):
        
        #Matrix X:  dimension [nWeeks, nObs]
        nWeeks = int(ds.shape[0] / nObs)
        print("N. of weeks: ", nWeeks)    
    
        #Week 0 for the first week 
        X = ds[0:nObs]
        #Rest of the weeks: one per row        
        for i in range(nWeeks-1):
            X = np.block([[ X ], [ds[nObs*(i+1):nObs*(i+2)] ]] )
        
        #P[i,j]= number of values of X[i] that belong to each bin j
        P = np.zeros([nWeeks,self.nbins])
        for i in range(nWeeks):
            P[i],b_edges = np.histogram(X[i], bins=self.nbins, range= (model.m,model.M))       
        
        #Normalization
        P = P/ nObs
        #print("X[i] distributions",P)
        
        #Bin edges
        #print("b_edges:",b_edges)
 
        return P, b_edges
    
    def buildModel(self,train):
        
        #Min, max values of the training set
        m = np.min(train)
        M = np.max(train)
        
        #Create model and set m,M
        model = KLDmodel(m,M);
        
        #Compute Xi distributions (nTrainWeeks,nbins) and bin edges
        P_Xi, b_edges = self.getXiDist(train,model)
        
        #Set X distribution (nTrainWeeks, nbins)
        model.setXdist(P_Xi)
        
        #Set KLD distribution (nTrainWeeks)
        model.setKdist(P_Xi)
        
        return model, b_edges
                
    def plotHistConsumption(self,PX,b_edges,tit):
        
        #Plot histogram
        plt.hist(b_edges[:-1],b_edges,weights=PX,histtype='bar')
        plt.gca().set(title=tit, xlabel='Consumption readings (kWh)', ylabel='Relative frequency')
        plt.show()
    
    def predictConsumption(self,test, model):
               
        #Compute Xi distributions (nTestWeeks,nbins) and bin edges
        P_Xa, b_edges = self.getXiDist(test,model)
        
        #print("Xi distributions (testing): ", P_Xa)
        
        nTestWeeks = P_Xa.shape[0]
        Ka = np.zeros(nTestWeeks)
        
        for i in range(nTestWeeks):
            Ka[i] = entropy(P_Xa[i],model.PX,base=2)  
        
        #print("Ka values (testing):", Ka)
        tit = "Non malicious distribution, week " + str(model.K.size+11) + "(K = " + str("{:.2f}".format(Ka[10]) + ")")
        self.plotHistConsumption(P_Xa[10], b_edges, tit)
        
        return Ka
   
    def computeOutliers(self,Ka,model):
        
        #Compute the (100-alpha) percentile of the K distribution
        perc = np.percentile(model.K, 100-self.signLevel)
        print(100-self.signLevel,"percentile: ",perc)
        
        #Setting a counter vector to zero
        n_out = np.zeros(Ka.size)
        #If Ka > percentile of K distribution => Week_a is anomalous
        n_out = np.where(Ka > perc,n_out+1,n_out)
        
        return np.sum(n_out)
         
if __name__ == '__main__':
    
    '''
    args: 
    sys.argv[1]: data_all_filtered folder
    sys.argv[2]: meterID
    sys.argv[3]: first week    
    sys.argv[4]: last week
    sys.argv[5]: number of bins
    sys.argv[6]: significance level alpha \in (0,100)
    '''
     
    if (len(sys.argv) == 7):

        # Loading dataset --------------------------------------------------------
        ld = loader()
        
        #Load the file names of the dataset files in files given the first 
        #param1 (directory): electricity/data_all_filtered/
        files = ld.getDataSetFiles(sys.argv[1],"ElectricityDataWeek")
        
        #param2: meterID 
        meterID = int(sys.argv[2])
        
        #param3 and param4: first and last weeks
        firstWeek = int(sys.argv[3])
        lastWeek = int(sys.argv[4])
        
        #Load time series of the meterID in memory
        #A new column is added besides DT,ID,Usage: the week number.
        ld.loadTimeSeries(files, firstWeek, lastWeek, meterID)
        
        #Training set / testing set
        percentage_training = 0.8
        #The partition of the overall period on week basis 
        training_period = round((lastWeek - firstWeek) * percentage_training) * nObs
        
        #Partition of the Usage column of the dataframe      
        train, test = ld.df.Usage[:training_period], ld.df.Usage[training_period:]
      
        # Build model based on the training set --------------------------------------

        #param5: number of bins
        kld = KLDdetector(int(sys.argv[5]),int(sys.argv[6]))
 
        #KLD model from the time-series      
        model,b_edges = kld.buildModel(train.to_numpy())
        
        #Plot baseline distribution X
        kld.plotHistConsumption(model.PX, b_edges, "Baseline distribution X: meterID " + str(meterID))
        
        #Prediction on the testing set ----------------------------------------------------
        Ka = kld.predictConsumption(test,model)
    
           
        #Compute false alarms
        n_false = kld.computeOutliers(Ka,model)
        
        print("ACCURACY RESULTS -----------------------")
        print("n. false alarms: ", n_false)
        print("n. observations: ", Ka.size)
        print("% of false alarms: ", n_false/Ka.size)
        