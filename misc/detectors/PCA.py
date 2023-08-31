'''
Created on August 6, 2019
Update on August 20, 2020

@author: Simona Bernardi
Analysis of the CER Electricity dataset according to the Badrinath thesis:
- Filtering is needed since there are missing values:
    We consider only the meterID for which all the nObs=336 timestamps of each week are present.
    The week n. 36 has been removed from the dataset since for most of 
    the meterID the no. of observations are n=334 (two missing timestamps)
    There are some meterID for which there are more than 336 timestamps for some days
    (nonsense?): they are removed from the analysis.
- Plotting (bar plot) of the customer classification for the filtered meterID
- PCA analysis: matrices A and B are generated and analyzed with PCA
    Matrix A of dimension (nObs X nWeeks, nMeterID): the objective of PCA(ncomp=2) is 
    to see whether two points are sufficient to characterize each meterID with
    respect to the others.
    The matrix is reduced to a (nMeterID,2) matrix and a biplot is generated to show
    the similarity among customers (the customer type classification provided by ISSDA
    is used).
    Matrix B of dimension (nObs,nMeterID x nWeek): the objective of PCA(ncomp=2) is
    to see whether two points are sufficient to characterize the behavior of
    a meterID all along the nWeeks.
    The matrix is reduced to a (nMeterID x nWeek,2) matrix and a biplot is generated to show
    how similar are the consumption weeks for each customer with respect to the
    weeks of other customers (each customer is represented with a different color)
-   Save the results (Y_B, P_B and testSet) in different files for the DBSCAN step
'''


import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


nObs = 336 #number of readings

class dataAnalyzer:
    
    def __init__(self):
        #dataframe with all the readings: columns ['ID','DT','Usage']
        self.df = pd.DataFrame()
        # list of selected (complete) meterIDs together with their classification
        self.meterID = pd.DataFrame()
    
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
    
    ####################################################################
    
    def getCustomerClassification(self,file,missingID):
        #Classification according to the file "SME and Residential allocations" 
        #(CER Electricity Revised March 2012)
        dset=pd.read_csv(file,sep=';',index_col='ID')
        customers=dset.drop(missingID,axis=0)
        #customer is a dataFrame with index='ID' and column='Code'
        print(customers)
        return customers
    
    def updateMeterID(self,clfile):
        ID=self.df.groupby('ID').count().index.to_numpy()
        ID=pd.DataFrame(ID,columns=['ID'])
        clset=pd.read_csv(clfile,sep=';',index_col='ID')
        self.meterID=pd.merge(ID,clset,on='ID',how='inner')
        
        return 0
        
    def writeFilteredData(self,files,firstWeek,lastWeek):
        
        #Consider all the weeks between the first and the last one (included)
        ID=self.meterID.index.to_numpy()
        groupID=pd.DataFrame(ID,columns=['ID'])
        for file in files[firstWeek:lastWeek+1]:
            print ("Reading File: ", file)
            dset=pd.read_csv(file) #load as a pd.dataFrame
            #Select those observations where the timestamp timecode is between 0-48 hours
            dset=dset[(dset.DT % 100 >= 0) & (dset.DT % 100 <= 48)]
            #Inner join to select just the complete meterIDs
            dset=pd.merge(dset,groupID,on='ID',how='inner')
            dset.round(3) 
            #Same file name but in different directory
            filef= 'gas/data_all_filtered/'+file.split("/")[-1] 
            dset.to_csv(filef,index=False)
        
        return 0
    
    def filterDataSet(self, files, firstWeek, lastWeek, clfile):
             
        ID = np.int_([]) #np array initialization
        #Consider all the weeks between the first and the last one (included)
        for file in files[firstWeek:lastWeek+1]:
            print ("Reading File: ", file)
            dset=pd.read_csv(file) #load as a pd.dataFrame
            #Group by meterID and count the number of observations
            groupID=dset.groupby('ID').count()
            #Select those that have the number of observation less than nObs
            groupID=groupID[groupID.Usage < nObs]
            #print("Missing data for: ", groupID, "length: ", len(groupID))
            #Concatenate the missing IDs in the ID array
            ID = np.concatenate((ID,groupID.index.to_numpy()))
            #Remove the ID repetitions
            ID=np.unique(ID)
            print("Missing meter ID obs: ", ID, "Length: ",ID.size)
        
        #load the classification of the customers with complete observations
        self.meterID = self.getCustomerClassification(clfile,ID)
        
        #Generate filtered dataset in data_all_filtered: very heavy to be executed once
        self.writeFilteredData(files,firstWeek,lastWeek)
        
        return 0
 
    def loadFilteredData(self, files, firstWeek, lastWeek ,clfile):

        df=pd.DataFrame()
        i=1
        for file in files[firstWeek:lastWeek+1]:
            
            print ("Reading File: ", file)
            dset=pd.read_csv(file) #load as a pd.dataFrame
            #insert a new column for weeks i-1
            dset.insert(1,"Week", (i-1) * np.ones(dset.shape[0],dtype='int'),True)
            print("Statistics by meterID: ", dset.groupby('ID').count().describe())
            df = pd.concat([df,dset])
            if dset.shape != df.shape:
                print(dset.shape)
                groupID=df.groupby('ID').count()
                groupID=groupID[groupID.Usage == i * nObs]
                ID=groupID.index.to_numpy()
                groupID=pd.DataFrame(ID,columns=['ID'])
                #Inner join to select just the complete meterIDs
                df=pd.merge(df,groupID,on='ID',how='inner')
            i+=1
 
        return df
    
    ####################################################################
    # Methods for the PCA analysis
    ####################################################################    
    def generateHeatMap(self,mat):
        
        plt.figure(figsize=(20, 5))
        plt.imshow(mat, interpolation='none', cmap='viridis')
        plt.ylabel("Meter ID (customers)")
        plt.xlabel("Time stamps")
        plt.colorbar()           
        plt.show()
    
    def scaleMatrix(self,mat):
        #Scaling data: normal
        scaler = StandardScaler(with_mean=True,with_std=False)
        scaler.fit(mat.transpose())
        print("Number of means: ", len(scaler.mean_))
        X_mat_scaled = scaler.transform(mat.transpose())
        print("Number of items: ", len(X_mat_scaled))
        return np.divide(X_mat_scaled,len(X_mat_scaled))
        #return X_mat_scaled
    
    def getMatrixB(self,mat):
        #Matrix B: rearranged from df dimension B[nObs=336,nMeterID X nweeks]
        nWeeks = int(mat.shape[0] / nObs)
        print("N. of weeks: ", nWeeks)
        
        #Week 0 for the first week 
        B = mat[0:nObs,:]
        #for cust in range(mat.shape[1]):
            #if cust > 0:
            #    B = np.block([ [B], [mat[0:nObs,cust]] ] )
                
        for i in range(nWeeks-1):
            B = np.block([ B, mat[nObs*(i+1):nObs*(i+2),:] ] )
                
        return B
    
    def PCA(self,mat,comp,plotGraph):
        
        # keep the first "comp" principal components of the data
        pca = PCA(n_components=comp, svd_solver='full')
        # fit PCA model to the dataset
        
        pca.fit(mat)

        # transform data onto the first "comp" principal components
        mat_pca = pca.transform(mat)
        print("Amount of variance captured by each component: ", pca.explained_variance_ratio_) 
        #PCA amount of variance retained by each component 
        if plotGraph == True:
            y = 100*pca.explained_variance_ratio_
            x = np.arange(1,comp+1,1,int)
        
            fig, axes = plt.subplots()
            axes.plot(x,y,'o-')
            axes.set(xlabel='components', ylabel='variance retained %', title='PCA analysis')
            axes.grid()
            plt.show()
        
        return mat_pca, pca.components_
    
    def plotCustomerClassification(self):
        
        #Print the number of customer by type
        minCode= self.meterID['Code'].min()
        maxCode= self.meterID['Code'].max() 
        
        ind = np.arange(1)    # the x locations for the bars
        width = 0.35   # the width of the bars: can also be len(x) sequence

        ncust = []
        print("Number of meters by classification: ")
        for i in range(minCode,maxCode+1):
            ncust.append(len(mg.meterID[mg.meterID.Code == i]))
        print(ncust)
        
        #Bar plot
        bar0 = plt.bar(ind, ncust[0], width) #residential
        bar1 = plt.bar(ind, ncust[1], width,bottom=ncust[0]) #SME   
        bar2 = plt.bar(ind, ncust[2], width,bottom=ncust[0]+ncust[1]) #Unclassified
        plt.legend((bar0[0],bar1[0],bar2[0]), ('Residential', 'SME','Unclassified'))
        plt.ylabel('No.of customers')
        plt.title('Customer classification')
        plt.xticks(ind, (' '))
        plt.yticks(np.arange(0, ncust[0]+ncust[1]+ncust[2], 500))
        plt.show()

        return 0
    
    
    def showBiplot(self,X_pca,colormat,plTitle,legTitle):
        
            
        fig,axes = plt.subplots(1, 1, figsize=(6, 6))

        axes.set_title(plTitle)
        
        
        scatter=axes.scatter(X_pca[:, 0], X_pca[:, 1], c=colormat, alpha=0.7,linewidths=0, s=60, cmap='viridis')
        axes.set_xlabel("component 1")
        axes.set_ylabel("component 2")
        
        # produce a legend with the unique colors from the scatter
        legend = axes.legend(*scatter.legend_elements(), loc="upper left", title=legTitle)
        axes.add_artist(legend)
        
        plt.show()
        return
    
    def getColors(self, mat):
        
        color= 6000*np.ones(mat.shape[0])
        #MeterID analyzed by Badrinath
        selectedCust=np.array([1028,2613,4767,5000])
        ncust = self.meterID.shape[0]
        
        for i in range(selectedCust.size):
            idx=self.meterID[mg.meterID.ID == selectedCust[i]].index.to_numpy()
            #print("Customer idx: ",idx)
            for j in range(color.size):
                if j % ncust == idx:
                    color[j] = selectedCust[i]

        return color
    
    def transformToDataFrame(self,mat):  
        
        ncust = self.meterID.shape[0]
        rowid = np.array([]) #an array with the meterID indexes 0-335
        ID = self.meterID['ID'].to_numpy()
        nWeeks = int(mat.shape[0] / ncust)
        week = np.array([]) 
        for i in range(nWeeks):
            week = np.concatenate([week,i*np.ones(ncust)])
            rowid = np.concatenate([rowid, ID])
            
        week = np.transpose(np.array([week]))
        rowid = np.transpose(np.array([rowid]))
        
        mat = np.block([mat,week,rowid])
        
        #B_pca dataframe
        mat_pca = pd.DataFrame(mat,columns=['0','1','week','meterID'])
        
        return mat_pca
 
if __name__ == '__main__':
 
    '''
    args: 
    sys.argv[1]: data_all folder
    sys.argv[2]: customer classification file
    sys.argv[3]: data_all_filtered folder    
    '''
   
    
    if (len(sys.argv) == 4):
        
        #new dataAnalyzer object
        mg = dataAnalyzer()
        
        ##################################################################################
        # Load and filter the dataset - 1st round (all weeks) - "data_all" directory 
        ##################################################################################
        '''
        #Load the names of the dataset files: input param. "data_all" directory
        
        files = mg.getDataSetFiles(sys.argv[1],"ElectricityDataWeek")
        
        #Get the meterID with complete observations from "data_all/" files
        
        #Input param. "customerClassification.csv" file for the classification of meterID
        #Heavy: it create the filtered dataset in "data_all_filtered"
        
        #print(sys.argv[2])
        mg.filterDataSet(files,0,76,sys.argv[2])
        '''
        ######################################################################################
        # Load and filter the dataset - 2nd round (all weeks) - "data_all_filtered" directory 
        ######################################################################################
 
        files = mg.getDataSetFiles(sys.argv[3],"ElectricityDataWeek")
        
        #Input param. "customerClassification.csv" file for the classification of meterID
        mg.df = mg.loadFilteredData(files,0,76,sys.argv[2])
        
        #Udate the list of complete meterIDs together with their classification
        mg.updateMeterID(sys.argv[2])
        
        #PLots the customer by type
        mg.plotCustomerClassification()
        
        
        
        ########################################################################
        # PCA of matrix A 
        # Training set: the first 60 weeks from 0 to 60 (there is not weeks 36) 
        ########################################################################
        nTrainWeeks = 61
        nTotWeeks = mg.df.groupby('Week').count().index.to_numpy().size
        print("Number of weeks", nTotWeeks)
        
        
        #Select just the rows related to the training weeks
        A = mg.df[mg.df.Week < nTrainWeeks]
        #print(A.groupby('Week').count())
        
        
        #Getting the matrix A=[nObs X nWeeks, nMeterID]      
        A = A.pivot(index='DT',columns='ID',values='Usage').to_numpy()   
        print("A original shape: {}".format(str(A.shape)))
        
        #Scale the matrix A with respect to the meterIDs  (see Badrinath)    
        #A_scaled = mg.scaleMatrix(A)
        #print("A_scaled shape: {}".format(str(A_scaled.shape)))
        
        #Show the HeatMap for the scaled data
        #mg.generateHeatMap(A_scaled)
        
        #PCA analysis: it returns the reduced dataset and the transformer matrix 
        Y_A, P_A = mg.PCA(np.transpose(A),2,True)
        print("Y_A shape: ", Y_A.shape)
        print("P_A shape: ", P_A.shape)
              
        #Colors: convert the DataFrame into a np 2-dimensional array
        meters = mg.meterID.to_numpy()
        #Show biplot
        mg.showBiplot(Y_A,meters[:,1],"Clustering by customer type","Types")
        
        ########################################################################
        # PCAof matrix B 
        ########################################################################  
 
        B = mg.getMatrixB(A)
        print("B original shape: {}".format(str(B.shape)))

        #Scale the matrix B with respect to the nObs  (see Badrinath)    
        #B_scaled = mg.scaleMatrix(B)
        #print("B_scaled shape: {}".format(str(B_scaled.shape)))

        #PCA analysis
        Y_B, P_B = mg.PCA(np.transpose(B),2,True)
        print("Y_B shape: ", Y_B.shape)
        print("P_B shape: ", P_B.shape)
     
        #Biplots of B_pca
        #Colors: selecting a subset of meterID and the rest the same color
        
        color = mg.getColors(Y_B)
        mg.showBiplot(Y_B, color,"Clustering by consumption week","MeterID")
        
        #Transform the B_pca[nCust X nweek, 2] matrix into a dataframe with 
        #the week and meterIdx columns
        Y_B = mg.transformToDataFrame(Y_B)
        
        #Biplots of B_pca
        #Colors: selecting a subset of meterID 
        BBred_pca = Y_B[Y_B.meterID == 1028] 
        BBred_pca = pd.concat([BBred_pca,Y_B[Y_B.meterID == 2613]])  
        BBred_pca = pd.concat([BBred_pca,Y_B[Y_B.meterID == 4767]])
        BBred_pca = pd.concat([BBred_pca,Y_B[Y_B.meterID == 5000]])                  
        mat = BBred_pca[['0','1']].to_numpy()
        color = BBred_pca['meterID'].to_numpy()       
        mg.showBiplot(mat, color,"Clustering by consumption week","MeterID") 

        
        ########################################################################
        # Save the Y_B, P_B and tests sets for the next processing step (DB-SCAN)
        ########################################################################

        path = os.path.join(os.path.dirname(__file__))
        fileo = path + "/script_results/Y_B.csv"
        Y_B.to_csv(fileo,sep=";")
        fileo = path + "/script_results/P_B.csv"
        P_B = pd.DataFrame(P_B)
        P_B.to_csv(fileo,sep=";")
        fileo = path + "/script_results/testSet.csv"
        testSet= mg.df[mg.df.Week >= nTrainWeeks]
        testSet.to_csv(fileo,sep=";")
        
    
        
        
        