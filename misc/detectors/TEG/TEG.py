"""
@Author: Simona Bernardi, Raúl Javierre
@Date: 04/03/2020
Time-Evolving-Graph detector Version 1.0  
New version of the TEG detector that is faster than Version 0 since the dataset is loaded once
and with a different approach with respect to V0.0.
- Builds the model based on TEGs: 
  -- the consumption graph of the whole training period: global_graph
  -- array of graphs distances: baseline[i] = distance(graph[i], global_graph), 
     where  "i" is a training week   
- Attack type: Min-Avg TO BE REVISED (NOW COMMENTED)
- Makes predictions: 
  -- array of graphs distances: graph_dist[i] = distance(graph[i], global_graph), 
     where  "i" is a testing week 
- Compute outliers: computes the number of weeks in the testing period whose distance from the global_graph
  is highest than the (100-alpha)-percentile of the baseline 

"""

import os
import random
import sys
from time import time

import numpy as np
import pandas as pd
from misc.detectors.TEG.graph_discovery import GraphGenerator,Graph
from misc.detectors.TEG.graph_comparison import GraphComparator,GraphHammingDissimilarity,GraphCosineDissimilarity,GraphKLDissimilarity


SEED = 19990722
HUNDRED = 100

class Loader:
    """Loader """
        
    def __init__(self,dpath, prefixFilename):
        # This part is very dependent on how the dataset is structured
        # Loads the list of filenames in myfiles array and then sort the files
        # according to the number of week
        dirFiles = os.listdir(dpath)
        self.myfiles = []
        for files in dirFiles:
            if files.startswith(prefixFilename):
                pathfile = (dpath + '/') + files
                self.myfiles.append(pathfile)
                self.myfiles.sort(key=lambda x: int(x.split()[-1]))

    def loadDatasetMeterID(self,meterID,firstWeek,nWeeks):

        # Loads the dataset of meterID, period [firstWeek,firstWeek+nWeeks-1]
        # into a dataframe: usages = [Idx, Usage]
        df = pd.read_csv(self.myfiles[firstWeek])
        usages = df[df['ID'] == int(meterID)].Usage
        for i in range(1, nWeeks):
            df = pd.read_csv(self.myfiles[firstWeek+i])
            usages = pd.concat([usages, df[df['ID'] == int(sys.argv[2])].Usage])

        return usages

    # TODO: remove method
    def load_same_training_dataset_last_experimentation(self, meterID):
        return pd.read_csv("./misc/script_results/gas_training_data/" + str(meterID) + "_0_60.csv")['Usage']

    # TODO: remove method
    def load_same_testing_dataset_last_experimentation(self, meterID, attack):
        if attack != "":
            return pd.read_csv("./misc/script_results/gas_testing_data/" + str(meterID) + "_" + attack + "_61_77.csv")['Usage']
        else:
            return pd.read_csv("./misc/script_results/gas_testing_data/" + str(meterID) + "_61_77.csv")['Usage']

class ConsumptionLevelExtractor:
    """Extractor of levels of consumptions from the original dataset"""

    def __init__(self, minValue, kw, n_bins):
        # Creates consumptionLevels [0,1,..,n_bins-1]
        self.ConsumptionLevel=np.arange(n_bins)
        self.kw = kw
        self.minValue = minValue

    def getLevel(self, gasConsumed):
        # Discretization of the "gasConsumed" according to the "self.ConsumptionLevels"
        # "gasConsumed" is a np.array (of floats)
        nObs = len(gasConsumed)    # number of observations
        level = -1 * np.ones(nObs) # array initialized with -1
        level = level.astype(int)  # level is a np.array of int
        
        #Case: "gasConsumed" (testing set) is lower than the minValue (of the training set)
        level = np.where( (gasConsumed < self.minValue + self.kw),
                         self.ConsumptionLevel[0],level)
        i = 1  # while iterator                         
        while i < len(self.ConsumptionLevel):
            lowerB = self.minValue + i * self.kw
            upperB = self.minValue + (i+1) * self.kw
            level = np.where((lowerB <= gasConsumed) & (gasConsumed < upperB),
                            self.ConsumptionLevel[i], level)
            i += 1
        
        #Case: "gasConsumed" (testing set) is greater than the maxValue (of the training set)
        n_bins = len(self.ConsumptionLevel)
        level = np.where( upperB <= gasConsumed, self.ConsumptionLevel[n_bins - 1], level) 
        
        return level

class AttackInjector:
    """General attack injector"""

    def __init__(self,kind,paramA,paramB):
        self.kind = kind
        self.paramA = float(paramA)
        self.paramB = float(paramB)

    def injectAttack(self, originalConsume, a, b):
        np.random.seed(SEED)
        noise = float(a) + np.random.rand(originalConsume.size) * (
                float(b) - float(a))
        consumedFaked = originalConsume * noise

        return consumedFaked
    
        
    def getSyntheticEL(self, data, nTestWeeks):
        nObs = 336  # number of observations per week

        if self.kind == 'RSA':
            consumedFaked = self.injectAttack(data,self.paramA,self.paramB)

        elif self.kind == 'Avg':
            nWeeks = int(data.size / nObs)
            mean_Kw = np.array([])
            for i in range(nWeeks):
                mean_Kw_week = np.ones(nObs) * np.mean(data[i*nObs:(i+1)*nObs])
                mean_Kw = np.concatenate((mean_Kw,mean_Kw_week), axis=None)
            consumedFaked = self.injectAttack(mean_Kw,self.paramA,self.paramB)

        elif self.kind == 'Min-Avg':
            sum_Kw = np.array([])
            for i in range(nTestWeeks):
                sum_Kw_week = np.sum(data[i * nObs:(i + 1) * nObs])
                sum_Kw = np.concatenate((sum_Kw, sum_Kw_week), axis=None)
            min_avg = sum_Kw.min() / nObs

            consumedFaked = []
            random.seed(SEED)
            for i in range(0, nObs * nTestWeeks):
                consumedFaked.append(random.uniform(min_avg, min_avg + 3))

        return consumedFaked


       
class TEGdetector:
    """Builds the model and makes predictions using TEG"""

    def __init__(self, minValue, kw, n_bins):
        # Creates an new consumption level extractor 
        self.el = ConsumptionLevelExtractor(minValue, kw, n_bins)
    
    #Pre: Graph "gr1" nodes set includes graph "gr2" nodes set
    def sumGraph(self,gr1,gr2): 
        
        for i in range(gr2.nodes.size):
            row = gr2.nodes[i]
            gr1.nodesFreq[row] += gr2.nodesFreq[i] 
            for j in range(gr2.nodes.size):
                   col = gr2.nodes[j]
                   gr1.matrix[row][col] += gr2.matrix[i][j]
    
    def getGlobalGraph(self,graphs):
        #Creates a global graph of max dimensions - initialized to zeros
        global_graph = Graph()
        n_bins = len(self.el.ConsumptionLevel)
        global_graph.nodes = np.arange(n_bins, dtype=int)
        global_graph.nodesFreq = np.zeros((n_bins,), dtype=int)
        global_graph.matrix = np.zeros((n_bins,n_bins), dtype=int)
        #print(global_graph.nodes)
        #print(global_graph.nodesFreq)
        #print(global_graph.matrix)
        for gr in graphs:
            self.sumGraph(global_graph, gr.graph)
            #print(global_graph.nodes)
            #print(global_graph.nodesFreq)
            #print(global_graph.matrix)
            #print(np.sum(global_graph.matrix))
        
        return global_graph
        
    def generateTEG(self,usagesClassified,nWeeks):
        #Creates the time evolving graphs series 
        nObs = int(len(usagesClassified)/nWeeks) # number of observations in a week
        graphs = []      
        for week in range(nWeeks):
            gr = GraphGenerator()
            eventlog = usagesClassified[week*nObs:(week+1)*nObs]
            #Transforms to a dataframe (needed to generate the graph)
            el = pd.DataFrame( {'Week': week*np.ones(nObs), 'Usage': eventlog})
            gr.generateGraph(el)
            graphs.append(gr)
 
        return graphs
    
    def computeGraphDist(self,gr1,gr2,metric):
        #Computes the distance between graphs "gr1" and "gr2" using the "metric"
        grcomp = GraphComparator(gr1,gr2)
        # Graph normalization
        grcomp.normalizeGraphs()
       
        # Computes the difference based on the metric                
        if metric == "Hamming":
            grcomp.__class__ = GraphHammingDissimilarity
            metricValue = grcomp.compareGraphs()
        elif metric == "Cosine":
            grcomp.__class__ = GraphCosineDissimilarity
            metricValue = grcomp.compareGraphs()
        elif metric == "KLD":
            grcomp.__class__ = GraphKLDissimilarity
            metricValue = grcomp.compareGraphs() 
            
        return metricValue
    
    def buildModel(self,metric,usages,nWeeks):
        #Builds: global_graph and
        #baseline[i] = distance(graph[i], global_graph), i in [0,nWeeks-1]
        
        #Gets consumption classification
        usagesClassified = self.el.getLevel(usages)
        
        #Generates the time-evolving graphs
        graphs = self.generateTEG(usagesClassified,nWeeks) 
        
        #Gets the consumption graph of the training period
        global_graph = self.getGlobalGraph(graphs)     
        
        #Computes the distance between each graph and the global graph
        graph_dist = np.empty(nWeeks)
        for week in range(nWeeks):
            graph_dist[week] = self.computeGraphDist(graphs[week].graph,global_graph,metric) 
        
        return graph_dist, global_graph
 
    def makePrediction(self,baseline,global_graph,metric,usages,nWeeks):
        #Builds: 
        #graph_dist[i] = distance(graph[i], global_graph), i in [0,nWeeks-1]

        #Gets consumption classification
        usagesClassified = self.el.getLevel(usages)

        #Generates the time-evolving graphs
        graphs = self.generateTEG(usagesClassified,nWeeks) 
        
        #Computes the distance between each graph and the global graph
        graph_dist = np.empty(nWeeks)
        for week in range(nWeeks):
            graph_dist[week] = self.computeGraphDist(graphs[week].graph,global_graph,metric) 
        
        return graph_dist
    
    
    def computeOutliers(self,model, test, sigLevel):
        # Computes the  percentile of the dissimilarity model 
        perc = np.percentile(model, sigLevel)
        #(sigLevel, "Percentile: ", perc)

        # Sets a counter vector to zero
        n_out = np.zeros(test.size)
        # Dissimilarity tests
        n_out = np.where(test > perc, n_out + 1, n_out)

        return np.sum(n_out)

    def print_metrics(self, meterID, n_bins, metric, alpha, attack, time_model_creation, time_model_prediction, n_tp, n_tn, n_fp, n_fn):
        print("\n\nMeterID:\t\t\t\t\t\t", meterID)
        print("Detector:\t\t\t\t\t\t", 'TEG_' + metric)
        print("Attack:\t\t\t\t\t\t\t", attack)
        print("Exec. time of model creation:\t", time_model_creation, "seconds")
        print("Exec. time of model prediction:\t", time_model_prediction, "seconds")
        print("Accuracy:\t\t\t\t\t\t", (n_tp + n_tn) / (n_tp + n_tn + n_fp + n_fn))
        print("Number of true positives:\t\t", n_tp)
        print("Number of false negatives:\t\t", n_fn)
        print("Number of true negatives:\t\t", n_tn)
        print("Number of false positives:\t\t", n_fp)
        print("[", n_tp, n_fp, "]")
        print("[", n_fn, n_tn, "]\n\n")

    def metrics_to_csv(self, meterID, n_bins, metric, alpha, attack, time_model_creation, time_model_prediction, n_tp, n_tn, n_fp, n_fn):
        resulting_csv_path = "./misc/script_results/gas_detector_comparer_results.csv"

        df = pd.DataFrame({'meterID': meterID,
                           'metric': 'TEG_' + metric,
                           'attack': attack,
                           'time_model_creation': time_model_creation,
                           'time_model_prediction': time_model_prediction,
                           'n_tp': n_tp,
                           'n_tn': n_tn,
                           'n_fp': n_fp,
                           'n_fn': n_fn,
                           'accuracy': (n_tp + n_tn) / (n_tp + n_tn + n_fp + n_fn)},
                          index=[0])

        df.to_csv(resulting_csv_path, mode='a', header=not os.path.exists(resulting_csv_path), index=False)


if __name__ == '__main__':
    '''
    args: 
    sys.argv[1]:dataset_directory (original data)
    sys.argv[2]:meterID 
    sys.argv[3]:n_bins (number of consumption levels)
    sys.argv[4]:metric_type(Hamming, Cosine, KLD) 
    sys.argv[5]:alpha (from this param. the significance level is set: 100 - alpha, alpha in [0,100])
    ---- OPTIONALS (attack scenario only):
    sys.argv[6]:attack_type (RSA, Avg, Min-Avg) 
    sys.argv[7]:paramA 
    sys.argv[8]:paramB
    '''
    if len(sys.argv) == 6 or len(sys.argv) == 9:

        # ------------- Loading of the dataset  --------------

        dpath=sys.argv[1]
        ld = Loader(dpath,'GasDataWeek')
        # Number of weeks of the training set
        nTrainWeeks = round(len(ld.myfiles) * 0.8)
        # Loads the dataset of the meterID of the firsts "nTrainWeeks"
        meterID = sys.argv[2]
        #usages = ld.loadDatasetMeterID(meterID,0,nTrainWeeks)                  #TODO: remove comment
        usages = ld.load_same_training_dataset_last_experimentation(meterID)    # TODO: remove line

        # ------------- Classification of the consumption --------------

        m = usages.min()           # min of the TRAINING dataset
        M = usages.max()           # max of the TRAINING dataset
        n_bins = int(sys.argv[3])  # Example: 7 -> [0, 1, 2, 3, 4, 5, 6]... 0: lowest consumption, 6: highest
        kw = (M-m) / n_bins        # equal distances of consumption between n_bins

        #print("Min usage:", m)
        #print("Max usage:", M)
        #print("Kw:", kw)
        #print("Number of bins:", n_bins)

        # ------------- Building the model -------------------------------------------

        # Creates a new detector
        teg = TEGdetector(m,kw,n_bins)

        metric=sys.argv[4]
        t0 = time()
        baseline, global_graph = teg.buildModel(metric,usages,nTrainWeeks)
        t1 = time()
        time_model_creation = t1 - t0

        #print("Metrics (training) - dimension ",len(baseline))
        #print(baseline)

        # ------------- Make predictions -------------------------------------------

        # Loads the dataset of the meterID of the weeks in [nTrainWeeks,nTrainWeeks+nTestWeeks-1]
        nTestWeeks = len(ld.myfiles) - nTrainWeeks
        #usagesTest = ld.loadDatasetMeterID(meterID,nTrainWeeks,nTestWeeks) #TODO: remove comment

        if len(sys.argv) == 6:
            # args: dataset_directory caseID n_bins metric_type alpha
            attack = False
            usagesTest = ld.load_same_testing_dataset_last_experimentation(meterID, attack="")  # TODO: remove line
        else:
            # args: dataset_directory caseID n_bins metric_type alpha attack_type paramA paramB
            attack = True
            ai = AttackInjector(sys.argv[6],sys.argv[7],sys.argv[8])
            usagesTest = ld.load_same_testing_dataset_last_experimentation(meterID, attack=sys.argv[6]) # TODO: remove line
            # Modifies "usagesTest" according to the attack type
            #if sys.argv[6] == "Min-Avg":                               #TODO: remove comment
            #    usagesTest = ai.getSyntheticEL(usages, nTestWeeks)     #TODO: remove comment
            #else:                                                      #TODO: remove comment
            #    usagesTest = ai.getSyntheticEL(usagesTest, nTestWeeks) #TODO: remove comment

        t0 = time()
        test = teg.makePrediction(baseline,global_graph,metric,usagesTest,nTestWeeks)
        t1 = time()
        time_model_prediction = t1 - t0

        #print("Metrics (testing) - dimension ", len(tests))
        #print(tests)

        # ------------- Compute outliers and print metrics ------------------------------
        alpha = int(sys.argv[5])
        sigLevel = HUNDRED - int(sys.argv[5])
        n_outliers = teg.computeOutliers(baseline, test, sigLevel)

        n_tp, n_tn, n_fp, n_fn = 0, 0, 0, 0
        if attack:  # if attacks were detected, they were true positives
            n_tp = n_outliers
            n_fn = test.size - n_outliers
        else:       # if attacks were detected, they were false positives
            n_fp = n_outliers
            n_tn = test.size - n_outliers

        attack = sys.argv[6] if attack else "False"

        teg.print_metrics(meterID, n_bins, metric, alpha, attack, time_model_creation, time_model_prediction, n_tp, n_tn, n_fp, n_fn)
        teg.metrics_to_csv(meterID, n_bins, metric, alpha, attack, time_model_creation, time_model_prediction, n_tp, n_tn, n_fp, n_fn)
