"""
@Author: Simona Bernardi, Raúl Javierre
@Date: updated 23/10/2020

Discretization module: Classes that enable:
--- to load the raw dataset with gas consumption in Kw (.csv format) of a meter ID
--- to generate an event log from the dataset by discretizing the gas consumption
--- to save the event log (.csv format)
--- to generate a synthetic event log that model an integrity attack 
    (implemented types: RSA, Average, MinAverage)
"""

import os
import random
import pandas as pd
import numpy as np
from numpy import PINF #infinity symbolic constant

DEBUG = True

SEED = 19990722


class DataSetLoader:
    """Dataset loader in memory"""

    def __init__(self):
        self.dataset = pd.DataFrame()

    def load(self, filename, caseID):
        # load csv file as dataFrame
        # print ("Reading File: ", filename)
        dset = pd.read_csv(filename)  # load as a pd.dataFrame
        self.dataset = dset[dset.ID == int(caseID)]
        # print(self.dataset)

    def load_electricity_week_files(self, firstWeek, lastWeek, caseID=None):
        for i in range(firstWeek, lastWeek + 1):
            filename = os.path.join(
                os.path.dirname(__file__)) + "../ISSDA-CER/Electricity/data/data_all_filtered/ElectricityDataWeek " + str(i)
            try:
                dset = pd.read_csv(filename)
            except FileNotFoundError:  # Range is not complete or is out of range
                continue
            if caseID is not None:
                dset = dset[dset.ID == int(caseID)]

            self.dataset = self.dataset.append(dset)

        return self.dataset

    def get_min_avg_of_training_weeks(self, caseID):
        min_avg = PINF

        for i in range(0, 60 + 1):  # training weeks [week0, week60]
            filename = os.path.join(
                os.path.dirname(__file__)) + "../ISSDA-CER/Electricity/data/data_all_filtered/ElectricityDataWeek " + str(i)
            try:
                dset = pd.read_csv(filename)
            except FileNotFoundError:  # Range is not complete or is out of range
                continue
            dset = dset[dset.ID == int(caseID)]
            avg = dset['Usage'].mean()

            if avg < min_avg:
                min_avg = avg

        return min_avg


class EventLogExtractor:
    """Extractor of event logs attributes"""

    # __daysInMonths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # __minutesInHour = __secondsInMinute = 60
    # __minutesInDay = 1440
    # __second = 59

    def __init__(self, minValue, kw, n_bins):
        self.eventLog = pd.DataFrame()
        # ConsumptionLevels
        #self.ConsumptionLevel = [chr(x) for x in range(ord('A'), ord('A') + n_bins)]
        self.ConsumptionLevel=np.arange(n_bins)
        self.Kw = kw
        self.minValue = minValue
    '''
    def setConsumptionParams(self, classification, KwRange):
        self.ConsumptionLevel = classification
        self.Kw = KwRange
    '''
    def getEvent(self, gasConsumed):
        # gasConsumed is a np.array (of floats)
        nObs = gasConsumed.size  # number of observations
        clas = -1 * np.ones(nObs) #array initialized with -1
        clas = clas.astype(int) # class is a np.array (of int)
        #This is possible when gasConsumed is the testing set (lower than the min value)
        clas = np.where( gasConsumed < self.minValue + self.Kw,
                         self.ConsumptionLevel[0],clas)
        i = 1  # while iterator                         
        while i < len(self.ConsumptionLevel):
            # comparison and classification at array level
            lowerB = self.minValue + i * self.Kw
            upperB = self.minValue + (i+1) * self.Kw
            clas = np.where((lowerB <= gasConsumed) & (gasConsumed < upperB),
                            self.ConsumptionLevel[i], clas)
            i += 1
        # This is possible when gasConsumed is in the testing set (greater than the max value)
        n_bins = len(self.ConsumptionLevel)
        clas = np.where( upperB <= gasConsumed, self.ConsumptionLevel[n_bins - 1], clas) 
  
        return clas

  
    def extractEL(self, data):
        # Extract and transform the data for a specific smart meter
        self.eventLog = pd.DataFrame(
            {'DT': data.DT, 'Usage': self.getEvent(data.Usage)})


class AttackInjector(EventLogExtractor):
    """General attack injector"""

    def __init__(self, minValue, kw, n_bins, firstWeek=None, lastWeek=None, caseID=None):
        super().__init__(minValue, kw, n_bins)
        self.firstWeek = firstWeek
        self.lastWeek = lastWeek
        self.caseID = caseID

    def injectAttack(self, originalConsume, a, b):
        np.random.seed(SEED)
        noise = float(a) + np.random.rand(originalConsume.size) * (
                float(b) - float(a))
        consumedFaked = originalConsume * noise

        return consumedFaked

    def getSyntheticEL(self, data, kind, a=None, b=None):

        if self.firstWeek is not None and self.lastWeek is not None and self.caseID is not None:  # Selected weeks
            data_set_loader = DataSetLoader()
            data = data_set_loader.load_electricity_week_files(self.firstWeek, self.lastWeek, self.caseID)

        # Possible Strategy pattern for other types of attacks
        if kind == 'RSA':
            consumedFaked = self.injectAttack(data.Usage.to_numpy(), a, b)
        elif kind == 'Avg':
            mean_Kw = np.ones(data.count()['Usage']) * data.mean()['Usage']
            consumedFaked = self.injectAttack(mean_Kw, a, b)
        elif kind == 'Min-Avg':
            data_set_loader = DataSetLoader()
            min_avg = data_set_loader.get_min_avg_of_training_weeks(self.caseID)
            consumedFaked = []
            for i in range(0, 336 * (self.lastWeek - self.firstWeek + 1)):
                random.seed(SEED)
                consumedFaked.append(random.uniform(min_avg, min_avg + 3))

        # Extract attributes
        self.eventLog = pd.DataFrame({'DT': data.DT, 'Usage': self.getEvent(consumedFaked)}) # Categorical Usage: A, B, C, D, E...

        return self.eventLog

