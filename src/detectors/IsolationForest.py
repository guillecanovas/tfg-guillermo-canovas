"""
@Author: Raúl Javierre
@Date: updated 16/11/2020

@Review: Simona Bernardi - 20/03/2021

This module provides the functionality of a detector based on the Isolation Forest Algorithm
See https://blog.paperspace.com/anomaly-detection-isolation-forest/
"""

from sklearn.ensemble import IsolationForest
from src.detectors.Detector import Detector
from time import time
import numpy as np


class Isolation(Detector):

    def get_std_of_all_days(self, dataset):
        min_day = int(dataset.DT.min() / 100) * 100  # Getting the min_day DT
        max_day = int(dataset.DT.max() / 100) * 100  # Getting the max_day DT
        day = min_day
        day_std = []
        while day < max_day:
            std = dataset.query('DT >= @day & DT < (@day + 1*100)')['Usage'].std()  
            # Getting the avg of the usages of the week
            #Sb: standard deviation ?

            if std >= 0:
                day_std = day_std + [std]

            day += 1 * 100

        return day_std

    def build_model(self, training_dataset):
        t0 = time()
        #SB: What is the rationale of using 1013 estimators/trees?
        isolation_forest = IsolationForest(n_estimators=1013)
        day_std = self.get_std_of_all_days(training_dataset)
        isolation_forest.fit(np.array(day_std).reshape(-1, 1))

        return isolation_forest, time() - t0

    def predict(self, testing_dataset, model):
        t0 = time()
        day_std = self.get_std_of_all_days(testing_dataset)
        #SB: Returns evenly spaced numbers over [min_std,max_std] ?
        xx = np.linspace(np.array(day_std).min(), np.array(day_std).max(), len(day_std)).reshape(-1, 1)
        #SB: Computes the average anomaly score of xx ?
        anomaly_score = model.decision_function(xx)
        outlier = model.predict(xx)

        #SB: len(anomaly_score) = len(day_std) ?
        return np.count_nonzero(outlier == -1), len(anomaly_score), time() - t0