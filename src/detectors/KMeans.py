"""
@Author: Raúl Javierre
@Date: updated 13/11/2020

@Review: Simona Bernardi - 19/03/2021

This module provides the functionality of a detector based on KMeans clustering algorithm.
"""

from sklearn.cluster import KMeans
import pandas as pd
import math
from src.detectors.Detector import Detector
from time import time


class K_Means(Detector):

    def build_model(self, training_dataset):
        t0 = time()
        return build_k_means_model(training_dataset), time() - t0

    def predict(self, testing_dataset, model):
        t0 = time()
        n_false, obs = compute_outliers_k_means(testing_dataset, model)
        return n_false, obs, time() - t0


def transform_dataset(dframe):
    min_day = int(dframe.DT.min() / 100) * 100  # Getting the min_day DT (19500)
    max_day = int(dframe.DT.max() / 100) * 100  # Getting the max_day DT (62100)

    day = min_day
    list_of_avg = []

    #Computing the average of consumption per week
    while day < max_day:
        avg = dframe.query('DT >= @day & DT < (@day + 7*100)')['Usage'].mean()
        if not math.isnan(avg):
            list_of_avg = list_of_avg + [avg]

        day += 7 * 100

    return pd.DataFrame({'Usage': list_of_avg})


def build_k_means_model(training_dataset):
    training_dataset = transform_dataset(training_dataset)

    #Classification of normal/anomalous average consumption
    return KMeans(n_clusters=2).fit(training_dataset)


def compute_outliers_k_means(testing_dataset, kmeans):
    testing_dataset = transform_dataset(testing_dataset)

    #SB: Falta un comentario
    #    No entiendo el criterio de detección: ¿cómo se sabe que el primer grupo es un consumo normal y el segundo
    #    grupo es anomalo? 
    n_anomalous = sum(map(lambda x: x == 1, kmeans.predict(testing_dataset)))
    n_obs = len(testing_dataset)

    return n_anomalous, n_obs
