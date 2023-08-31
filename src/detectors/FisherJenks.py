"""
@Author: Raúl Javierre
@Date: updated 16/11/2020

@Review: Simona Bernardi - 19/03/2021

This module provides the functionality of a detector based on Fisher-Jenks algorithm.
See https://pbpython.com/natural-breaks.html
"""

import pandas as pd
import jenkspy
from src.detectors.Detector import Detector
from time import time


class FisherJenks(Detector):

    def build_model(self, training_dataset):
        return 0, 0

    def predict(self, testing_dataset, model):
        t0 = time()
        try:
            #Labeling of consumption data
            testing_dataset['Fisher-Jenks'] = pd.cut(
                testing_dataset['Usage'],
                bins=jenkspy.jenks_breaks(testing_dataset['Usage'], nb_class=3),
                labels=['normal', 'suspicious', 'anomalous'],
                include_lowest=True,
                duplicates="drop")

            anomalous_dataframe = testing_dataset[testing_dataset['Fisher-Jenks'].values == "anomalous"]

            return len(anomalous_dataframe.index), len(testing_dataset), time() - t0

        except ValueError:  # All usages are equal: cannot split into bins
            return len(testing_dataset), len(testing_dataset), time() - t0
