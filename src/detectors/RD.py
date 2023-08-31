"""
@Author: Raúl Javierre
@Date: updated 02/02/2021

(To be refined...) LEGACY?

This module implements a detector that tries to distinguish between RSA_0.5_1.5 and Normal behaviours.
"""

from time import time
from src.detectors.Detector import Detector

MEDITIONS_PER_DAY = 48


class RD(Detector):

    def build_model(self, training_dataset):
        t0 = time()

        model = {}
        for i in range(1, MEDITIONS_PER_DAY):
            reference = training_dataset[(training_dataset.DT % 100 == i)]['Usage'].std()
            model[str(i)] = [reference*0.8, reference*1.2]

        return model, time() - t0

    def predict(self, testing_dataset, model):
        """Try to detect if the measurements are between x0.8 x1.2 in the uniform distribution"""
        t0 = time()

        attacks = 0
        for i in range(1, MEDITIONS_PER_DAY):
            measurement = testing_dataset[(testing_dataset.DT % 100 == i)]['Usage'].std()

            if measurement < model[str(i)][0] or measurement > model[str(i)][1]:
                attacks += 1

        return attacks, MEDITIONS_PER_DAY, time() - t0
