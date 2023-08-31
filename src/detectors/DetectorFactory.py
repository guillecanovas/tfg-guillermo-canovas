"""
@Author: Raúl Javierre
@Date: updated 08/07/2021
@Review:
Simona Bernardi:  TEG detector updated with the tegdet implementation

This module provides a DetectorFactory class
"""

from .ARIMA import ARIMA
from .ARIMAX import ARIMAX
from .FisherJenks import FisherJenks
from .IsolationForest import Isolation
from .JSD import JSD
from .KLD import KLD
from .KMeans import K_Means
from .MinAverage import MinAverage
from .MiniBatchKMeans import MiniBatchK_Means
from .NN import NN
from .NN_v2 import NN_v2
from .PCADBSCAN import PCADBSCAN
from src.detectors.tegdet.teg import TEGDetector

class DetectorFactory:
    """
    It provides a static method to create concrete detectors
    """

    @staticmethod
    def create_detector(detector):
        if detector == "Min-Avg":
            return MinAverage()
        elif detector == "JSD":
            return JSD()
        elif detector == "ARIMA":
            return ARIMA()
        elif detector == "ARIMAX":
            return ARIMAX()
        elif detector == "FisherJenks":
            return FisherJenks()
        elif detector == "KLD":
            return KLD()
        elif detector == "K-Means":
            return K_Means()
        elif detector == "MiniBatchK-Means":
            return MiniBatchK_Means()
        elif detector == "PCA-DBSCAN":
            return PCADBSCAN()
        elif detector == "IsolationForest":
            return Isolation()
        elif detector == "NN":
            return NN()
        elif detector == "NN_v2":
            return NN_v2()
        elif detector.find("TEG") == 0:     # TEG_JSD_10, TEG_KLD_40, ... <-> TEG_<metric>_<n_bins>
            metric = detector.split("_")[1]
            n_bins = int(detector.split("_")[2])
            return TEGDetector(metric, n_bins)
        else:
            raise KeyError("Detector " + detector + " not found. You must add a conditional branch in /src/detectors/DetectorFactory.py")
