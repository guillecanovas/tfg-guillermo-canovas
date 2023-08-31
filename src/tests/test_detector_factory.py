"""
@Author: Raúl Javierre
@Date: 08/07/2022
@Revision: Simona B. - updated unit tests of TEG detector factory (TEG detectors)

It tests the src/detectors/DetectorFactory class
"""

import unittest

from src.detectors.DetectorFactory import DetectorFactory


class TestDetectorFactory(unittest.TestCase):

    def testIfCreateMinAvgDetectorWorks(self):
        detector = DetectorFactory.create_detector("Min-Avg")
        self.assertEqual('MinAverage', detector.__class__.__name__)

    def testIfCreateJSDDetectorWorks(self):
        detector = DetectorFactory.create_detector("JSD")
        self.assertEqual('JSD', detector.__class__.__name__)

    def testIfCreateARIMADetectorWorks(self):
        detector = DetectorFactory.create_detector("ARIMA")
        self.assertEqual('ARIMA', detector.__class__.__name__)

    def testIfCreateARIMAXDetectorWorks(self):
        detector = DetectorFactory.create_detector("ARIMAX")
        self.assertEqual('ARIMAX', detector.__class__.__name__)

    def testIfCreateFisherJenksDetectorWorks(self):
        detector = DetectorFactory.create_detector("FisherJenks")
        self.assertEqual('FisherJenks', detector.__class__.__name__)

    def testIfCreateKLDDetectorWorks(self):
        detector = DetectorFactory.create_detector("KLD")
        self.assertEqual('KLD', detector.__class__.__name__)

    def testIfCreateKMeansDetectorWorks(self):
        detector = DetectorFactory.create_detector("K-Means")
        self.assertEqual('K_Means', detector.__class__.__name__)

    def testIfCreateMiniBatchKMeansDetectorWorks(self):
        detector = DetectorFactory.create_detector("MiniBatchK-Means")
        self.assertEqual('MiniBatchK_Means', detector.__class__.__name__)

    def testIfCreateIsolationForestDetectorWorks(self):
        detector = DetectorFactory.create_detector("IsolationForest")
        self.assertEqual('Isolation', detector.__class__.__name__)

    def testIfCreateNNDetectorWorks(self):
        detector = DetectorFactory.create_detector("NN")
        self.assertEqual('NN', detector.__class__.__name__)

    def testIfCreateTEGCosineDetectorWorks(self):
        detector = DetectorFactory.create_detector("TEG_Cosine_35")
        self.assertEqual('TEGDetector', detector.__class__.__name__)
        metric = "_TEGDetector__metric"
        self.assertEqual('Cosine', detector.__dict__[metric])
        n_bins = "_TEGDetector__n_bins"
        self.assertEqual(35, detector.__dict__[n_bins])

    def testIfCreateTEGHammingDetectorWorks(self):
        detector = DetectorFactory.create_detector("TEG_Hamming_30")
        self.assertEqual('TEGDetector', detector.__class__.__name__)
        metric = "_TEGDetector__metric"
        self.assertEqual('Hamming', detector.__dict__[metric])
        n_bins = "_TEGDetector__n_bins"
        self.assertEqual(30, detector.__dict__[n_bins])

    def testIfCreateTEGKLDDetectorWorks(self):
        detector = DetectorFactory.create_detector("TEG_KL_10")
        self.assertEqual('TEGDetector', detector.__class__.__name__)
        metric = "_TEGDetector__metric"
        self.assertEqual('KL', detector.__dict__[metric])
        n_bins = "_TEGDetector__n_bins"
        self.assertEqual(10, detector.__dict__[n_bins])

    def testIfCreateNonExistentDetectorWorks(self):
        self.assertRaises(KeyError, DetectorFactory.create_detector, "Non Existent Detector")


if __name__ == '__main__':
    unittest.main()
