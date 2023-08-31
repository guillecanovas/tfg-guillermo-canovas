"""
@Author: Raúl Javierre
@Date: 20/04/2021

It tests the detection comparer script
"""

import unittest
from parameterized import parameterized
import pandas as pd
import os
from src.experiments import test_tuple_of_attacks, test_list_of_detectors


class TestDetectorComparer(unittest.TestCase):

    N_METER_IDS_TESTED = 1
    N_ATTACKS_TESTED = len(test_tuple_of_attacks)
    N_DETECTORS_TESTED = len(test_list_of_detectors)

    @parameterized.expand([
        ["electricity", "on"],
        ["gas", "on"]
    ])
    def testIfWeGetTheExpectedLinesInTheCSVFile(self, dataset, test_mode):
        os.system('python3 ./src/experiments/detector_comparer.py ' + dataset + ' ' + test_mode)
        df = pd.read_csv('./script_results/test_' + dataset + '_detector_comparer_results.csv')
        self.assertTrue(self.N_METER_IDS_TESTED * self.N_ATTACKS_TESTED * self.N_DETECTORS_TESTED, len(df.index))
        os.remove('./script_results/test_' + dataset + '_detector_comparer_results.csv')
        os.remove('./script_results/pca_dbscan_training.csv')


if __name__ == '__main__':
    unittest.main()
