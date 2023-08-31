"""
@Author: Raúl Javierre, Guillermo Cánovas
@Date: updated 18/09/2023
@Review: Simona Bernardi

The main program generates some *basic metrics* to facilitate the comparison of the *detectors* for each *scenario*.
The results with all the metrics are generated in /script_results/<dataset>_detector_comparer_results.csv

**Metrics:**
- Execution time of model creation
- Execution time of model prediction
- Accuracy = (TP + TN) /(TP + TN + FP + FN)
- Number of TP (True Positives)
- Number of FP (False Positives)
- Number of TN (True Negatives)
- Number of FN (False Negatives)

**Detectors:**
- Min-Avg
- ARIMAX
- ARIMA
- PCA-DBSCAN
- K-Means
- MiniBatchK-Means
- FisherJenks
- NN
- KLD
- JSD
- IsolationForest
- TEG_Hamming_n_bins
- TEG_Cosine_n_bins
- TEG_Jaccard_n_bins
- TEG_Dice_n_bins
- TEG_KL_n_bins
- TEG_Jeffreys_n_bins
- TEG_JS_n_bins
- TEG_Euclidean_n_bins
- TEG_Cityblock_n_bins
- TEG_Chebyshev_n_bins
- TEG_Minkowski_n_bins
- TEG_Braycurtis_n_bins
- TEG_Gower_n_bins
- TEG_Soergel_n_bins
- TEG_Kulczynski_n_bins
- TEG_Canberra_n_bins
- TEG_Lorentzian_n_bins
- TEG_Bhattacharyya_n_bins
- TEG_Hellinger_n_bins
- TEG_Matusita_n_bins
- TEG_Squaredchord_n_bins
- TEG_Pearson_n_bins
- TEG_Neyman_n_bins
- TEG_Squared_n_bins
- TEG_Probsymmetric_n_bins
- TEG_Divergence_n_bins
- TEG_Clark_n_bins
- TEG_Additivesymmetric_n_bins


**Attacks:**
- False
- RSA_0.5_1.5
- RSA_0.25_1.1
- RSA_0.5_3
- Avg
- Min-Avg
- Swap
- FDI0
- FDI5
- FDI10
- FDI20
- FDI30
"""

import sys
import subprocess
from src.detectors.DetectorFactory import DetectorFactory
from src.experiments import test_tuple_of_attacks, test_list_of_detectors
from src import meterIDsGas, meterIDsElectricity, meterIDsSolar
from time import time

# The dataset is set in the first parameter
# The meterIDs used are specific for each dataset
# You can customize the attacks and the detectors you want to use here
#tuple_of_attacks = (False, "RSA_0.5_1.5", "RSA_0.25_1.1", "RSA_0.5_3", "Avg", "Min-Avg", "Swap", "FDI0", "FDI5", "FDI10", "FDI20", "FDI30")
#tuple_of_attacks = (False, "RSA_0.25_1.1", "RSA_0.5_3", "Avg", "Swap", "FDI10", "FDI30")
#list_of_detectors = [#"ARIMAX", "ARIMA", "NN", "Min-Avg", "JSD", "FisherJenks", "KLD", "K-Means", "MiniBatchK-Means", #"PCA-DBSCAN", "IsolationForest", 
#                   "TEG_Hamming_30" , "TEG_Cosine_30", "TEG_Jaccard_30", "TEG_Dice_30", "TEG_KL_30", "TEG_Jeffreys_30", "TEG_JS_30", 
#                    "TEG_Euclidean_30", "TEG_Cityblock_30", "TEG_Chebyshev_30", "TEG_Minkowski_30", "TEG_Braycurtis_30",
#                    "TEG_Gower_30", "TEG_Soergel_30", "TEG_Kulczynski_30", "TEG_Canberra_30", "TEG_Lorentzian_30",
#                    "TEG_Bhattacharyya_30", "TEG_Hellinger_30", "TEG_Matusita_30", "TEG_Squaredchord_30",
#                    "TEG_Pearson_30", "TEG_Neyman_30", "TEG_Squared_30", "TEG_Probsymmetric_30", "TEG_Divergence_30",
#                    "TEG_Clark_30", "TEG_Additivesymmetric_30" 
#                    ]

#list_of_detectors = ["PCA-DBSCAN"]
#list_of_detectors = ["TEG_Hamming_30"]
#list_of_detectors = ["TEG_Cosine_30", "TEG_Jaccard_30", "TEG_Dice_30", "TEG_KL_30", "TEG_Jeffreys_30", "TEG_JS_30"]
#list_of_detectors = ["TEG_Euclidean_30", "TEG_Cityblock_30", "TEG_Chebyshev_30", "TEG_Minkowski_30", "TEG_Braycurtis_30", "TEG_Gower_30", "TEG_Soergel_30", "TEG_Kulczynski_30", "TEG_Canberra_30", "TEG_Lorentzian_30"]
#list_of_detectors = ["TEG_Bhattacharyya_30", "TEG_Hellinger_30", "TEG_Matusita_30", "TEG_Squaredchord_30"]
#list_of_detectors = ["TEG_Pearson_30", "TEG_Neyman_30", "TEG_Squared_30", "TEG_Probsymmetric_30", "TEG_Divergence_30"]
#list_of_detectors = ["TEG_Clark_30", "TEG_Additivesymmetric_30"]
'''list_of_detectors = ["TEG_Hamming_30" , "TEG_Cosine_30", "TEG_Jaccard_30", "TEG_Dice_30", "TEG_KL_30", "TEG_Jeffreys_30", "TEG_JS_30", 
                    "TEG_Euclidean_30", "TEG_Cityblock_30", "TEG_Chebyshev_30", "TEG_Minkowski_30", "TEG_Braycurtis_30",
                    "TEG_Gower_30", "TEG_Soergel_30", "TEG_Kulczynski_30", "TEG_Canberra_30", "TEG_Lorentzian_30",
                    "TEG_Bhattacharyya_30", "TEG_Hellinger_30", "TEG_Matusita_30", "TEG_Squaredchord_30",
                    "TEG_Pearson_30", "TEG_Neyman_30", "TEG_Squared_30", "TEG_Probsymmetric_30", "TEG_Divergence_30",
                    "TEG_Clark_30", "TEG_Additivesymmetric_30" ]'''


list_of_detectors=["ARIMA", "ARIMAX"]


if __name__ == '__main__':
    """
    args:
    sys.argv[1]:dataset ("electricity" or "gas" or "solar_consumption" or "solar_generation")
    """
    if sys.argv[1] != "electricity" and sys.argv[1] != "gas" and sys.argv[1] != "solar_consumption" and sys.argv[1] != "solar_generation" :
        print("Usage: python3 detector_comparer.py <electricity/gas/solar_consumption/solar_generation>")
        exit(85)

    type_of_dataset = sys.argv[1]
    if type_of_dataset == "electricity":
        tuple_of_attacks = (False, "Avg", "FDI10", "FDI30", "RSA_0.25_1.1", "RSA_0.5_3", "Swap")
        list_of_meterIDs = meterIDsElectricity
    elif type_of_dataset == "gas":
        tuple_of_attacks = (False, "Avg", "FDI10", "FDI30", "RSA_0.25_1.1", "RSA_0.5_3", "Swap")
        list_of_meterIDs = meterIDsGas
    elif type_of_dataset == "solar_consumption":
        tuple_of_attacks = (False, "Avg", "FDI10", "FDI30", "RSA_0.25_1.1", "RSA_0.5_3", "Swap")
        list_of_meterIDs = meterIDsSolar
    else:
        tuple_of_attacks = (False, "Percentile", "RSA_0.5_3", "Rating")
        list_of_meterIDs = meterIDsSolar

    '''test_mode = len(sys.argv) == 3 and sys.argv[2] == "on"
    if test_mode:
        list_of_meterIDs = [list_of_meterIDs[0]]
        tuple_of_attacks = test_tuple_of_attacks
        list_of_detectors = test_list_of_detectors'''

    processed_meterIDs = 0
    t0 = time()
    

    for meterID in list_of_meterIDs:

        for name_of_detector in list_of_detectors:

            if name_of_detector != "LSTM":
                detector = DetectorFactory.create_detector(name_of_detector)
                training_dataset = detector.get_training_dataset(meterID, sys.argv[1])
                model, time_model_creation = detector.build_model(training_dataset)
        
            index = 0
            for attack in tuple_of_attacks:
                
                if name_of_detector != "LSTM":

                    testing_dataset = detector.get_testing_dataset(attack, meterID, sys.argv[1])
                    predictions, obs, time_model_prediction = detector.predict(testing_dataset, model, index)
                    n_tp, n_tn, n_fp, n_fn = detector.compute_outliers(obs, predictions, attack)

                    detector.print_metrics(meterID, name_of_detector, attack, time_model_creation, time_model_prediction, n_tp, n_tn, n_fp, n_fn)
                    detector.metrics_to_csv(meterID, name_of_detector, attack, time_model_creation, time_model_prediction, n_tp, n_tn, n_fp, n_fn, type_of_dataset)
                
                else:
                    command = f"python3 src/detectors/LSTM.py {type_of_dataset} {meterID} {attack}"
                    subprocess.run(command, shell=True)

                index += 1


        processed_meterIDs += 1

        remaining_meterIDs = len(list_of_meterIDs) - processed_meterIDs
        avg_time = (time() - t0) / processed_meterIDs

        print(str(remaining_meterIDs) + " meterIDs remaining. It will be completed in " + str(remaining_meterIDs * avg_time) + " seconds (aprox.)")
