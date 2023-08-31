"""
@Author: Guillermo Cánovas
@Date: updated 14/08/2023
"""

import subprocess
import sys
from src import meterIDsGas, meterIDsElectricity, meterIDsSolar
import random


if __name__ == '__main__':

    """
    args:
    sys.argv[1]: type of dataset (electricity, gas, solar_consumption, solar_generation)
    """

    if len(sys.argv) != 2:
        print("Usage: python3 threshold_experiments.py <electricity/gas/solar_consumption/solar_generation")
        exit(85)

    type_of_dataset = sys.argv[1]

    if type_of_dataset == "electricity":
        list_of_meterIDs = meterIDsElectricity
    elif type_of_dataset == "gas":
        list_of_meterIDs = meterIDsGas
    elif type_of_dataset == "solar_consumption" or type_of_dataset == "solar_generation":
        list_of_meterIDs = meterIDsSolar
    else:
        print("Usage: python3 run_hyperparameter_tuning.py <electricity/gas/solar_consumption/solar_generation")
        exit(85)

    thresholds = [0.0, 0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5]
    attack_injection = ['Avg', 'FDI10', 'Swap', 'None', 'RSA_0.25_1.1', 'RSA_0.5_3', 'FDI30']
    type_of_dataset = sys.argv[1]

    # Randomly select 100 meter IDs for this experiment
    list_of_meterIDs = random.sample(list_of_meterIDs, 20)

    # Call the command with each combination as parameters
    for arg1 in list_of_meterIDs:
        for arg2 in thresholds:
            for arg3 in attack_injection:
                command = f"python3 src/experiments/LSTM_detector_threshold.py {type_of_dataset} {arg1} {arg2} {arg3}"
                subprocess.run(command, shell=True)


