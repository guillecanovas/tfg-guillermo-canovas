"""
@Author: Guillermo Cánovas
@Date: updated 20/08/2023
"""

import subprocess
import sys
from src import meterIDsGas, meterIDsElectricity, meterIDsSolar
from time import time

if __name__ == '__main__':

    """
    args:
    sys.argv[1]: type of dataset (electricity, gas, solar_consumption, solar_generation)
    """

    if len(sys.argv) != 2:
        print("Usage: python3 run_lstm_detector.py <electricity/gas/solar_consumption/solar_generation")
        exit(85)

    type_of_dataset = sys.argv[1]
    if type_of_dataset == "electricity":
        list_of_meterIDs = meterIDsElectricity
    elif type_of_dataset == "gas":
        list_of_meterIDs = meterIDsGas
    elif type_of_dataset == "solar_consumption" or type_of_dataset == "solar_generation":
        list_of_meterIDs = meterIDsSolar
    else:
        print("Usage: python3 run_lstm_detector.py <electricity/gas/solar_consumption/solar_generation")
        exit(85)

    tuple_of_attacks = (False, "Avg", "FDI10", "FDI30", "RSA_0.25_1.1", "RSA_0.5_3", "Swap")

    # Call the command with each combination as parameters
    for meterID in list_of_meterIDs:
        for attack in tuple_of_attacks:
            command = f"python3 src/detectors/LSTM_v1.py {type_of_dataset} {meterID} {attack}"
            subprocess.run(command, shell=True)


