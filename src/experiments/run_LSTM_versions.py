"""
@Author: Guillermo Cánovas
@Date: updated 20/08/2023
"""

import subprocess
import sys
from src import meterIDsGas, meterIDsElectricity, meterIDsSolar
import random


if __name__ == '__main__':

    """
    args:
    sys.argv[1]: type of dataset (electricity, gas, solar_consumption, solar_generation>)
    """

    if len(sys.argv) != 2:
        print("Usage: python3 run_LSTM_versions.py <electricity/gas/solar_consumption/solar_generation>")
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
    elif type_of_dataset == "solar_generation":
        tuple_of_attacks = (False, "Percentile", "Rating", "RSA_0.5_3")
        list_of_meterIDs = meterIDsSolar
    else:
        print("Usage: python3 run_LSTM_versions.py <electricity/gas/solar_consumption/solar_generation>")
        exit(85)


    # Randomly select 20 meter IDs
    list_of_meterIDs = random.sample(list_of_meterIDs, 20)
    

    # Call the command with each combination as parameters
    for meter_id in list_of_meterIDs:
        for attack in tuple_of_attacks:
            for version in range(1, 4):
                command = f"python3 src/experiments/LSTM_detector_versions.py {version} {type_of_dataset} {meter_id} {attack}"
                subprocess.run(command, shell=True)

