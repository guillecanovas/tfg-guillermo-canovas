"""
@Author: Guillermo Cánovas
@Date: updated 13/08/2023
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
        print("Usage: python3 run_hyperparameter_tuning_NN.py <electricity/gas/solar_consumption/solar_generation")
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


    # Randomly select 10 meter IDs
    list_of_meterIDs = random.sample(list_of_meterIDs, 10)
    

    # Call the command with each combination as parameters
    for meter_id in list_of_meterIDs:
        command = f"python3 src/experiments/hyperparameter_tuning_NN.py {type_of_dataset} {meter_id}"
        subprocess.run(command, shell=True)

