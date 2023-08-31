"""
@Author: Raúl Javierre
@Date: Updated on August 14, 2020

Two purposes:
1. Print the number of meterIDs considered in the filtered datasets.
2. Generates two files (meter_ids_electricity.txt/meter_ids_gas.txt) with that meterIDs in the script_results directory.
"""

from misc import get_meter_ids_and_files_electricity, get_meter_ids_and_files_gas

meter_ids_electricity = "./script_results/meter_ids_electricity.txt"
meter_ids_gas = "./script_results/meter_ids_gas.txt"

if __name__ == '__main__':
    # Electricity
    meterIDs, _ = get_meter_ids_and_files_electricity()    # files ignored here
    print("Number of meterIDs in ElectricityDataSet:", len(meterIDs))
    output = open(meter_ids_electricity, "w")
    for meterID in meterIDs:
        output.write(str(meterID) + "\n")
    output.close()

    # Gas
    meterIDs, _ = get_meter_ids_and_files_gas()  # files ignored here
    print("Number of meterIDs in GasDataSet:", len(meterIDs))
    output = open(meter_ids_gas, "w")
    for meterID in meterIDs:
        output.write(str(meterID) + "\n")
    output.close()
