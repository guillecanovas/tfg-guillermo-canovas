import os
import re

folder_path_electricity_test= './script_results/electricity_testing_data/'
folder_path_electricity_train= './script_results/electricity_training_data/'

folder_path_gas_test = './script_results/gas_testing_data/'
folder_path_gas_train = './script_results/gas_training_data/'

file_list_electricity_test = os.listdir(folder_path_electricity_test)
file_list_electricity_train = os.listdir(folder_path_electricity_train)

file_list_gas_test = os.listdir(folder_path_gas_test)
file_list_gas_train = os.listdir(folder_path_gas_train)

prefix_set_electricity = set()
prefix_set_gas = set()

# A meter id should have:
    # 1. A file for each of this scenarios [Normal, FDI10, FDI30, RSA05_3, RSA025_11, Swap, Avg] in training and testing data

# Eliminate all the csv files that contains one of the following strings in their name [RSA_0.5_1.5,FDI0,FDI20,Min-Avg] 
for file_name_electricity in file_list_electricity_train:
    if re.search('.*(RSA_0.5_1.5|FDI5|FDI0|FDI20|Min-Avg).*', file_name_electricity) and file_name_electricity.endswith('.csv'):
        os.remove(os.path.join(folder_path_electricity_train, file_name_electricity))

for file_name_electricity in file_list_electricity_test:
    if re.search('.*(RSA_0.5_1.5|FDI5|FDI0|FDI20|Min-Avg).*', file_name_electricity) and file_name_electricity.endswith('.csv'):
        os.remove(os.path.join(folder_path_electricity_test, file_name_electricity))

for file_name_gas in file_list_gas_train:
    if re.search('.*(RSA_0.5_1.5|FDI5|FDI0|FDI20|Min-Avg).*', file_name_gas) and file_name_gas.endswith('.csv'):
        os.remove(os.path.join(folder_path_gas_train, file_name_gas))

for file_name_gas in file_list_gas_test:
    if re.search('.*(RSA_0.5_1.5|FDI5|FDI0|FDI20|Min-Avg).*', file_name_gas) and file_name_gas.endswith('.csv'):
        os.remove(os.path.join(folder_path_gas_test, file_name_gas))

# ELECTRICITY -------------------------------------------------------------------------------------------------------
# Loop through the files in the folder in order to get the IDs of the electricity meters we are training and testing

for file_name_electricity in file_list_electricity_test:
    if file_name_electricity.endswith('.csv'):
        prefix = file_name_electricity.split('_')[0]
        if prefix not in prefix_set_electricity: 
            prefix_set_electricity.add(prefix)

meterIDsElectricity = list(prefix_set_electricity)

# GAS ------------------------------------------------------------------------------------------------------
# Loop through the files in the folder in order to get the IDs of the gas meters we are training and testing

for file_name_gas in file_list_gas_test:
    if file_name_gas.endswith('.csv'):
        prefix = file_name_gas.split('_')[0]
        if prefix not in prefix_set_gas:
            prefix_set_gas.add(prefix)

meterIDsGas = list(prefix_set_gas)


# SOLAR ------------------------------------------------------------------------------------------------------
meterIDsSolar = range(1,300+1)
