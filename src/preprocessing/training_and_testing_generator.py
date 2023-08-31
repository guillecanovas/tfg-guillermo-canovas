"""
@Author: Raul Javierre, Guillermo Cánovas
@Date: updated 28/08/2023
======================================================================================================================================================
@Code review: Simona Bernardi
Tested with Python 3.10.4 (pandas 1.4.2, numpy 1.22.4, scipy 1.8.1)
- replaced append (deprecated) with concat (also to solve performance issues)
- removed "SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame." (not all)
Try using .loc[row_indexer,col_indexer] = value instead
See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
---> all the places where a new column is added to an existing dataframe or when values are set to slices of an existing dataframe (performance issues)
=========================================================================================================================================================
The main program generates:
    1. Training csv files on SCRIPT_RESULTS<dataset>_training_data (GDrive)
        with syntax: meterID_0_60.csv | example: 2458_0_60.csv  (no attack)
        or syntax: meterID_kind_0_60.csv | example: 2458_Avg_0_60.csv   (attack = kind)
    2. Testing csv files on SCRIPT_RESULTS<dataset>_testing_data (GDrive)
        with syntax: meterID_kind_x_y.csv | example: 2458_Avg_x_y.csv   (attack = kind)
        or syntax: meterID_x_y.csv | example: 2458_x_y.csv  (no attack)

        (x and y are the first and last testing weeks, respectively)

for a list of N meterIDs previously selected from /src/__init__.py
"""

import pandas as pd
import numpy as np
import sys
from src import meterIDsGas, meterIDsElectricity, meterIDsSolar
from scipy.optimize import linprog
import random

#DATASET PATHS
ISSDA_PATH =  "./datasets/ISSDA-CER/" 
AUSGRID_PATH = "./datasets/Ausgrid/"
SCRIPT_RESULTS = "./script_results/"

#Constants of the specific datasets
NINE_AM = 19

FIRST_WEEK_TRAINING = 0 #used in "get_min_avg_and_max_usage_of_training_weeks"
FIRST_WEEK_TRAINING_ELECTRICITY = 0
FIRST_WEEK_TRAINING_GAS = 0
FIRST_WEEK_TRAINING_SOLAR = 0

LAST_WEEK_TRAINING = 60 #used in "get_min_avg_and_max_usage_of_training_weeks"
LAST_WEEK_TRAINING_ELECTRICITY = 60
LAST_WEEK_TRAINING_GAS = 60
LAST_WEEK_TRAINING_SOLAR = 50

FIRST_WEEK_TESTING_ELECTRICITY = LAST_WEEK_TRAINING_ELECTRICITY + 1
FIRST_WEEK_TESTING_GAS = LAST_WEEK_TRAINING_GAS + 1
FIRST_WEEK_TESTING_SOLAR = LAST_WEEK_TRAINING_SOLAR + 1

LAST_WEEK_TESTING_ELECTRICITY = 75
LAST_WEEK_TESTING_GAS = 77
LAST_WEEK_TESTING_SOLAR = 101

SEVEN_DAYS_PER_WEEK = 7
NOBS_PER_DAY = 48

THIRTY_PER_CENT = 0.3
TWENTY_PER_CENT = 0.2
TEN_PER_CENT = 0.1
FIVE_PER_CENT = 0.05

INFINITY = float('inf')
SEED = 19990722
dataset = None

RECORDS_BETWEEN_00_00_AND_09_00 = 18
RECORDS_BETWEEN_09_00_AND_00_00 = 30
PRICE_KWH_OFF_PEAK = 0.04
PRICE_KW_PEAK = 0.11

random.seed(SEED)
np.random.seed(SEED)


#SB: restructuración? 
# Hay funciones con diferentes própositos. 
# Pondría las que cargan los datasets training/testing como métodos de una clase Loader, por ejemplo.
# Las demás funciones son auxiliares a métodos para generar los datasets sínteticos y deberían ponerse en la clase
# AttackInjector

def get_the_eighteen_highest_usage_rows_of_peak_period(df):
    df_09_24 = df[df.DT % 100 > NINE_AM]   # takes the records between 09:00 and 24:00
    return df_09_24.nlargest(RECORDS_BETWEEN_00_00_AND_09_00, 'Usage')   # returns the 18 maximum values of Usage between 09:00 and 24:00


def get_the_off_peak_period(df):
    return df[df.DT % 100 < NINE_AM]   # returns the records between 00:00 and 09:00


def swap_usages(df):
    list_df = [df[i:i + NOBS_PER_DAY] for i in range(0, df.shape[0], NOBS_PER_DAY)]     # get a dataframe for each day

    for i in range(0, len(list_df)):
        eighteen_highest_peak_period = get_the_eighteen_highest_usage_rows_of_peak_period(list_df[i])
        off_peak_period = get_the_off_peak_period(list_df[i])

        dt_eighteen_highest_peak_period = eighteen_highest_peak_period['DT'].to_list()
        eighteen_highest_peak_period['DT'] = off_peak_period['DT'].to_list()
        off_peak_period['DT'] = dt_eighteen_highest_peak_period #SettingWithCopyWarning
        

        list_dts_modified = eighteen_highest_peak_period['DT'].to_list() + off_peak_period['DT'].to_list()
        list_df[i] = list_df[i].query('DT not in @list_dts_modified')
        list_df[i] = pd.concat([list_df[i],eighteen_highest_peak_period])
        list_df[i] = pd.concat([list_df[i],off_peak_period])
        list_df[i] = list_df[i].sort_values(by=['DT'])

    return pd.concat(list_df)['Usage'].to_list()


def get_min_avg_and_max_usage_of_training_weeks(caseID):
    min_avg = INFINITY

    all_data = pd.read_csv(SCRIPT_RESULTS + dataset.lower() + '_training_data/' + str(caseID) + '_' + str(FIRST_WEEK_TRAINING) + '_' + str(LAST_WEEK_TRAINING) + '.csv')
    max_usage = all_data['Usage'].max()
    data_weeks = np.array_split(all_data, len(all_data.index) / (NOBS_PER_DAY * SEVEN_DAYS_PER_WEEK))

    for week in data_weeks:
        avg = week['Usage'].mean()

        if avg < min_avg:
            min_avg = avg

    return min_avg, max_usage


def get_max_usage_per_week(data):
    data_weeks = np.array_split(data, len(data.index) / (NOBS_PER_DAY * SEVEN_DAYS_PER_WEEK))

    max_usage_per_week = []
    for week in data_weeks:
        max_usage_per_week.append(week['Usage'].max())

    return max_usage_per_week

def load_week_files_issda_cer(firstWeek, lastWeek):
    df = pd.DataFrame()
    for i in range(firstWeek, lastWeek + 1):
        filename = ISSDA_PATH + dataset + "/data/data_all_filtered/" + dataset + "DataWeek " + str(i)
        try:
            dset = pd.read_csv(filename)
            print(filename, "successfully obtained")
        except FileNotFoundError:  # Range is not complete or is out of range
            continue

        #df = df.append(dset)
        df = pd.concat([df, dset])

    return df

def load_ausgrid_data(type):
    df = pd.DataFrame()

    training_filename = AUSGRID_PATH + "data_all_filtered/" + type + "/2010-2011 Solar home electricity data.csv"
    testing_filename = AUSGRID_PATH + "data_all_filtered/" + type + "/2011-2012 Solar home electricity data v2.csv"
    
    training_df = pd.read_csv(training_filename)
    print(training_filename, "successfully obtained")

    # Remove the first four days to begin on Monday and the last four days to finish on Sunday: 51 weeks remaining
    training_df = training_df[(training_df.DT > 18600) & (training_df.DT < 54300)] 
    # Remove the first three days to begin on Monday and the last six days to end on Sunday: 51 weeks remaining
    testing_df = pd.read_csv(testing_filename)

    print(testing_filename, "successfully obtained")
    testing_df = testing_df[(testing_df.DT > 55000) & (testing_df.DT < 90700)] 

    return training_df, testing_df


class AttackInjector:
    """General attack injector"""

    def __init__(self, caseID=None):
        super().__init__()
        self.caseID = caseID

    def inject_attack(self, original_consume, a, b):
        noise = float(a) + np.random.rand(original_consume.size) * (float(b) - float(a))
        return original_consume * noise

    def attack_dataset(self, data, kind, a=None, b=None):
        consumed_faked = None

    #SB: este método necesita restructuración, por ejemplo usar "kind" (cadena), y posible parámetros, 
    #para hacer una llamada a un método "kind" que produzca el dataset sintético correspondiente

        if kind.startswith('RSA'):
            consumed_faked = self.inject_attack(data.Usage.to_numpy(), a, b)

        elif kind == 'Avg':
            mean_Kw = []
            # Split the dataframe into nWeeks dataframes
            data_weeks = np.array_split(data, len(data.index) / (NOBS_PER_DAY * SEVEN_DAYS_PER_WEEK))
            for data_week in data_weeks:
                # Create an array with len = NOBS_PER_DAY * 7 times
                mean_data_week = np.empty(len(data_week.index))
                # Fill the array with the mean of the week
                mean_data_week.fill(data_week['Usage'].mean())
                mean_Kw.append(mean_data_week)
            consumed_faked = self.inject_attack(np.array(mean_Kw).flatten(), a, b)

        elif kind == 'Min-Avg':
            min_avg, max_usage = get_min_avg_and_max_usage_of_training_weeks(self.caseID)
            max_usage_per_week = get_max_usage_per_week(data)
            prices_off_peak = np.full(RECORDS_BETWEEN_00_00_AND_09_00, PRICE_KWH_OFF_PEAK)
            prices_peak = np.full(RECORDS_BETWEEN_09_00_AND_00_00, PRICE_KW_PEAK)

            c = np.tile(np.append(prices_off_peak, prices_peak), SEVEN_DAYS_PER_WEEK) # c[i] is the value of the price of kwh for the moment i
            A_ub = np.full((1, len(c)), -1) # A_ub[0..335] = -1
            b_ub = -1 * len(c) * min_avg # B_ub = -336 * min(for all the weeks : avg(week_usage)) where weeks are retrieved from the meterID's training set

            consumed_faked = []
            for i in range(int(len(data.index) / (NOBS_PER_DAY * SEVEN_DAYS_PER_WEEK))):
                if max_usage_per_week[i] < max_usage:
                    random_max_usage = random.uniform(max_usage_per_week[i], max_usage)
                else:
                    random_max_usage = random.uniform(max_usage, max_usage_per_week[i])
                bounds = (0, random_max_usage)
                consumed_faked.append(linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds).get('x'))

            consumed_faked = np.array(consumed_faked).flatten()

        elif kind == 'Swap':
            consumed_faked = swap_usages(data)
        elif kind == 'FDI0':
            consumed_faked = np.zeros(len(data))
        elif kind == 'FDI5':
            #data['5%'] = data['Usage'] * FIVE_PER_CENT # SettingWithCopyWarning
            consumed_faked =  (data['Usage'] * FIVE_PER_CENT).tolist()
        elif kind == 'FDI10':
            #data['10%'] = data['Usage'] * TEN_PER_CENT # SettingWithCopyWarning
            consumed_faked = (data['Usage'] * TEN_PER_CENT).tolist()
        elif kind == 'FDI20':
            #data['20%'] = data['Usage'] * TWENTY_PER_CENT # SettingWithCopyWarning
            consumed_faked = (data['Usage'] * TWENTY_PER_CENT).tolist()
        elif kind == 'FDI30':
            #data['30%'] = data['Usage'] * THIRTY_PER_CENT # SettingWithCopyWarning
            consumed_faked = (data['Usage'] * THIRTY_PER_CENT).tolist()
        elif kind == "Rating":
            df = pd.read_csv('./datasets/Ausgrid/data/2010-2011 Solar home electricity data.csv', skiprows=1)
            postcode = df[df.Customer == self.caseID]['Postcode'].tolist()[0]
            capacity = df[df.Customer == self.caseID]['Generator Capacity'].tolist()[0]
            df = pd.read_csv('./script_results/solar_hours_by_postcode.csv')
            df = df[df.postcode == postcode]
            df = df[(df.DT >= data.DT.min()) & (df.DT <= data.DT.max())]
            is_sunny_list = df['is_sunny?'].tolist()
            consumed_faked = []
            for is_sunny in is_sunny_list:
                if is_sunny:
                    consumed_faked.append(capacity)
                else:
                    consumed_faked.append(0)

        elif kind == "Percentile":
            df = pd.read_csv('./datasets/Ausgrid/data/2010-2011 Solar home electricity data.csv', skiprows=1)
            postcode = df[df.Customer == self.caseID]['Postcode'].tolist()[0]
            df = pd.read_csv('./script_results/solar_hours_by_postcode.csv')
            df = df[df.postcode == postcode]
            df = df[(df.DT >= data.DT.min()) & (df.DT <= data.DT.max())]
            is_sunny_list = df['is_sunny?'].tolist()
            consumed_faked = []
            percentile_for_each_week = []
            data_weeks = np.array_split(data, len(data.index) / (NOBS_PER_DAY * SEVEN_DAYS_PER_WEEK))
            for data_week in data_weeks:
                data_week = data_week[data_week['Usage'] > 0]
                percentile = data_week['Usage'].quantile(0.9)
                percentile_for_each_week.append(percentile)
            print(percentile_for_each_week)
            week = 0    
            i = 0
            for is_sunny in is_sunny_list:
                if is_sunny:
                    consumed_faked.append(percentile_for_each_week[week])
                else:
                    consumed_faked.append(0)
                i += 1
                # 1 week = 48 * 7 = NOBS_PER_DAY * SEVEN_DAYS_PER_WEEK = 336 nObs
                week = int(i / (NOBS_PER_DAY * SEVEN_DAYS_PER_WEEK))
        else:
            print("Error: kind", kind, "not found")
            exit(1)

        return pd.DataFrame({'ID': self.caseID, 'DT': data.DT, 'Usage': consumed_faked})


def generate_attacked_file(attack_injector, meterID, dataset, d_type, first_week, last_week, dir, kind, a=None, b=None):
    if d_type == "solar_generation":
        print("Generating file:", SCRIPT_RESULTS + d_type.lower() + "_" + dir + "/" + str(meterID) + "_" + kind + "_" + str(first_week) + "_" + str(last_week) + ".csv")
        test_set = attack_injector.attack_dataset(data=dataset, kind=kind, a=a, b=b)
        test_set.to_csv(SCRIPT_RESULTS + d_type.lower() + "_" + dir + "/" + str(meterID) + "_" + kind + "_" + str(first_week) + "_" + str(last_week) + ".csv", index=False)  # GDrive
        
    else:
        print("Generating file:", SCRIPT_RESULTS + d_type.lower() + "_" + dir + "/" + str(meterID) + "_" + kind + "_" + str(first_week) + "_" + str(last_week) + ".csv")
        test_set = attack_injector.attack_dataset(data=dataset, kind=kind, a=a, b=b)
        test_set.to_csv(SCRIPT_RESULTS + d_type.lower() + "_" + dir + "/" + str(meterID) + "_" + kind + "_" + str(first_week) + "_" + str(last_week) + ".csv", index=False)  # GDrive


if __name__ == '__main__':
    '''
    args: 
    sys.argv[1]:dataset (Electricity/Gas/Solar_Consumption/Solar_Generation)
    '''

    if len(sys.argv) < 1 or sys.argv[1] != "Electricity" and sys.argv[1] != "Gas" and sys.argv[1] != "Solar_Consumption" and sys.argv[1] != "Solar_Generation":
        print("Usage: python3 training_and_testing_generator.py <Electricity/Gas/Solar_Consumption/Solar_Generation> ")
        exit(85)
   
    # Global variables
    dataset = sys.argv[1]

    if dataset == "Electricity":
        first_week_train = FIRST_WEEK_TRAINING_ELECTRICITY
        last_week_train = LAST_WEEK_TRAINING_ELECTRICITY
        first_week_test = FIRST_WEEK_TESTING_ELECTRICITY
        last_week_test = LAST_WEEK_TESTING_ELECTRICITY
        meterIDs = meterIDsElectricity
        dataset_training_all_meter_ids = load_week_files_issda_cer(firstWeek=first_week_train, lastWeek=last_week_train)
        dataset_testing_all_meter_ids = load_week_files_issda_cer(firstWeek=first_week_test, lastWeek=last_week_test)

    elif dataset == "Gas":
        first_week_train = FIRST_WEEK_TRAINING_GAS
        last_week_train = LAST_WEEK_TRAINING_GAS
        first_week_test = FIRST_WEEK_TESTING_GAS
        last_week_test = LAST_WEEK_TESTING_GAS
        meterIDs = meterIDsGas
        dataset_training_all_meter_ids = load_week_files_issda_cer(firstWeek=first_week_train, lastWeek=last_week_train)
        dataset_testing_all_meter_ids = load_week_files_issda_cer(firstWeek=first_week_test, lastWeek=last_week_test)

    elif dataset == "Solar":
        first_week_train = FIRST_WEEK_TRAINING_SOLAR
        last_week_train = LAST_WEEK_TRAINING_SOLAR
        first_week_test = FIRST_WEEK_TESTING_SOLAR
        last_week_test = LAST_WEEK_TESTING_SOLAR
        meterIDs = meterIDsSolar
        dataset_training_all_meter_ids, dataset_testing_all_meter_ids = load_ausgrid_data(type.lower())

    for meterID in meterIDs:
        attack_injector = AttackInjector(caseID=meterID)
        
        # Training weeks
        training_set = dataset_training_all_meter_ids.query('ID == @meterID')

        if dataset.lower() == "solar_generation":

            print("\nGenerating Training " + dataset.lower() + " file:", SCRIPT_RESULTS + dataset.lower() + "_training_data/" + str(meterID) + "_" + str(first_week_train) + "_" + str(last_week_train) + ".csv")
            training_set.to_csv(SCRIPT_RESULTS + dataset.lower() + '_' + type.lower() + "_training_data/" + str(meterID) + "_" + str(first_week_train) + "_" + str(last_week_train) + ".csv", index=False) 
            
            generate_attacked_file(attack_injector, meterID, training_set,  dataset.lower(), first_week_train, last_week_train, "training_data", "Rating")
            generate_attacked_file(attack_injector, meterID, training_set,  dataset.lower(), first_week_train, last_week_train, "training_data", "Percentile")
            generate_attacked_file(attack_injector, meterID, training_set,  dataset.lower(), first_week_train, last_week_train, "training_data", "RSA_0.5_3", 0.5, 3)
            

        else:
            print("\nGenerating Training " + dataset.lower() +  " file:", SCRIPT_RESULTS + dataset.lower() + "_training_data/" + str(meterID) + "_" + str(first_week_train) + "_" + str(last_week_train) + ".csv")
            training_set.to_csv(SCRIPT_RESULTS + dataset.lower() + "_training_data/" + str(meterID) + "_" + str(first_week_train) + "_" + str(last_week_train) + ".csv", index=False)   # GDrive
            
            generate_attacked_file(attack_injector, meterID, training_set, dataset.lower(), first_week_train, last_week_train, "training_data", "Swap")
            #generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "RSA_0.5_1.5", 0.5, 1.5)
            generate_attacked_file(attack_injector, meterID, training_set, dataset.lower(), first_week_train, last_week_train, "training_data", "Avg", 0.5, 1.5)
            #generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "Min-Avg")
            #generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "FDI0")
            #generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "FDI5")
            generate_attacked_file(attack_injector, meterID, training_set, dataset.lower(), first_week_train, last_week_train, "training_data", "FDI10")
            #generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "FDI20")
            generate_attacked_file(attack_injector, meterID, training_set, dataset.lower(), first_week_train, last_week_train, "training_data", "FDI30")
            generate_attacked_file(attack_injector, meterID, training_set, dataset.lower(), first_week_train, last_week_train, "training_data", "RSA_0.25_1.1", 0.25, 1.1)
            generate_attacked_file(attack_injector, meterID, training_set, dataset.lower(), first_week_train, last_week_train, "training_data", "RSA_0.5_3", 0.5, 3)
     

        # Testing weeks
        testing_set = dataset_testing_all_meter_ids.query('ID == @meterID')
        
        if dataset.lower() == "solar_generation":

            print("Generating Testing " + dataset.lower() +  "file:", SCRIPT_RESULTS + dataset.lower() + "_testing_data/" + str(meterID) + "_" + str(first_week_test) + "_" + str(last_week_test) + ".csv")
            testing_set.to_csv(SCRIPT_RESULTS + dataset.lower() + '_' + type.lower() + "_testing_data/" + str(meterID) + "_" + str(first_week_test) + "_" + str(last_week_test) + ".csv", index=False)   # GDrive

            generate_attacked_file(attack_injector, meterID, testing_set, dataset.lower(), first_week_test, last_week_test, "testing_data", "Rating")
            generate_attacked_file(attack_injector, meterID, testing_set, dataset.lower(), first_week_test, last_week_test, "testing_data", "Percentile")
            generate_attacked_file(attack_injector, meterID, testing_set, dataset.lower(), first_week_test, last_week_test, "testing_data", "RSA_0.5_3", 0.5, 3)
            
       
        else:
            print("Generating Testing " + dataset.lower() +  " file:", SCRIPT_RESULTS + dataset.lower() + "_testing_data/" + str(meterID) + "_" + str(first_week_test) + "_" + str(last_week_test) + ".csv")
            testing_set.to_csv(SCRIPT_RESULTS + dataset.lower() + "_testing_data/" + str(meterID) + "_" + str(first_week_test) + "_" + str(last_week_test) + ".csv", index=False)   # GDrive

            generate_attacked_file(attack_injector, meterID, testing_set, dataset.lower(), first_week_test, last_week_test, "testing_data", "Swap")
            #generate_attacked_file(attack_injector, meterID, testing_set, dataset, first_week_test, last_week_test,"testing_data", "RSA_0.5_1.5", 0.5, 1.5)
            generate_attacked_file(attack_injector, meterID, testing_set, dataset.lower(), first_week_test, last_week_test, "testing_data", "Avg", 0.5, 1.5)
            #generate_attacked_file(attack_injector, meterID, testing_set, dataset, first_week_test, last_week_test, "testing_data", "Min-Avg")
            #generate_attacked_file(attack_injector, meterID, testing_set, dataset, first_week_test, last_week_test, "testing_data", "FDI0")
            #generate_attacked_file(attack_injector, meterID, testing_set, dataset, first_week_test, last_week_test, "testing_data", "FDI5")
            generate_attacked_file(attack_injector, meterID, testing_set, dataset.lower(), first_week_test, last_week_test, "testing_data", "FDI10")
            #generate_attacked_file(attack_injector, meterID, testing_set, dataset, first_week_test, last_week_test, "testing_data", "FDI20")
            generate_attacked_file(attack_injector, meterID, testing_set, dataset.lower(), first_week_test, last_week_test, "testing_data", "FDI30")
            generate_attacked_file(attack_injector, meterID, testing_set, dataset.lower(), first_week_test, last_week_test, "testing_data", "RSA_0.25_1.1", 0.25, 1.1)
            generate_attacked_file(attack_injector, meterID, testing_set, dataset.lower(), first_week_test, last_week_test, "testing_data", "RSA_0.5_3", 0.5, 3)
    