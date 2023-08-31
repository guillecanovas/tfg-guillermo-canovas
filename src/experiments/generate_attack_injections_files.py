"""
@Author: Guillermo Cánovas
@Date: updated 15/07/2023
"""

import os
import datetime
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import subprocess
import shutil
from src import meterIDsGas, meterIDsElectricity, meterIDsSolar

FIRST_WEEK_TRAINING_ELECTRICITY = 0
LAST_WEEK_TRAINING_ELECTRICITY = 60
FIRST_WEEK_TESTING_ELECTRICITY = 61
LAST_WEEK_TESTING_ELECTRICITY = 75

FIRST_WEEK_TRAINING_GAS = 0
LAST_WEEK_TRAINING_GAS = 60
FIRST_WEEK_TESTING_GAS = 61
LAST_WEEK_TESTING_GAS = 77

FIRST_WEEK_TRAINING_SOLAR = 0
LAST_WEEK_TRAINING_SOLAR = 50
FIRST_WEEK_TESTING_SOLAR = 51
LAST_WEEK_TESTING_SOLAR = 101


def get_training_testing(type, meter_id, attack_injection, dataset_type):
  """
    Given a string with value 'training' or 'testing' returns the corresponding pandas DataFrame
    Inputs:
        type : string
    Outputs:
        electricity_data : pandas DataFrame
  """

  if dataset_type == 'electricity':
    dir_path = './script_results/electricity_testing_data/'

  elif dataset_type == 'gas':
    dir_path = './script_results/gas_testing_data/'

  elif dataset_type == 'solar_consumption':
    dir_path = './script_results/solar_consumption_testing_data/'

  elif dataset_type == 'solar_generation':
    dir_path = './script_results/solar_generation_testing_data/'

  if dataset_type == "electricity":
    FIRST_WEEK_TRAINING = FIRST_WEEK_TRAINING_ELECTRICITY
    LAST_WEEK_TRAINING = LAST_WEEK_TRAINING_ELECTRICITY
    FIRST_WEEK_TESTING = FIRST_WEEK_TESTING_ELECTRICITY
    LAST_WEEK_TESTING = LAST_WEEK_TESTING_ELECTRICITY
  elif dataset_type == "gas":
    FIRST_WEEK_TRAINING = FIRST_WEEK_TRAINING_GAS
    LAST_WEEK_TRAINING = LAST_WEEK_TRAINING_GAS
    FIRST_WEEK_TESTING = FIRST_WEEK_TESTING_GAS
    LAST_WEEK_TESTING = LAST_WEEK_TESTING_GAS
  else:
    FIRST_WEEK_TRAINING = FIRST_WEEK_TRAINING_SOLAR
    LAST_WEEK_TRAINING = LAST_WEEK_TRAINING_SOLAR  
    FIRST_WEEK_TESTING = FIRST_WEEK_TESTING_SOLAR
    LAST_WEEK_TESTING = LAST_WEEK_TESTING_SOLAR

  
  if type == 'testing attack':
    file_name = str(meter_id) + '_' + str(attack_injection) + '_' + str(FIRST_WEEK_TESTING) + '_' + str(LAST_WEEK_TESTING) + '.csv'
    file_list = [file_name]

  elif type == 'testing':
    file_name = str(meter_id) + '_' + str(FIRST_WEEK_TESTING) + '_' + str(LAST_WEEK_TESTING) + '.csv'
    file_list = [file_name]


  df_list = []
  for file in file_list:
      df = pd.read_csv(os.path.join(dir_path, file))
      df['filename'] = file
      df_list.append(df)

  electricity_data = pd.concat(df_list)
  electricity_data = electricity_data.drop(columns=['filename'])

  return electricity_data

def calculate_date(day):
  """
    Transforms a numeric value representing the day of the year into a value of type string with the format %Y-%m-%d
    Inputs:
        day : int
    Outputs:
        date_string : string
  """
  start_date = datetime(2008, 12, 31)
  future_date = start_date + timedelta(days = day)
  date_string = future_date.strftime('%Y-%m-%d')

  return date_string

def data_preprocessing(df):
  """
    Transforms a pandas DataFrame with columns 'ID', 'DT', 'Usage' into 
    DataFrame with columns 'ID', 'Usage', 'Dia', 'Media_hora' and 'Fecha' and finally into a DataFrame 
    with 'ID' and 'Datetime' as indexes and 'Usage' as the main column. 'Datetime' format its as follows: %Y-%m-%d %H:%M:%S
    Inputs:
        df : pandas DataFrame
    Outputs:x
        df : pandas DataFrame
  """

  df['Dia'] = df['DT'].astype(str).str[:3].astype(int) # Crear una nueva columna "dia" a partir de la columna "fecha_hora"
  df['Media_hora'] = df['DT'].astype(str).str[3:].astype(int) # Crear una nueva columna "hora" a partir de la columna "fecha_hora"
  df.drop('DT', axis=1, inplace=True) # Eliminar la columna "fecha_hora" original
  df['Fecha'] = df['Dia'].apply(calculate_date)

  df['Fecha'] = pd.to_datetime(df['Fecha']) # Convierte la columna de fecha a datetime
  df['Minutos'] = (df['Media_hora'] - 1) * 30 # Calcula la hora correspondiente en minutos
  df['Timedelta'] = pd.to_timedelta(df['Minutos'], unit='m') # Convierte los minutos a timedelta
  df['Datetime'] = df['Fecha'] + df['Timedelta'] # Suma la columna de fecha y la columna de timedelta
  df['Datetime'] = pd.to_datetime(df['Datetime'], format = '%Y-%m-%d %H:%M:%S')
  df['Is_Attack'] = 'False'

  # Option 1: con todos los IDs a la vez
  '''df = df.drop(columns=['Dia', 'Media_hora', 'Fecha', 'Timedelta', 'Minutos']) # Elimina las columnas innecesarias
  df.set_index(['Datetime','ID'], inplace=True)'''

  # Option 2: uno por uno cada ID
  df = df.drop(columns=['Dia', 'Media_hora', 'ID', 'Fecha', 'Timedelta', 'Minutos']) # Elimina las columnas innecesarias

  new_order = ['Datetime', 'Usage', 'Is_Attack']

  df = df[new_order]
  
  return df


def create_files(meter_id, attack_injection, dataset_type):

    #Get training and testing dataframe
    df_test_attack = get_training_testing('testing attack', meter_id, attack_injection, dataset_type)
    df_test = get_training_testing('testing', meter_id, attack_injection, dataset_type)

    df_test_attack = data_preprocessing(df_test_attack)
    df_test = data_preprocessing(df_test)

    rows, cols = df_test.shape
    num_modifications = len(df_test) * 0.10 #El 15% son valores modificados
    random_rows = np.random.randint(0, rows, int(num_modifications)) # Generar índices aleatorios para seleccionar las celdas a modificar
    random_rows[:5]

    # Modificar los valores en las celdas seleccionadas
    for row in random_rows:      
      df_test.iloc[row,1] = df_test_attack.iloc[row,1]
      df_test.iloc[row,2] = 'Attack'

    dir_path = './script_results/false_injection_training_testing_data/' + str(dataset_type) + "/"
    new_filename_test = str(meter_id) + "_Injection_" +  str(attack_injection) + "_Test.csv"
    df_test.to_csv(dir_path + new_filename_test, index=False)

    if dataset_type == "electricity":
      FIRST_WEEK_TRAINING = FIRST_WEEK_TRAINING_ELECTRICITY
      LAST_WEEK_TRAINING = LAST_WEEK_TRAINING_ELECTRICITY
      FIRST_WEEK_TESTING = FIRST_WEEK_TESTING_ELECTRICITY
      LAST_WEEK_TESTING = LAST_WEEK_TESTING_ELECTRICITY
    elif dataset_type == "gas":
      FIRST_WEEK_TRAINING = FIRST_WEEK_TRAINING_GAS
      LAST_WEEK_TRAINING = LAST_WEEK_TRAINING_GAS
      FIRST_WEEK_TESTING = FIRST_WEEK_TESTING_GAS
      LAST_WEEK_TESTING = LAST_WEEK_TESTING_GAS
    else:
      FIRST_WEEK_TRAINING = FIRST_WEEK_TRAINING_SOLAR
      LAST_WEEK_TRAINING = LAST_WEEK_TRAINING_SOLAR  
      FIRST_WEEK_TESTING = FIRST_WEEK_TESTING_SOLAR
      LAST_WEEK_TESTING = LAST_WEEK_TESTING_SOLAR


    #We add to the path the 'Normal scenario' training and testing data
    train_dir_path = './script_results/' + str(dataset_type) + '_training_data/' + str(meter_id) + '_' + str(FIRST_WEEK_TRAINING) + '_' + str(LAST_WEEK_TRAINING) + '.csv'
    test_dir_path =  './script_results/' + str(dataset_type) + '_testing_data/' + str(meter_id) + '_' + str(FIRST_WEEK_TESTING) + "_" + str(LAST_WEEK_TESTING) + '.csv'

    #We copy the files into our new directory
    shutil.copy2(train_dir_path, dir_path + str(meter_id) + '_' + str(FIRST_WEEK_TRAINING) + '_' + str(LAST_WEEK_TRAINING) + '.csv')
    shutil.copy2(test_dir_path, dir_path +  str(meter_id) + '_' + str(FIRST_WEEK_TESTING) + "_" + str(LAST_WEEK_TESTING) + '.csv')

    print(f"El fichero {new_filename_test} ha sido creado con éxito.")


if __name__ == '__main__':

    """
    args:
    sys.argv[1]: <electricity/gas/solar_consumption/solar_generation>
    """

    if len(sys.argv) != 2:
        print("Usage: python3 generate_attack_injections_files.py <electricity|gas|solar_generation|solar_consumption>")
        sys.exit(1)

    # Get the first argument which should be "electricity" or "gas"
    dataset_type = sys.argv[1]
    if dataset_type not in ["electricity", "gas", "solar_generation", "solar_consumption"]:
        print("The first argument should be 'electricity', 'gas', 'solar_generation' or 'solar_consumption'")
        sys.exit(1)

    if dataset_type == "electricity":
        list_of_meterIDs = meterIDsElectricity
    elif dataset_type == "gas":
        list_of_meterIDs = meterIDsGas
    elif dataset_type == "solar_consumption" or dataset_type == "solar_generation":
        list_of_meterIDs = meterIDsSolar

    attack_injection = ['Avg', 'FDI10', 'FDI30','Swap', 'RSA_0.25_1.1', 'RSA_0.5_3']

    for arg1 in list_of_meterIDs:
        for arg2 in attack_injection:
            create_files(int(arg1), arg2, dataset_type)