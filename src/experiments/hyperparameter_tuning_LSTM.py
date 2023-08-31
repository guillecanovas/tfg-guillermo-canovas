"""
@Author: Guillermo Cánovas
@Date: updated 15/07/2023
"""

import sys

from tensorflow import keras
from time import time
import pandas as pd
import numpy as np
import os
import scipy.stats as ss

from datetime import datetime, timedelta
from sklearn.preprocessing import *
from sklearn.metrics import accuracy_score

import tensorflow as tf

from tensorflow.keras.callbacks import *

from keras.models import Sequential
from keras.layers import *

from keras_tuner import HyperModel, Hyperband
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.tuners import *

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam



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

WINDOW_SIZE = 2     # Lenght of the "lookback window"
COLUMNS = 13


def get_training_testing(type):
  """
    Given a string with value 'training' or 'testing' returns the corresponding pandas DataFrame
    Inputs:
        type : string
    Outputs:
        electricity_data : pandas DataFrame
  """

  dir_path = './script_results/' + TYPE_OF_DATASET + "_" + type + '_data/'

  if TYPE_OF_DATASET == "electricity":
      FIRST_WEEK_TRAINING = FIRST_WEEK_TRAINING_ELECTRICITY
      LAST_WEEK_TRAINING = LAST_WEEK_TRAINING_ELECTRICITY
      FIRST_WEEK_TESTING = FIRST_WEEK_TESTING_ELECTRICITY
      LAST_WEEK_TESTING = LAST_WEEK_TESTING_ELECTRICITY
  elif TYPE_OF_DATASET == "gas":
      FIRST_WEEK_TRAINING = FIRST_WEEK_TRAINING_GAS
      LAST_WEEK_TRAINING = LAST_WEEK_TRAINING_GAS
      FIRST_WEEK_TESTING = FIRST_WEEK_TESTING_GAS
      LAST_WEEK_TESTING = LAST_WEEK_TESTING_GAS
  else:
      FIRST_WEEK_TRAINING = FIRST_WEEK_TRAINING_SOLAR
      LAST_WEEK_TRAINING = LAST_WEEK_TRAINING_SOLAR  
      FIRST_WEEK_TESTING = FIRST_WEEK_TESTING_SOLAR
      LAST_WEEK_TESTING = LAST_WEEK_TESTING_SOLAR


  if type == 'training':
    #file_list = ['1014_0_60.csv','1019_0_60.csv', '1021_0_60.csv', '1035_0_60.csv', '1047_0_60.csv']
    file_name = str(METER_ID) + '_' + str(FIRST_WEEK_TRAINING) + '_' + str(LAST_WEEK_TRAINING) + '.csv'
    file_list = [file_name]

  elif type == 'testing':
    #file_list = ['1014_61_75.csv','1019_61_75.csv', '1021_61_75.csv', '1035_61_75.csv', '1047_61_75.csv']
    file_name = str(METER_ID) + '_' + str(FIRST_WEEK_TESTING) + "_" + str(LAST_WEEK_TESTING) + '.csv'
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

  df = df.drop(columns=['Dia', 'Media_hora', 'ID', 'Fecha', 'Timedelta', 'Minutos']) # Elimina las columnas innecesarias
  df.set_index(['Datetime'], inplace=True)

  return df

def generate_input(df):
  """
  Generates new columns in a DataFrame based on different calculations of the "Usage" column.
  Input:
      df : DataFrame
          Input DataFrame that contains a "Usage" column with numerical data..
  """
  
  df['Mean Day'] = df.groupby(df.index.date)['Usage'].transform('mean')
  df['Mode Day'] = df.groupby(df.index.date)['Usage'].apply(lambda x: x.mode().iloc[0])
  df['Mode Day'] = df['Mode Day'].fillna(method='ffill')
  df['Max Day'] = df.groupby(df.index.date)['Usage'].transform('max')
  df['Min Day'] = df.groupby(df.index.date)['Usage'].transform('min')
  df['STD Day'] = df.groupby(df.index.date)['Usage'].transform('std')
  df['CV Day'] = df.groupby(df.index.date)['Usage'].transform('std') / df.groupby(df.index.date)['Usage'].transform('mean')

  hour_minute = df.index.to_series().dt.hour * 60 + df.index.to_series().dt.minute
  df['Half Hour'] = (hour_minute // 30) + 1

  df['Previous Day'] = df.groupby('Half Hour')['Usage'].shift()
  df['Previous Day After'] = df['Usage'].shift(47)
  df['Last'] = df['Usage'].shift(1)
  df['Actual - Last'] = df['Usage'] - df['Last']

  df['TS Mean'] = df.groupby('Half Hour')['Usage'].rolling(window=48, min_periods=1).mean().reset_index(0, drop=True)
  df['TS Max'] = df.groupby('Half Hour')['Usage'].expanding().max().reset_index(0, drop=True)
  df['TS Min'] = df.groupby('Half Hour')['Usage'].expanding().min().reset_index(0, drop=True)

  df = df[['Usage', 'Mean Day', 'Mode Day', 'Max Day', 'Min Day', 'STD Day', 'CV Day', 'Half Hour', 'Previous Day', 'Previous Day After', 'Last', 'Actual - Last', 'TS Mean', 'TS Max', 'TS Min']]


def create_sequence(train, test, time_steps=WINDOW_SIZE):
  X_train, y_train, X_test, y_test = [], [], [], []

  for i in range(time_steps, train.shape[0]):
        X_train.append(train[i-time_steps:i])
        y_train.append(train[i][0])

  for i in range(time_steps, test.shape[0]):
        X_test.append(test[i-time_steps:i])
        y_test.append(test[i][0])

  return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


class MyHyperModel(HyperModel):
  
  def build(self, hp):

    model = Sequential()

    model.add(LSTM(
          units = hp.get('LSTM_units'),
          input_shape = INPUT_SHAPE,
          return_sequences = True,
          dropout = 0.2
    ))

    #Hidden layers 
    for i in np.arange(0, hp.get('n_layers')): 
      model.add(LSTM(
          return_sequences = True,
          units = hp.get('LSTM_units'),
          dropout = 0.2
      ))

    model.add(LSTM(
          units = hp.get('LSTM_units'),
          return_sequences = False,
    ))

    #Output layer  
    model.add(Dense(units = 1, activation = 'linear'))

    # Compiling the model
    metrics=['mean_squared_error', 'mean_absolute_error']
    model.compile(loss='mse', optimizer='adam', metrics=metrics)

    return model

  def fit(self, hp, model, *args, **kwargs):
      return model.fit(*args, batch_size = hp.get('batch_size'), **kwargs)
  


if __name__ == '__main__':

    """
    args:
    sys.argv[1]: meter_id
    sys.argv[2]: type of dataset
    """

    if len(sys.argv) != 3:
        print("Usage: python3 hyperparameter_tuning_LSTM.py <meter_id> <electricity/gas/solar_consumption/solar_generation>")
        exit(85)

    METER_ID = sys.argv[1]
    TYPE_OF_DATASET = sys.argv[2]

    #Get training and testing dataframe
    df_train = get_training_testing('training')
    df_test = get_training_testing('testing')

    df_train = data_preprocessing(df_train)
    df_test = data_preprocessing(df_test)

    #Generate the new columns
    generate_input(df_train)
    generate_input(df_test)

    df_train = df_train.drop(columns=['Last', 'Half Hour']) 
    df_test = df_test.drop(columns=['Last', 'Half Hour'])

    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # Normalizar
    sc = MinMaxScaler(feature_range=(0, 1))
    train = sc.fit_transform(df_train)
    test = sc.transform(df_test)
    print(train.shape,test.shape)

    X_train, y_train, X_test, y_test = create_sequence(train, test, WINDOW_SIZE)

    INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])


    #Hyperparameter tuning
    hp = HyperParameters()
    hp.Int("n_layers", min_value = 0 , max_value = 3) 
    hp.Int("batch_size", min_value = 8, max_value = 256, step = 16)
    hp.Int("LSTM_units", min_value = 32, max_value = 512, step = 32)

    hyperband_tuner = Hyperband(
                        MyHyperModel(),
                        hyperparameters = hp,
                        objective = "val_loss", 
                        project_name ="hyperband_LSTM_tuning_" + str(WINDOW_SIZE) + '_' + str(METER_ID),
                        max_epochs = 15,
                        hyperband_iterations = 1, 
                        directory = "main_dir",
                        overwrite = True, 
                    )
  
    #early_stopping = EarlyStopping(monitor='val_loss', verbose = 0, patience = 10, min_delta = 1e-3, restore_best_weights = True)

    hyperband_tuner.search( X_train, 
                            y_train, 
                            steps_per_epoch = None, 
                            shuffle = False, 
                            validation_split = 0.2, 
                            verbose = 1, 
                            #callbacks = [early_stopping], 
                            use_multiprocessing = True, 
                        )

    hyperband_best_hps = hyperband_tuner.get_best_hyperparameters()[0]
    hyperband_best_model = hyperband_tuner.hypermodel.build(hyperband_best_hps)

    with open('./script_results/tuner_results/LSTM/tuner_lstm_results' + '.txt', 'a') as f:
      print("LSTM (meter_id = " + str(METER_ID) + ", window = " + str(WINDOW_SIZE) + ", dataset = " + str(TYPE_OF_DATASET) + ")", file=f)
      print(hyperband_best_hps.values, file=f)
      print("================================================", file=f)

    hyperband_best_hps.values


