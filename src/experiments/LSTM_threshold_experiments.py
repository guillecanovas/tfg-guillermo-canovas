"""
@Author: Guillermo Cánovas
@Date: updated 30/07/2023
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

from keras_tuner import HyperModel
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

WINDOW_SIZE = 2     # Lenght of the "lookback window"
COLUMNS = 13

#THRESHOLD = 0.1
#METER_ID = 1014
#ATTACK_INJECTION = 'Avg'

batch_size = 32
num_epochs = 20


def get_training_testing(type):
  """
    Given a string with value 'training' or 'testing' returns the corresponding pandas DataFrame
    Inputs:
        type : string
    Outputs:
        electricity_data : pandas DataFrame
  """
  dir_path = './script_results/false_injection_training_testing_data/'

  if type == 'training':
    #file_list = ['1014_0_60.csv','1019_0_60.csv', '1021_0_60.csv', '1035_0_60.csv', '1047_0_60.csv']
    file_name = str(METER_ID) + '_0_60.csv'
    file_list = [file_name]

  elif type == 'testing':
    #file_list = ['1014_61_75.csv','1019_61_75.csv', '1021_61_75.csv', '1035_61_75.csv', '1047_61_75.csv']
    file_name = str(METER_ID) + '_Injection_' + str(ATTACK_INJECTION) + '_Test.csv'
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

def build_model(X_train, y_train):
        t0 = time()

        #Building the model
        model = Sequential()
        model.add(LSTM(256,return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        metrics=['mean_squared_error', 'mean_absolute_error']
        model.compile(loss='mse', optimizer='adam', metrics=metrics)
        model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', verbose = 0, patience = 7, restore_best_weights = True)

        # Fit the model
        history = model.fit(X_train,
                            y_train,
                            epochs=num_epochs,
                            batch_size=batch_size,
                            validation_split = 0.2,
                            shuffle=False,
                            verbose=1, # 0 == quiet
                            )

        return model, time() - t0

def calculate_mae(a,b):
  return np.mean(np.abs(a - b), axis=1)

def is_anomaly(original_value, predicted_value, threshold):
    """
    Check if the predicted value is an anomaly based on an original value and a threshold.
    Args:
        original_value (float): The original value.
        predicted_value (float): The predicted value.
        threshold (float, optional): The threshold as a percentage. Defaults to 0.1 (10%).
    Returns:
        bool: True if the predicted value is an anomaly, False otherwise.
    """
    max_difference = original_value * threshold  # Calculate the maximum allowed difference
    absolute_difference = abs(predicted_value - original_value)  # Calculate the absolute difference
    if absolute_difference <= max_difference:
        return False  # Not an anomaly
    else:
        return True  # Anomaly

def detect_anomalies(row):
    """
    Detect anomalies based on original and predicted values in a dataframe row.
    Args:
        row (pandas.Series): A row from a dataframe containing 'original_value' and 'predicted_value'.
    Returns:
        bool: True if an anomaly is detected, False otherwise.
    """
    original_value = row['Original']
    predicted_value = row['Prediction']
    anomaly = is_anomaly(original_value, predicted_value, THRESHOLD)
    return anomaly


def evaluate_model(model, X_test, y_test):
    y_predicted = model.predict(X_test,  batch_size = 32, verbose = 0, use_multiprocessing = True)

    prueba_train = df_train.iloc[:, 0].values
    prueba_test = df_test.iloc[:, 0].values

    # Normalizar
    sc = MinMaxScaler(feature_range=(0, 1))
    train2 = sc.fit_transform(prueba_train.reshape(-1,1))
    test2 = sc.transform(prueba_test.reshape(-1,1))
    print(train2.shape,test2.shape)

    y_predicted = model.predict(X_test,  batch_size = 32, verbose = 0, use_multiprocessing = True)

    y_predicted_des = sc.inverse_transform(y_predicted)
    y_test_des = sc.inverse_transform(y_test.reshape(-1,1))

    prediction_reshaped_arr = y_predicted_des.reshape((len(y_test),))
    test_reshaped_arr = y_test_des.reshape(len(y_test),)

    prediction_results = pd.DataFrame(data={'Prediction':prediction_reshaped_arr, 'Original':test_reshaped_arr})

    #Get the predictions and original values in arrays
    pred_res_array = prediction_results["Prediction"].values
    pred_res_array = pred_res_array.reshape(-1, 1)
    orig_res_array = prediction_results["Original"].values
    orig_res_array = orig_res_array.reshape(-1, 1)

    mean_absolute_error = calculate_mae(pred_res_array, orig_res_array)

    anomaly_results = prediction_results
    anomaly_results['MAE'] = mean_absolute_error
    anomaly_results['Threshold'] = THRESHOLD
    anomaly_results['Anomaly'] = anomaly_results.MAE > anomaly_results.Threshold
    anomaly_results['Should be'] = column_attack_test.values[48+WINDOW_SIZE:]
    anomaly_results

    rows_tp = anomaly_results[(anomaly_results['Anomaly'] == True) & (anomaly_results['Should be'] == 'Attack')]
    rows_fn = anomaly_results[(anomaly_results['Anomaly'] == False) & (anomaly_results['Should be'] == 'Attack')]
    rows_tn = anomaly_results[(anomaly_results['Anomaly'] == False) & (anomaly_results['Should be'] == 'False')]
    rows_fp = anomaly_results[(anomaly_results['Anomaly'] == True) & (anomaly_results['Should be'] == 'False')]
    
    num_tp = rows_tp.shape[0]
    num_fn = rows_fn.shape[0]
    num_tn = rows_tn.shape[0]
    num_fp = rows_fp.shape[0]

    accuracy = (num_tp + num_tn) / (num_tp + num_tn + num_fp + num_fn)

    print('Prueba con threshold de ' + str(THRESHOLD))
    print('TP: ' + str(num_tp))
    print('FN: ' + str(num_fn))
    #print("Accuracy: " + str(num_tp/(num_tp + num_fn)))
    print("Accuracy: " + str(accuracy))

    return num_tp, num_tn, num_fp, num_fn, accuracy

    
def metrics_to_csv(n_tp, n_tn, n_fp, n_fn, accuracy):
    
    resulting_csv_path = "./script_results/lstm_threshold/lstm_threshold_experiments.csv" 

    df = pd.DataFrame({ 'meterID': METER_ID,
                        'threshold': THRESHOLD,
                        'attack injection': ATTACK_INJECTION,
                        'n_tp': n_tp,
                        'n_tn': n_tn,
                        'n_fp': n_fp,
                        'n_fn': n_fn,
                        'accuracy': accuracy},
                        index=[0])

    df.to_csv(resulting_csv_path, mode='a', header=not os.path.exists(resulting_csv_path), index=False)
  

if __name__ == '__main__':

    """
    args:
    sys.argv[1]: meter_id
    sys.argv[2]: threshold value
    sys.argv[3]: type of attack that will be injected
    """

    if len(sys.argv) != 4:
        print("Usage: python3 LSTM_v1 <meter_id> <threshold value> <Avg/FDI10/Swap>")
        exit(85)

    METER_ID = sys.argv[1]
    THRESHOLD = float(sys.argv[2])
    ATTACK_INJECTION = sys.argv[3]

    #Get training and testing dataframe
    df_train = get_training_testing('training')
    df_test = get_training_testing('testing')
    df_all = pd.concat([df_train, df_test])

    df_train = data_preprocessing(df_train)
    #df_test = data_preprocessing(df_test)

    df_test['Datetime'] = df_test['Datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    df_test.set_index(['Datetime'], inplace=True)

    column_attack_test = df_test['Is_Attack']
    df_test = df_test.drop(['Is_Attack'], axis=1)

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

    model, model_time = build_model(X_train, y_train)
    n_tp, n_tn, n_fp, n_fn, accuracy = evaluate_model(model, X_test, y_test)

    metrics_to_csv(n_tp, n_tn, n_fp, n_fn, accuracy)

