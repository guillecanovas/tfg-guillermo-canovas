"""
@Author: Guillermo Cánovas
@Date: updated 15/08/2023
"""

from tensorflow import keras
from time import time
from src.detectors.Detector import Detector
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import scipy.stats as ss
import sys

from tensorflow.keras.callbacks import *

from keras.models import Sequential
from keras.layers import *

from keras_tuner import HyperModel, Hyperband
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.tuners import *

from tensorflow.keras.callbacks import ModelCheckpoint


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

WINDOW_LEN = 48     # 48 = one day window
WINDOW_SIZE = 30     # Lenght of the "lookback window"


def get_training_dataset(meterID, type_of_dataset, type_of_attack=None):
    list_of_dataframes = list()

    d_path = "./script_results/" + type_of_dataset + "_training_data/"

    dir_files = os.listdir(d_path)
    dir_files.sort()
    for file in dir_files:
        if file.startswith(str(meterID)+"_"):
            training_scenario = pd.read_csv(d_path + file)
            list_of_dataframes.append(training_scenario)

    # list_of_dataframes[0] -> normal
    return list_of_dataframes

def get_testing_dataset(attack, meterID, type_of_dataset, type_of_attack=None):
    """
    Returns the testing dataset for the meterID passed
    """
    if type_of_dataset == "electricity":
        FIRST_WEEK_TESTING = FIRST_WEEK_TESTING_ELECTRICITY
        LAST_WEEK_TESTING = LAST_WEEK_TESTING_ELECTRICITY
    elif type_of_dataset == "gas":
        FIRST_WEEK_TESTING = FIRST_WEEK_TESTING_GAS
        LAST_WEEK_TESTING = LAST_WEEK_TESTING_GAS
    else:
        FIRST_WEEK_TESTING = FIRST_WEEK_TESTING_SOLAR
        LAST_WEEK_TESTING = LAST_WEEK_TESTING_SOLAR
        if attack:
            return pd.read_csv("./script_results/" + type_of_dataset + "_testing_data/" + str(meterID) + "_" + attack + "_" + str(FIRST_WEEK_TESTING) + "_" + str(LAST_WEEK_TESTING) + ".csv")
        else:
            return pd.read_csv("./script_results/" + type_of_dataset + "_testing_data/" + str(meterID) + "_" + str(FIRST_WEEK_TESTING) + "_" + str(LAST_WEEK_TESTING) + ".csv")

    if attack:
        testing_dataset = pd.read_csv("./script_results/" + type_of_dataset + "_testing_data/" + str(meterID) + "_" + attack + "_" + str(FIRST_WEEK_TESTING) + "_" + str(LAST_WEEK_TESTING) + ".csv")
    else:
        testing_dataset = pd.read_csv("./script_results/" + type_of_dataset + "_testing_data/" + str(meterID) + "_" + str(FIRST_WEEK_TESTING) + "_" + str(LAST_WEEK_TESTING) + ".csv")

    return testing_dataset


'''
def build_model(training_dataset):
    t0 = time()

    window_len = WINDOW_LEN

    model = keras.Sequential()
    model.add(keras.layers.Dense(units=50, activation='relu'))
    model.add(keras.layers.Dense(units=len(training_dataset), activation='softmax'))  # one neuron for each class
    model.compile(loss='binary_crossentropy', optimizer='adam')

    x = []
    y = []

    # For each kind of behaviour
    for i in range(0, len(training_dataset)):   # i:0 -> normal

        # split into windows of two days
        list_df = [training_dataset[i][w:w + window_len] for w in range(0, training_dataset[i].shape[0], window_len)]

        # Generate training set for each window of two days
        for j in range(WINDOW_SIZE-1, len(list_df)):
            dynamic_list = [list_df[j - i] for i in range(WINDOW_SIZE)]
            x = x + [generate_input(*dynamic_list)]
            y = y + [generate_label(i, number_of_classes=len(training_dataset))]

    x = np.array(x)
    y = np.array(y)
    model.fit(x=x, y=y, epochs=300, verbose=1)

    return model, time() - t0
'''

def predict(testing_dataset, model, num_class):
    t0 = time()

    window_len = WINDOW_LEN
    list_df = [testing_dataset[w:w + window_len] for w in range(0, testing_dataset.shape[0], window_len)]
    n_attacks = 0

    print("El numero de clase es " + str(num_class))

    for j in range(WINDOW_SIZE-1, len(list_df)):

        dynamic_list = [list_df[j - i] for i in range(WINDOW_SIZE)]
        predicted = model.predict([generate_input(*dynamic_list)])[0]
        # predicted[0] += 0.04
            
        print("La prediccion de clase es la clase numero " + str(np.argmax(predicted)))
        print(predicted)

        if num_class >= 1: # If we are looking for attacks
            if np.argmax(predicted) == num_class:
                n_attacks += 1
        
        else: # If we are looking for non-class attacks
            if np.argmax(predicted) >= 1:
                n_attacks += 1

    return n_attacks, len(list_df)-WINDOW_SIZE-1, time() - t0


def generate_input(*dfs):
    input_scale = {}

    for i, df in enumerate(dfs):
        prefix = '' if i == 0 else f'(t-{i})'

        input_scale[f'mean {prefix}'] = df['Usage'].mean()
        input_scale[f'mean² {prefix}'] = df['Usage'].mean() ** 2
        input_scale[f'mean³ {prefix}'] = df['Usage'].mean() ** 3
        input_scale[f'std {prefix}'] = df['Usage'].std()
        input_scale[f'std² {prefix}'] = df['Usage'].std() ** 2
        input_scale[f'mode {prefix}'] = df['Usage'].mode()
        input_scale[f'range {prefix}'] = df['Usage'].max() - df['Usage'].min()
        input_scale[f'cv {prefix}'] = df['Usage'].std() / (df['Usage'].mean() + 0.00001)
        input_scale[f'cv² {prefix}'] = (df['Usage'].std() / (df['Usage'].mean() + 0.00001)) ** 2
        input_scale[f'skew {prefix}'] = ss.skew(df['Usage'])
        input_scale[f'q1 {prefix}'] = df['Usage'].quantile(0.25)
        input_scale[f'q2 {prefix}'] = df['Usage'].quantile(0.5)
        input_scale[f'q3 {prefix}'] = df['Usage'].quantile(0.75)
        input_scale[f'iqr {prefix}'] = df['Usage'].quantile(0.75) - df['Usage'].quantile(0.25)
        input_scale[f'last_minus_first {prefix}'] = df.tail(1)['Usage'].values[0] - df.head(1)['Usage'].values[0]

    return preprocessing.scale(pd.DataFrame(input_scale, index=[0]).values.flatten().tolist()).tolist()


def generate_label(kind, number_of_classes):
    labels = np.zeros(number_of_classes, dtype=int)
    labels[kind] = 1
    return labels


def print_metrics(self, meterID, detector, attack, time_model_creation, time_model_prediction, n_tp, n_tn, n_fp, n_fn):
    print("\n\nMeterID:\t\t\t", meterID)
    print("Detector:\t\t\t", detector + "_" + str(WINDOW_SIZE))
    print("Attack:\t\t\t\t", attack)
    print("Exec. time of model creation:\t", time_model_creation, "seconds")
    print("Exec. time of model prediction:\t", time_model_prediction, "seconds")
    print("Accuracy:\t\t\t", (n_tp + n_tn) / (n_tp + n_tn + n_fp + n_fn))
    print("Number of true positives:\t", n_tp)
    print("Number of false negatives:\t", n_fn)
    print("Number of true negatives:\t", n_tn)
    print("Number of false positives:\t", n_fp)
    print("[", n_tp, n_fp, "]")
    print("[", n_fn, n_tn, "]\n\n")

def metrics_to_csv(self, meterID, detector, attack, time_model_creation, time_model_prediction, n_tp, n_tn, n_fp, n_fn, type_of_dataset, type_of_attack):
    
    if type_of_attack != None:
        resulting_csv_path = "./script_results/" + type_of_dataset + "_" + type_of_attack + "_detector_comparer_results_NN_window.csv"
    else:
        resulting_csv_path = "./script_results/" + type_of_dataset + "_detector_comparer_results_NN_window.csv"

    df = pd.DataFrame({'meterID': meterID,
                        'detector': detector + "_" + str(WINDOW_SIZE),
                        'attack': attack,
                        'time_model_creation': time_model_creation,
                        'time_model_prediction': time_model_prediction,
                        'n_tp': n_tp,
                        'n_tn': n_tn,
                        'n_fp': n_fp,
                        'n_fn': n_fn,
                        'accuracy': (n_tp + n_tn) / (n_tp + n_tn + n_fp + n_fn)},
                        index=[0])

    df.to_csv(resulting_csv_path, mode='a', header=not os.path.exists(resulting_csv_path), index=False)


class MyHyperModel(HyperModel):
  
  def build(self, hp):

    model = Sequential()

    model.add(Dense(
          units = hp.get('neuron_units'),
          activation = 'relu',
    ))

    model.add(Dropout(0.2))

    #Hidden layers 
    for i in np.arange(0, hp.get('n_layers')): 
      model.add(Dense(
          units = hp.get('neuron_units'),
          activation = 'relu',
      ))

      model.add(Dropout(0.2))


    #Output layer  
    model.add(Dense(units=len(training_dataset), activation='softmax'))  # one neuron for each class

    # Compiling the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

  def fit(self, hp, model, *args, **kwargs):
      return model.fit(*args, batch_size = hp.get('batch_size'), **kwargs)



if __name__ == '__main__':


    if len(sys.argv) != 3:
        print("Usage: python3 hyperparameter_tuning_NN.py <electricity/gas/solar_consumption/solar_generation> <meter_id>")
        sys.exit(1)

    meterID = sys.argv[2]  # The meterID is set in the second parameter

    if sys.argv[1] == 'electricity':
        num_of_attacks = (105-int(WINDOW_SIZE), 105, 105, 105, 105, 105, 105)
        type_of_dataset = sys.argv[1]
        tuple_of_attacks = (False, "Avg", "FDI10", "FDI30", "RSA_0.25_1.1", "RSA_0.5_3", "Swap")

    elif sys.argv[1] == 'gas':
        num_of_attacks = (119-int(WINDOW_SIZE), 119, 119, 119, 119, 119, 119)
        type_of_dataset = sys.argv[1]
        tuple_of_attacks = (False, "Avg", "FDI10", "FDI30", "RSA_0.25_1.1", "RSA_0.5_3", "Swap")

    else:
        num_of_attacks = (357-int(WINDOW_SIZE), 357, 357, 357, 357, 357, 357)
        type_of_dataset = 'solar'

    if sys.argv[1] == 'solar_consumption':
        type_of_attack = 'consumption'
        tuple_of_attacks = (False, "Avg", "FDI10", "FDI30", "RSA_0.25_1.1", "RSA_0.5_3", "Swap")

    elif sys.argv[1] == 'solar_generation':
        type_of_attack = 'generation'
        tuple_of_attacks = (False, "Percentile", "Rating", "RSA_0.5_3")

    else:
        type_of_attack = None

    name_of_detector = "NN_v2_" + str(WINDOW_SIZE)


    training_dataset = get_training_dataset(meterID, sys.argv[1], type_of_attack)
    #model, time_model_creation = build_model(training_dataset)

    window_len = WINDOW_LEN
    x = []
    y = []

    # For each kind of behaviour
    for i in range(0, len(training_dataset)):   # i:0 -> normal

        # split into windows of two days
        list_df = [training_dataset[i][w:w + window_len] for w in range(0, training_dataset[i].shape[0], window_len)]

        # Generate training set for each window of two days
        for j in range(WINDOW_SIZE-1, len(list_df)):
            dynamic_list = [list_df[j - i] for i in range(WINDOW_SIZE)]
            x = x + [generate_input(*dynamic_list)]
            y = y + [generate_label(i, number_of_classes=len(training_dataset))]

    x = np.array(x)
    y = np.array(y)


    hp = HyperParameters()
    hp.Int("n_layers", min_value = 0 , max_value = 4) 
    hp.Int("batch_size", min_value = 8, max_value = 256, step = 16)
    hp.Int("neuron_units", min_value = 32, max_value = 512, step = 32)

    hyperband_tuner = Hyperband(
                        MyHyperModel(),
                        hyperparameters = hp,
                        objective = "val_accuracy", 
                        project_name ="hyperband_NN_tuning_" + str(WINDOW_SIZE) + '_' + str(meterID),
                        max_epochs = 10,
                        hyperband_iterations = 1, 
                        directory = "main_dir",
                        overwrite = True, 
                    )
  
    early_stopping = EarlyStopping(monitor='val_accuracy', verbose = 0, patience = 10, min_delta = 1e-3, restore_best_weights = True)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=False, mode='min', verbose=0)

    hyperband_tuner.search( x, 
                            y, 
                            steps_per_epoch = None, 
                            validation_split = 0.2, 
                            verbose = 1, #
                            #callbacks = [checkpoint], 
                            use_multiprocessing = True, 
                        )

    hyperband_best_hps = hyperband_tuner.get_best_hyperparameters()[0]
    hyperband_best_model = hyperband_tuner.hypermodel.build(hyperband_best_hps)

    with open('./script_results/tuner_results/NN/tuner_nn_results' + '.txt', 'a') as f:
        print("NN (meter_id = " + str(meterID) + ", window = " + str(WINDOW_SIZE) + ", dataset = " + str(type_of_dataset) + ")", file=f)
        print(hyperband_best_hps.values, file=f)
        print("================================================", file=f)


    hyperband_best_hps.values
