import math
import os
from math import sqrt

import numpy as np
import pandas as pd

CSV_FILE = './src/analytics/detector,dataset,scenario,metrics.csv'

# MCC is 0 for unique scenarios
# TPR is 0 for normal scenario
# TNR is 0 for attack scenarios


def generate_file():
    try:
        os.remove(CSV_FILE)
    except FileNotFoundError:
        pass    # The file didn't exist
    print("Generating file...")
    generate_electricity_results()
    generate_gas_results()
    generate_solar_results()


def generate_results(df, dataset):
    detectors = df['detector'].unique().tolist() + ['All']
    scenarios = df['attack'].unique().tolist() + ['All']

    for detector in detectors:
        for scenario in scenarios:
            if scenario == 'All' and detector == 'All':
                df_detector_scenario = df
            elif scenario == 'All':
                df_detector_scenario = df[(df['detector'] == detector)]
            elif detector == 'All':
                df_detector_scenario = df[(df['attack'] == scenario)]
            else:
                df_detector_scenario = df[(df['detector'] == detector) & (df['attack'] == scenario)]

            df_detector_scenario = df_detector_scenario[df_detector_scenario.n_tp >= 0]
            generate_new_line(df_detector_scenario, dataset, detector, scenario)


def generate_electricity_results():
    df = pd.read_csv('./script_results/electricity_detector_comparer_results_v2.csv')
    dataset = 'Electricity (ISSDA-CER)'
    generate_results(df, dataset)


def generate_gas_results():
    df = pd.read_csv('./script_results/gas_detector_comparer_results_new.csv')
    dataset = 'Gas (ISSDA-CER)'
    generate_results(df, dataset)

def generate_solar_results():
    df_generation = pd.read_csv('./script_results/solar_generation_detector_comparer_results_new.csv')
    df_consumption = pd.read_csv('./script_results/solar_consumption_detector_comparer_results_new.csv')
    dataset = 'Solar (Ausgrid)'
    generate_results(df_generation, dataset)
    generate_results(df_consumption, dataset)

def generate_new_line(df, dataset, detector, scenario):
    def compute_MCC(mat):
        D = (mat[0, 0] + mat[0, 1]) * (mat[0, 0] + mat[1, 0]) * (mat[1, 1] + mat[0, 1]) * (mat[1, 1] + mat[1, 0])
        return ((mat[0, 0] * mat[1, 1]) - (mat[0, 1] * mat[1, 0])) / sqrt(D)

    positives = [df['n_tp'].sum(), df['n_fp'].sum()]
    negatives = [df['n_fn'].sum(), df['n_tn'].sum()]
    matrix = np.array([positives, negatives])

    scenario = "Normal" if scenario == "False" else scenario

    print("\n\n------------- Summary metrics -------------")
    print("Detector:\t", detector)
    print("Dataset:\t", dataset)
    print("Scenario:\t", scenario)

    # Accuracy: How often it is correct
    acc = (matrix[0, 0] + matrix[1, 1]) / matrix.sum()
    acc = round(acc, 3)
    print("\nAccuracy: ", acc)

    # Recall (=Sensitivity=True Positive Rate): when it is actually "Yes", how often it predicts "Yes"
    rec = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
    rec = 0 if math.isnan(rec) else rec
    rec = round(rec, 3)
    print("Recall: ", rec)

    # True Negative Rate (=Specificity): when it is actually "No", how often it predicts "No"
    tnr = matrix[1, 1] / (matrix[0, 1] + matrix[1, 1])
    tnr = 0 if math.isnan(tnr) else tnr
    tnr = round(tnr, 3)
    print("True Negative Rate (specificity): ", tnr)

    # Balanced accuracy (useful for imbalanced data)
    bacc = (rec + tnr) / 2
    bacc = round(bacc, 3)
    print("Balanced accuracy: ", bacc)

    # Matthews correlation coefficient (useful for imbalanced data)
    mcc = compute_MCC(matrix)
    mcc = 0 if math.isnan(mcc) else mcc
    mcc = round(mcc, 3)
    print("Mcc: ", mcc)

    tb = df['time_model_creation'].mean()
    tb = round(tb, 3)
    print("Time to build the model (s.): ", tb)

    tp = df['time_model_prediction'].mean()
    tp = round(tp, 3)
    print("Time to predict (s.): ", tp)


    return pd.DataFrame({
        'detector': detector,
        'dataset': dataset,
        'scenario': scenario,
        'acc': acc,
        'mcc': mcc,
        'bacc': bacc,
        'tpr': rec,
        'tnr': tnr,
        'tb (s.)': tb,
        'tp (s.)': tp,
    }, index=[0]).to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)

if __name__ == '__main__':
    """
    args: no args
    """
    generate_file()


