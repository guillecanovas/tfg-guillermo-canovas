"""
@Author: Raúl Javierre
@Date: Updated on 15/05/2021

ONLY ELECTRICITY dataset

1. Generates the TS for all the meterIDs
2. Does the two tests of stationarity for each TS
    - ADF (linear or difference stationarity)
    - KPSS (trend stationarity)
    * Strict stationarity: A TS is stationary if its mean, variance, autocorrelation
    are constant over time (autocorrelation=correlation of the time series
    with its previous values)
3. Generates a csv (electricity_results_customers_analysis.csv). Example:
    ID,Classification,ADF,ADF_p-value,ADF-99%,ADF-95%,ADF-90%,KPSS-S,KPSS_p-value,KPSS-99%,KPSS-97.5%,KPSS-95%,KPSS-90%
    ###################################################################################################################
    5155,1,-10.435833057108383,1.5673486471821276e-18,-3.4306100281314627,-2.8616549245917264,-2.5668311709946092,2.070409387169639,0.01,0.739,0.574,0.463,0.347
    2458,1,-17.372278631941843,5.129447416584784e-30,-3.4306100281314627,-2.8616549245917264,-2.5668311709946092,1.3017459380707836,0.01,0.739,0.574,0.463,0.347
    3979,1,-21.06315763014405,0.0,-3.4306100281314627,-2.8616549245917264,-2.5668311709946092,0.9202189578654664,0.01,0.739,0.574,0.463,0.347

Observation: later, electricity_results_customers_analysis.csv was manually updated writing (yes/no) according
to the numerical values and sorting the file by meterIDs
"""

from misc import *
import pandas as pd

results = "./script_results/electricity_results_customers_analysis.csv"


def generate_results_customers_analysis(meterID, classification, result_adf, result_kpss, output):
    result = dict()

    for key, value in result_adf[4].items():
        result["ADF-" + str(key)] = value

    for key, value in result_kpss[3].items():
        result["KPSS-" + str(key)] = value

    df = pd.DataFrame({'ID': meterID,
                       'Classification': classification,
                       'ADF': result_adf[0],
                       'ADF_p-value': result_adf[1],
                       'ADF-99%': result["ADF-1%"],
                       'ADF-95%': result["ADF-5%"],
                       'ADF-90%': result["ADF-10%"],
                       'KPSS-S': result_kpss[0],
                       'KPSS_p-value': result_kpss[1],
                       'KPSS-99%': result["KPSS-1%"],
                       'KPSS-97.5%': result["KPSS-2.5%"],
                       'KPSS-95%': result["KPSS-5%"],
                       'KPSS-90%': result["KPSS-10%"]},
                      index=[0])

    df.to_csv(output, header=meterID == 5155, index=False)


if __name__ == '__main__':
    # Gets 1330 also (used by Badrinath). Last row of the csv (type "other")
    meterIDs = get_808_random_residential_meter_ids() + get_72_random_SME_meter_ids() + \
               get_119_random_other_meter_ids() + [1330]

    count_meter_ids_proportion(meterIDs, "Electricity")

    files = get_data_set_files_of_electricity_filtered()

    output = open(results, "a")  # appending!!!

    for meterID in meterIDs:
        classification = get_classification_of_meter_id(meterID, "Electricity")
        tsa = TSanalyzer()
        tsa.load_time_series(files=files, firstWeek=0, lastWeek=75, meterID=meterID)
        result_adf, result_kpss = tsa.stationarity_tests(tsa.df.Usage.values)
        generate_results_customers_analysis(meterID, classification, result_adf, result_kpss, output)

    output.close()
