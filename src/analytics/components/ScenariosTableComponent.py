"""
@Author: Raúl Javierre
@Date: updated 26/05/2021

This module provides a table component to compare the scenarios considering the benefits and the bills
"""


import dash_table
import pandas as pd
from src.analytics.components.Component import Component

N_OBS = 336


class ScenariosTableComponent(Component):

    dataframes = {}

    def update_data(self):
        if self.dataset == 'Electricity (ISSDA-CER)':
            self.dataframes['Normal'] = pd.read_csv('./script_results/electricity_training_data/1540_0_60.csv')
            self.dataframes['FDI0'] = pd.read_csv('./script_results/electricity_training_data/1540_FDI0_0_60.csv')
            self.dataframes['FDI5'] = pd.read_csv('./script_results/electricity_training_data/1540_FDI5_0_60.csv')
            self.dataframes['FDI10'] = pd.read_csv('./script_results/electricity_training_data/1540_FDI10_0_60.csv')
            self.dataframes['FDI20'] = pd.read_csv('./script_results/electricity_training_data/1540_FDI20_0_60.csv')
            self.dataframes['FDI30'] = pd.read_csv('./script_results/electricity_training_data/1540_FDI30_0_60.csv')
            self.dataframes['Swap'] = pd.read_csv('./script_results/electricity_training_data/1540_Swap_0_60.csv')
            self.dataframes['RSA_0.5_1.5'] = pd.read_csv('./script_results/electricity_training_data/1540_RSA_0.5_1.5_0_60.csv')
            self.dataframes['RSA_0.5_3'] = pd.read_csv('./script_results/electricity_training_data/1540_RSA_0.5_3_0_60.csv')
            self.dataframes['RSA_0.25_1.1'] = pd.read_csv('./script_results/electricity_training_data/1540_RSA_0.25_1.1_0_60.csv')
            self.dataframes['Avg'] = pd.read_csv('./script_results/electricity_training_data/1540_Avg_0_60.csv')
            self.dataframes['Min-Avg'] = pd.read_csv('./script_results/electricity_training_data/1540_Min-Avg_0_60.csv')

        elif self.dataset == 'Gas (ISSDA-CER)':
            self.dataframes['Normal'] = pd.read_csv('./script_results/gas_training_data/1838_0_60.csv')
            self.dataframes['FDI0'] = pd.read_csv('./script_results/gas_training_data/1838_FDI0_0_60.csv')
            self.dataframes['FDI5'] = pd.read_csv('./script_results/gas_training_data/1838_FDI5_0_60.csv')
            self.dataframes['FDI10'] = pd.read_csv('./script_results/gas_training_data/1838_FDI10_0_60.csv')
            self.dataframes['FDI20'] = pd.read_csv('./script_results/gas_training_data/1838_FDI20_0_60.csv')
            self.dataframes['FDI30'] = pd.read_csv('./script_results/gas_training_data/1838_FDI30_0_60.csv')
            self.dataframes['Swap'] = pd.read_csv('./script_results/gas_training_data/1838_Swap_0_60.csv')
            self.dataframes['RSA_0.5_1.5'] = pd.read_csv('./script_results/gas_training_data/1838_RSA_0.5_1.5_0_60.csv')
            self.dataframes['RSA_0.5_3'] = pd.read_csv('./script_results/gas_training_data/1838_RSA_0.5_3_0_60.csv')
            self.dataframes['RSA_0.25_1.1'] = pd.read_csv('./script_results/gas_training_data/1838_RSA_0.25_1.1_0_60.csv')
            self.dataframes['Avg'] = pd.read_csv('./script_results/gas_training_data/1838_Avg_0_60.csv')
            self.dataframes['Min-Avg'] = pd.read_csv('./script_results/gas_training_data/1838_Min-Avg_0_60.csv')

    def update_component(self):
        bills = []
        benefits = []
        scenarios = self.dataframes.keys()

        for scenario, dataframe in self.dataframes.items():
            bill = generate_bill(dataframe)
            self.dataframes[scenario] = bill
            bills.append(bill)

            benefit = round(self.dataframes['Normal'] - bill, 2)
            benefits.append(benefit)

        df = pd.DataFrame({
            "Scenario": scenarios,
            "Bill (€)": bills,
            "Benefit (€)": benefits
        })

        self.component = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            style_cell={'fontFamily': 'Nunito', 'textAlign': 'center'},
            style_header={'fontWeight': 'bold'}
        )


def generate_bill(df):
    AVG_PRICE_KWH_OFF_PEAK_PERIOD = 0.04  # €
    AVG_PRICE_KWH_PEAK_PERIOD = 0.11  # €

    # Off peak period is from 00:00 to 09:00 and peak period is from 09:00 to 24:00
    NINE_AM = 19

    # takes the records between 09:00 and 24:00
    df_prices_peak = df[df.DT % 100 > NINE_AM]['Usage']

    # takes the records between 00:00 and 09:00
    df_prices_off_peak = df[df.DT % 100 <= NINE_AM]['Usage']

    behaviour_total_price = df_prices_peak.sum() * AVG_PRICE_KWH_PEAK_PERIOD + df_prices_off_peak.sum() * AVG_PRICE_KWH_OFF_PEAK_PERIOD

    return round(behaviour_total_price, 2)