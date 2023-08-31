"""
@Author: Raúl Javierre
@Date: updated 26/05/2021

This module provides a graphical component to compare the usages of different scenarios
"""

import dash_core_components as dcc
import pandas as pd

from src.analytics.components.Component import Component
import plotly.graph_objects as go

N_OBS = 336


class NormalVsRSAsComponent(Component):

    dataframe = None
    dataframe2 = None
    dataframe3 = None
    dataframe4 = None

    def update_data(self):
        if self.dataset == 'Electricity (ISSDA-CER)':
            self.dataframe = pd.read_csv('./script_results/electricity_training_data/1540_0_60.csv')
            self.dataframe2 = pd.read_csv('./script_results/electricity_training_data/1540_RSA_0.5_1.5_0_60.csv')
            self.dataframe3 = pd.read_csv('./script_results/electricity_training_data/1540_RSA_0.5_3_0_60.csv')
            self.dataframe4 = pd.read_csv('./script_results/electricity_training_data/1540_RSA_0.25_1.1_0_60.csv')
        elif self.dataset == 'Gas (ISSDA-CER)':
            self.dataframe = pd.read_csv('./script_results/gas_training_data/1838_0_60.csv')
            self.dataframe2 = pd.read_csv('./script_results/gas_training_data/1838_RSA_0.5_1.5_0_60.csv')
            self.dataframe3 = pd.read_csv('./script_results/gas_training_data/1838_RSA_0.5_3_0_60.csv')
            self.dataframe4 = pd.read_csv('./script_results/gas_training_data/1838_RSA_0.25_1.1_0_60.csv')

        self.dataframe = self.dataframe.head(N_OBS)
        self.dataframe2 = self.dataframe2.head(N_OBS)
        self.dataframe3 = self.dataframe3.head(N_OBS)
        self.dataframe4 = self.dataframe4.head(N_OBS)

    def update_component(self):
        trace1 = go.Scatter(
            x=pd.Series(range(N_OBS)),
            y=self.dataframe['Usage'],
            name='Normal'
        )

        trace2 = go.Scatter(
            x=pd.Series(range(N_OBS)),
            y=self.dataframe2['Usage'],
            name='RSA_0.5_1.5'
        )

        trace3 = go.Scatter(
            x=pd.Series(range(N_OBS)),
            y=self.dataframe3['Usage'],
            name='RSA_0.5_3'
        )

        trace4 = go.Scatter(
            x=pd.Series(range(N_OBS)),
            y=self.dataframe4['Usage'],
            name='RSA_0.25_1.1'
        )

        self.component = dcc.Graph(figure=go.Figure(data=[trace1, trace4, trace3, trace2], layout={'xaxis':{'title':'time'},'yaxis':{'title':'Usage in kWh'}}))
