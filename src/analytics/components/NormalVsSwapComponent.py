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


class NormalVsSwapComponent(Component):

    dataframe = None
    dataframe2 = None

    def update_data(self):
        if self.dataset == 'Electricity (ISSDA-CER)':
            self.dataframe = pd.read_csv('./script_results/electricity_training_data/1540_0_60.csv')
            self.dataframe2 = pd.read_csv('./script_results/electricity_training_data/1540_Swap_0_60.csv')
        elif self.dataset == 'Gas (ISSDA-CER)':
            self.dataframe = pd.read_csv('./script_results/gas_training_data/1838_0_60.csv')
            self.dataframe2 = pd.read_csv('./script_results/gas_training_data/1838_Swap_0_60.csv')

        self.dataframe = self.dataframe.head(N_OBS)
        self.dataframe2 = self.dataframe2.head(N_OBS)

    def update_component(self):
        trace1 = go.Scatter(
            x=pd.Series(range(N_OBS)),
            y=self.dataframe['Usage'],
            name='Normal'
        )

        trace2 = go.Scatter(
            x=pd.Series(range(N_OBS)),
            y=self.dataframe2['Usage'],
            name='Swap'
        )

        self.component = dcc.Graph(figure=go.Figure(data=[trace1, trace2], layout={'xaxis':{'title':'time'},'yaxis':{'title':'Usage in kWh'}}))
