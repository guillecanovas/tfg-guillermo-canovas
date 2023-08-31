"""
@Author: Raúl Javierre
@Date: updated 09/07/2021

This module provides a graphical component to compare the usages of different scenarios
"""

import dash_core_components as dcc
import pandas as pd

from src.analytics.components.Component import Component
import plotly.graph_objects as go

N_OBS = 336


class NormalVsRatingVsPercentileComponent(Component):

    dataframe = None
    dataframe2 = None
    dataframe3 = None

    def update_data(self):
        """
        if self.dataset == 'Electricity (ISSDA-CER)':
            pass
        elif self.dataset == 'Gas (ISSDA-CER)':
            pass
        """
        if self.dataset == 'Solar (Ausgrid)':
            self.dataframe = pd.read_csv('./script_results/solar_training_data/1_0_50.csv')
            self.dataframe2 = pd.read_csv('./script_results/solar_training_data/1_Rating_0_50.csv')
            self.dataframe3 = pd.read_csv('./script_results/solar_training_data/1_Percentile_0_50.csv')

        self.dataframe = self.dataframe.head(N_OBS)
        self.dataframe2 = self.dataframe2.head(N_OBS)
        self.dataframe3 = self.dataframe3.head(N_OBS)

    def update_component(self):
        trace1 = go.Scatter(
            x=pd.Series(range(N_OBS)),
            y=self.dataframe['Usage'],
            name='Normal'
        )

        trace2 = go.Scatter(
            x=pd.Series(range(N_OBS)),
            y=self.dataframe2['Usage'],
            name='Rating'
        )

        trace3 = go.Scatter(
            x=pd.Series(range(N_OBS)),
            y=self.dataframe3['Usage'],
            name='Percentile'
        )

        self.component = dcc.Graph(figure=go.Figure(data=[trace1, trace2, trace3], layout={'xaxis':{'title':'time'},'yaxis':{'title':'Usage in kWh'}}))
