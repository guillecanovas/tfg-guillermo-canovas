"""
@Author: Raúl Javierre
@Date: updated 26/05/2021

This module provides a table component to compare the detectors considering a lot of metrics
"""

import dash_table
from src.analytics.components.Component import Component


class DetectorsTableComponent(Component):
    def update_component(self):
        self.component = dash_table.DataTable(
    columns=[{"name": i, "id": i} for i in self.dataframe.columns],
    data=self.dataframe.to_dict('records'),
    style_cell={'fontFamily': 'Nunito', 'textAlign': 'center'},
    style_header={'fontWeight': 'bold'}
)
