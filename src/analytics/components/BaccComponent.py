"""
@Author: Raúl Javierre
@Date: updated 26/05/2021

This module provides a graphical component (balanced accuracy)
"""

import dash_core_components as dcc
import plotly.express as px
from src.analytics.components.Component import Component


class BaccComponent(Component):
    def update_component(self):
        self.component = dcc.Graph(figure=px.bar(self.dataframe, x="detector", y="bacc"))
