"""
@Author: Raúl Javierre
@Date: updated 26/05/2021

This module provides a graphical component (time to build metric)
"""

import dash_core_components as dcc
import plotly.express as px
from src.analytics.components.Component import Component


class TbComponent(Component):
    def update_component(self):
        self.component = dcc.Graph(figure=px.bar(self.dataframe, x="detector", y="tb (s.)"))
