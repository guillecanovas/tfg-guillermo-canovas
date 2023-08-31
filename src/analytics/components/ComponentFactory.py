"""
@Author: Raúl Javierre
@Date: updated 26/05/2021

This module provides a ComponentFactory class
"""

from src.analytics.components.AccComponent import AccComponent
from src.analytics.components.BaccComponent import BaccComponent
from src.analytics.components.DetectorsTableComponent import DetectorsTableComponent
from src.analytics.components.MccComponent import MccComponent
from src.analytics.components.NormalVsAveragesComponent import NormalVsAveragesComponent
from src.analytics.components.NormalVsFDIsComponent import NormalVsFDIsComponent
from src.analytics.components.NormalVsRSAsComponent import NormalVsRSAsComponent
from src.analytics.components.NormalVsSwapComponent import NormalVsSwapComponent
from src.analytics.components.ScenariosTableComponent import ScenariosTableComponent
from src.analytics.components.TbComponent import TbComponent
from src.analytics.components.TnrComponent import TnrComponent
from src.analytics.components.TpComponent import TpComponent
from src.analytics.components.TprComponent import TprComponent
from src.analytics.components.NormalVsRatingVsPercentileComponent import NormalVsRatingVsPercentileComponent


class ComponentFactory:
    """
    It provides a static method to create concrete components
    """

    @staticmethod
    def create_component(component):
        if component == "AccComponent":
            return AccComponent()
        elif component == "BaccComponent":
            return BaccComponent()
        elif component == "DetectorsTableComponent":
            return DetectorsTableComponent()
        elif component == "MccComponent":
            return MccComponent()
        elif component == "NormalVsAveragesComponent":
            return NormalVsAveragesComponent()
        elif component == "NormalVsFDIsComponent":
            return NormalVsFDIsComponent()
        elif component == "NormalVsRSAsComponent":
            return NormalVsRSAsComponent()
        elif component == "NormalVsSwapComponent":
            return NormalVsSwapComponent()
        elif component == "ScenariosTableComponent":
            return ScenariosTableComponent()
        elif component == "TbComponent":
            return TbComponent()
        elif component == "TnrComponent":
            return TnrComponent()
        elif component == "TpComponent":
            return TpComponent()
        elif component == "TprComponent":
            return TprComponent()
        elif component == "NormalVsRatingVsPercentileComponent":
            return NormalVsRatingVsPercentileComponent()
        else:
            raise KeyError("Component " + component + " not found in /src/analytics/components")
