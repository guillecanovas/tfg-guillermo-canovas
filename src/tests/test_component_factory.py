"""
@Author: Raúl Javierre
@Date: 26/05/2021

It tests the src/analytics/components/ComponentFactory class
"""

import unittest

from parameterized import parameterized

from src.analytics.components.ComponentFactory import ComponentFactory
from src.detectors.DetectorFactory import DetectorFactory


class TestComponentFactory(unittest.TestCase):

    @parameterized.expand([
        ["AccComponent"],
        ["BaccComponent"],
        ["DetectorsTableComponent"],
        ["MccComponent"],
        ["NormalVsAveragesComponent"],
        ["NormalVsFDIsComponent"],
        ["NormalVsRSAsComponent"],
        ["NormalVsSwapComponent"],
        ["ScenariosTableComponent"],
        ["TbComponent"],
        ["TnrComponent"],
        ["TpComponent"],
        ["TprComponent"],
    ])
    def testIfCreateComponentWorks(self, component_name):
        component = ComponentFactory.create_component(component_name)
        self.assertEqual(component_name, component.__class__.__name__)

    def testIfCreateNonExistentComponentWorks(self):
        self.assertRaises(KeyError, ComponentFactory.create_component, "Non Existent Component")


if __name__ == '__main__':
    unittest.main()
