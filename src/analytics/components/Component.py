"""
@Author: Raúl Javierre
@Date: updated 26/05/2021

This module provides an abstract class to build components
"""

from abc import abstractmethod, ABC
import pandas as pd
from src.analytics import CSV_FILE


class Component(ABC):

    component = None

    dataset = None
    scenario = None

    dataframe = None

    def update(self, dataset, scenario):
        self.update_dataset(dataset)
        self.update_scenario(scenario)
        self.update_data()
        self.update_component()

    def update_dataset(self, dataset):
        self.dataset = dataset
        print('[' + __name__ + ']: New dataset: ' + self.dataset)

    def update_scenario(self, scenario):
        self.scenario = scenario
        print('[' + __name__ + ']: New scenario: ' + self.scenario)

    def update_data(self):
        self.dataframe = pd.read_csv(CSV_FILE)
        self.dataframe = self.dataframe[(self.dataframe['dataset'] == self.dataset) & (self.dataframe['scenario'] == self.scenario)]

    @abstractmethod
    def update_component(self):
        pass