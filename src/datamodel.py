import colorama
from colorama import Fore
import logging
from typing import Dict
import pandas as pd
import sys, os

class Data:

    def __init__(
            self, dataset_1,
            dataset_2=None,
            ground_truth=None,
            attributes=None,
            with_header=None
        ) -> None:

        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.entities_d1: pd.DataFrame
        self.entities_d2: pd.DataFrame = None
        self.ground_truth = ground_truth
        self.is_dirty_er = False if dataset_2 else True
        self.dataset_limit = self.num_of_entities_1 = len(dataset_1)
        self.num_of_entities_2: int = len(dataset_2) if dataset_2 else 0
        self.num_of_entities: int = self.num_of_entities_1 + self.num_of_entities_2
        self.attributes: list = attributes if attributes else dataset_1.columns.values.tolist()
        self.entities: pd.DataFrame

    def process(self, text_cleaning_method=None) -> None:
        
        if self.attributes: self.dataset_1 = self.dataset_1[self.attributes]
        self.entities = self.dataset_1 = self.dataset_1.apply(text_cleaning_method)
        self.entities_d1 = self.dataset_1.apply(" ".join, axis=1)
        
        if not self.is_dirty_er:
            if self.attributes: self.dataset_2 = self.dataset_2[self.attributes]
            self.dataset_2 = self.dataset_2.apply(text_cleaning_method)
            self.entities_d2 = self.dataset_2.apply(" ".join, axis=1)
            self.entities = pd.concat([self.dataset_1, self.dataset_2])
            self.ground_truth.iloc[:, 1] = self.ground_truth.iloc[:, 1] + len(self.ground_truth.iloc[:, 0])

    def print_specs(self):
        print("Type of Entity Resolution: ", "Dirty" if self.is_dirty_er else "Clean-Clean" )
        print("Number of entities in D1: ", self.num_of_entities_1)
        if not self.is_dirty_er:
            print("Number of entities in D1: ", self.num_of_entities_2)
        print("Total number of entities: ", self.num_of_entities)
        print("Attributes provided: ", self.dataset_1.columns.values.tolist())
        
class Block:
    '''
    Block entity
    ---
    Consists of 2 sets of profile entities (1 for Dirty ER and 2 for Clean-Clean ER)
    '''

    def __init__(self) -> None:
        self.entities_D1: set = set()
        self.entities_D2: set = set()

    def get_cardinality(self, is_dirty_er) -> int:
        if is_dirty_er:
            return len(self.entities_D1)*(len(self.entities_D1)-1)/2
        return len(self.entities_D1) * len(self.entities_D2)

    def get_size(self) -> int:
        return len(self.entities_D1) + len(self.entities_D2)

    def verbose(self, is_dirty_er):
        print("\nBlock ", "\033[1;32m"+self.key+"\033[0m", " contains entities with ids: ")
        if is_dirty_er:
            print("Dirty dataset: " + "[\033[1;34m" + \
             str(len(self.entities_D1)) + " entities\033[0m]")
            print(self.entities_D1)
        else:
            print("Clean dataset 1: " + "[\033[1;34m" + \
             str(len(self.entities_D1)) + " entities\033[0m]")
            print(self.entities_D1)
            print("Clean dataset 2: " + "[\033[1;34m" + str(len(self.entities_D2)) + \
            " entities\033[0m]")
            print(self.entities_D2)
