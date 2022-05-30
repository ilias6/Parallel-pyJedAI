import colorama
from colorama import Fore
import logging
from typing import Dict
import pandas as pd
import sys, os

# print(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from src.blocks.building import AbstractBlockBuilding
# from src.blocks.cleaning import AbstractBlockCleaning

class WorkFlow:

    __is_dirty_er: bool
    __dataset_1: pd.DataFrame
    __dataset_2: pd.DataFrame

    __blocks: dict
    __num_of_blocks: int
    __entity_index: dict
    __dataset_lim: int
    __num_of_entities_1: int
    __num_of_entities_2: int
    __ground_truth: pd.DataFrame

    def __init__(
            self
        ) -> None:
        pass

class Block:
    '''
    Block entity
    ---
    Consists of 2 sets of profile entities (1 for Dirty ER and 2 for Clean-Clean ER)
    '''

    def __init__(self, key) -> None:
        self.key = key
        self.entities_D1: set = set()
        self.entities_D2: set = set()

    def get_cardinality(self, is_dirty_er) -> int:
        if is_dirty_er:
            return len(self.entities_D1)
        return len(self.entities_D1) * len(self.entities_D2)

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
