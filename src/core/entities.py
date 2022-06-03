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

    def __init__(
            self, dataset_1,
            dataset_2=None,
            ground_truth=None
        ) -> None:

        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.ground_truth = ground_truth
        if dataset_2 is None:
            self.is_dirty_er = True
        else:
            self.is_dirty_er = False

        is_dirty_er: bool
        dataset_1: pd.DataFrame
        dataset_2: pd.DataFrame = None
        ground_truth: pd.DataFrame

        self.blocks: dict = None
        self.num_of_blocks: int = None
        self.entity_index: dict = None
        self.dataset_lim: int = None
        self.num_of_entities_1: int = None
        self.num_of_entities_2: int = None
        self.num_of_entities: int = None

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

    def get_total_block_assignments(self, is_dirty_er: bool) -> int:
        if is_dirty_er:
            return len(self.entities_D1)
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