import logging
import os
from pydoc import describe
import sys

import pandas as pd
import nltk
import numpy as np

from tqdm import tqdm
from sortedcontainers import SortedList, SortedSet

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.entities import Block, WorkFlow
from src.utils.utils import insert_to_dict
from src.utils.constants import LIST, SET
from src.blocks.utils import drop_single_entity_blocks, create_entity_index



class AbstractBlockCleaning:
    def __init__(self) -> None:
        pass

class BlockFiltering(AbstractBlockCleaning):
    '''
    Block Filtering
    ---
    Retains every entity in a subset of its smallest blocks

    Filtering consists of 3 steps:
    - Blocks sort in ascending cardinality
    - Creation of Entity Index: inversed block dictionary
    - Retain every entity in ratio % of its smallest blocks
    - Blocks reconstruction
    '''

    _method_name = "Block Filtering"
    _method_info = ": it retains every entity in a subset of its smallest blocks."

    def __init__(self, is_dirty_er: bool, ratio: float = 0.8) -> None:
        super().__init__()
        self.ratio = ratio
        self._is_dirty_er = is_dirty_er

    def __str__(self) -> str:
        print(self._method_name + self._method_info)
        print("Ratio: ", self.ratio)
        return super().__str__()

    def process(self, blocks: dict, dataset_lim: int) -> dict:
        '''
        Main function of Block Filtering
        ---
        Input: dict of keys -> Block
        Returns: dict of keys -> Block
        '''
        pbar = tqdm(total=3, desc="Block Filtering")

        sorted_blocks = self._sort_blocks_cardinality(blocks)
        pbar.update(1)
        entity_index, _ = create_entity_index(sorted_blocks, self._is_dirty_er)
        pbar.update(1)

        filtered_blocks = {}
        for entity_id, block_keys in entity_index.items():
            # print(entity_id, " : ", block_keys, " or ", [blocks[n].get_cardinality() for n in block_keys])
            # Create new blocks from the entity index
            for key in block_keys[:int(self.ratio*len(block_keys))]:
                filtered_blocks.setdefault(key, Block(key))

                # Entities ids start to 0 ... n-1 for 1st dataset
                # and n ... m for 2nd dataset
                if entity_id < dataset_lim:
                    filtered_blocks[key].entities_D1.add(entity_id)
                else:
                    filtered_blocks[key].entities_D2.add(entity_id)
        pbar.update(1)

        return drop_single_entity_blocks(filtered_blocks, self._is_dirty_er)

    def _sort_blocks_cardinality(self, blocks: dict) -> dict:
        return dict(sorted(blocks.items(), key=lambda x: x[1].get_cardinality(self._is_dirty_er)))

class BlockClustering(AbstractBlockCleaning):
    pass
