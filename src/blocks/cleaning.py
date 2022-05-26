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

    __method_name = "Block Filtering"
    __method_info = ": it retains every entity in a subset of its smallest blocks."

    def __init__(self, ratio: float = 0.8) -> None:
        super().__init__()
        self.ratio = ratio

    def __str__(self) -> str:
        print(self.__method_name + self.__method_info)
        print("Ratio: ", self.ratio)
        return super().__str__()

    def process(self, workflow: WorkFlow) -> dict:
        '''
        Main function of Block Filtering
        ---
        Input: dict of keys -> Block
        Returns: dict of keys -> Block
        '''
        self.entities_size_D1 = workflow.num_of_entities_1
        pbar = tqdm(total=3, desc="Block Filtering")

        sorted_blocks = self.sort_blocks_cardinality(workflow.blocks)
        pbar.update(1)
        workflow.entity_index = self.create_entity_index(sorted_blocks)
        pbar.update(1)

        filtered_blocks = {}
        for entity_id, block_keys in workflow.entity_index.items():
            # print(entity_id, " : ", block_keys, " or ", [blocks[n].get_cardinality() for n in block_keys])
            # Create new blocks from the entity index
            for key in block_keys[:int(self.ratio*len(block_keys))]:
                filtered_blocks.setdefault(key, Block(key, self._is_dirty_er))

                # Entities ids start to 0 ... n-1 for 1st dataset
                # and n ... m for 2nd dataset
                if entity_id < self.entities_size_D1:
                    filtered_blocks[key].entities_D1.add(entity_id)
                else:
                    filtered_blocks[key].entities_D2.add(entity_id)
        pbar.update(1)

        workflow.blocks = self.drop_single_entity_blocks(filtered_blocks)
        
        return workflow

    def create_entity_index(self, blocks_dict) -> dict:
        '''
        Creates a dict of entity ids -> block ids
        '''
        entity_index = {}
        for key, block in blocks_dict.items():
            self._is_dirty_er = block.is_dirty_er()

            for entity_id in block.entities_D1:
                entity_index.setdefault(entity_id, [])
                entity_index[entity_id].append(key)

            if not self._is_dirty_er:
                for entity_id in block.entities_D2:
                    entity_index.setdefault(entity_id, [])
                    entity_index[entity_id].append(key)

        return entity_index

    def drop_single_entity_blocks(self, blocks):
        '''
        Removes one-size blocks for DER and empty for CCER
        '''
        all_keys = list(blocks.keys())
        # print("All keys before: ", len(all_keys))
        if self._is_dirty_er:
            for key in all_keys:
                if len(blocks[key].entities_D1) == 1:
                    blocks.pop(key)
        else:
            for key in all_keys:
                if len(blocks[key].entities_D1) == 0 or len(blocks[key].entities_D2) == 0:
                    blocks.pop(key)
        return blocks

    def sort_blocks_cardinality(self, blocks: dict) -> dict:
        return dict(sorted(blocks.items(), key=lambda x: x[1].get_cardinality()))

class BlockClustering(AbstractBlockCleaning):
    pass
