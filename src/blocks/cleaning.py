import logging
import os
import sys

import pandas as pd
import nltk
import numpy as np

from sortedcontainers import SortedList, SortedSet

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.entities import Block
from src.utils.utils import insert_to_dict

class AbstractBlockCleaning:
    def __init__(self) -> None:
        pass


class BlockFiltering(AbstractBlockCleaning):
    '''
    Block Filtering
    ---
    Retains every entity in a subset of its smallest blocks

    Input:
    Returns:
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

    def process(self, blocks: dict) -> dict:

        sorted_blocks = self.sort_blocks_cardinality(blocks)
        entity_index = dict()

        for key, block in sorted_blocks.items():
            print(block.get_cardinality())

        entity_index: dict = self.create_entity_index(sorted_blocks)

        filtered_blocks = {}
        for entity_id, block_keys in entity_index.items():

            # Sort each based on ascending cardinality and keep omly the ratio
            final_block_keys = sorted(block_keys, key=lambda x: x[1])[:int(self.ratio*len(block_keys))]
            
            # Create new blocks from the entity index
            for key, _ in final_block_keys:
                if self._is_dirty_er:
                    if key not in filtered_blocks.keys():
                        filtered_blocks[key] = Block(key, self._is_dirty_er)
                    filtered_blocks[key].entities_D1.add(entity_id)
                # else:
                    

        return self.drop_single_entity_blocks(filtered_blocks)

    def create_entity_index(self, blocks_dict) -> dict:

        entity_index = {}

        for key, block in blocks_dict.items():
            
            self._is_dirty_er = block.is_dirty_er()

            for entity_id in block.entities_D1:
                if entity_id not in entity_index.keys():
                    entity_index[entity_id] = set()
                entity_index[entity_id].add((key, blocks_dict[key].get_cardinality()))

            if not self._is_dirty_er:
                for entity_id in block.entities_D2:
                    if entity_id not in entity_index.keys():
                        entity_index[entity_id] = set()
                    entity_index[entity_id].add((key, blocks_dict[key].get_cardinality()))

        return entity_index

    def drop_single_entity_blocks(self, blocks):
        all_keys = list(blocks.keys())
        print("All keys before: ", len(all_keys))
        for key in all_keys:
            if self._is_dirty_er:
                if len(blocks[key].entities_D1) == 1:
                    blocks.pop(key)
            else:
                if (len(blocks[key].entities_D1) == 0 and len(blocks[key].entities_D2) != 0) or \
                    (len(blocks[key].entities_D1) != 0 and len(blocks[key].entities_D2) == 0):
                    blocks.pop(key)

        print("All keys after: ", len(blocks.keys()))
        return blocks

    def sort_blocks_cardinality(self, blocks: dict) -> dict:
        return dict(sorted(blocks.items(), key=lambda x: x[1].get_cardinality()))

class BlockClustering(AbstractBlockCleaning):
    pass
