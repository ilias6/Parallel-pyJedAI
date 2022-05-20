import logging
import os
import sys

import pandas as pd
import nltk
import numpy as np

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

    def process(self, blocks: dict) -> dict:

        sorted_blocks = self.find_blocks_cardinality(blocks)
        entity_index = dict()

        print(sorted_blocks)
        for k,v in sorted_blocks.items():
            print(len(v))

        

    def find_blocks_cardinality(self, blocks: dict) -> dict:
        
        return dict(sorted(blocks.items(), key=lambda x: len(x[1])))



class BlockClustering(AbstractBlockCleaning):
    pass
