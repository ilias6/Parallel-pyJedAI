'''
Block purging methods
'''
import numpy as np
import os, sys
import tqdm
import math
from tqdm import tqdm
from math import log10

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datamodel import Data
from utils.enums import WEIGHTING_SCHEME
from utils.constants import EMPTY
from blocks.utils import create_entity_index
from utils.constants import DISCRETIZATION_FACTOR



class AbstractBlockPurging:
    def __init__(self) -> None:
        pass
    
    
    def process(
            self,
            blocks: dict,
            data: Data
    ) -> dict:
        '''
        TODO: add description
        '''
        
        if not blocks:
            print("Empty dict of blocks was given as input!") #TODO error
            return blocks
        
        new_blocks = dict()
        self._set_threshold(new_blocks)
        
        
        num_of_purged_blocks = 0
        total_comparisons = 0
        for block_key, block in blocks:
            
            if self._satisfies_threshold(block):
                total_comparisons += block.get
            else:
                num_of_purged_blocks += 1
                

class ComparisonsBasedBlockPurging(AbstractBlockPurging):
    '''
    ComparisonsBasedBlockPurging
    '''
    _method_name = "Comparison-based Block Purging"
    _method_info = ": it discards the blocks exceeding a certain number of comparisons."

    def __init__(self, smoothing_factor: float = None) -> any:
        self.smoothing_factor: float = smoothing_factor
        self.max_comparisons_per_block: float
    
    def _set_threshold(self) -> None:
        
        
    def _satisfies_threshold(self, block) -> bool:
        pass
    
class SizeBasedBlockPurging(AbstractBlockPurging):
    pass
