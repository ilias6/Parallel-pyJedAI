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
from datamodel import Data, Block
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
        self.data = data
        if not blocks:
            print("Empty dict of blocks was given as input!") #TODO error
            return blocks
        
        new_blocks = blocks.copy()
        
        self._set_threshold(new_blocks)
        print(new_blocks)
        
        num_of_purged_blocks = 0
        total_comparisons = 0
        for block_key, block in new_blocks:            
            print(block_key, type(block))
            if self._satisfies_threshold(block):
                total_comparisons += block.get_cardinality(self.data.is_dirty_er)
            else:
                num_of_purged_blocks += 1
                new_blocks.remove(block_key)
                
        print("Purged blocks\t:\t", num_of_purged_blocks)
        print("Retained blocks\t:\t", len(new_blocks))
        print("Retained comparisons\t:\t", total_comparisons)
        
        return new_blocks
                

class ComparisonsBasedBlockPurging(AbstractBlockPurging):
    '''
    ComparisonsBasedBlockPurging
    '''
    _method_name = "Comparison-based Block Purging"
    _method_info = ": it discards the blocks exceeding a certain number of comparisons."

    def __init__(self, smoothing_factor: float = 1.025) -> any:
        self.smoothing_factor: float = smoothing_factor
        self.max_comparisons_per_block: float
    
    def _set_threshold(self, blocks: dict) -> None:
        
        sorted_blocks = list(sorted(blocks.items(), key=lambda item: item[1].get_cardinality(self.data.is_dirty_er)))
        distinct_comparisons_level = set(b.get_cardinality(self.data.is_dirty_er) for k, b in sorted_blocks)
        block_assignments = np.empty([len(distinct_comparisons_level)])
        comparisons_level = np.empty([len(distinct_comparisons_level)])
        total_comparisons_per_level = np.empty([len(distinct_comparisons_level)])
        index = -1
        
        for block_key, block in sorted_blocks:
            block_size = block.get_size()
            if index == -1:
                index += 1
                comparisons_level[index] = block.get_cardinality(self.data.is_dirty_er)
                block_assignments[index] = 0
                total_comparisons_per_level[index] = 0
            elif block.get_cardinality(self.data.is_dirty_er) != comparisons_level[index]:
                index += 1
                comparisons_level[index] = block.get_cardinality(self.data.is_dirty_er)
                block_assignments[index] = block_assignments[index-1]
                total_comparisons_per_level[index] = total_comparisons_per_level[index-1]
            
            block_assignments[index] += block.get_size()
            total_comparisons_per_level += block.get_cardinality(self.data.is_dirty_er)
            
        
        current_bc = 0; current_cc = 0; current_size = 0
        previous_bc = 0; previous_cc = 0; previous_size = 0
        array_size = len(block_assignments)
        
        for i in range(array_size-1, 0, -1):
            previous_size = current_size
            previous_bc = current_bc
            previous_cc = current_cc

            current_bc = block_assignments[i]
            current_size = comparisons_level[i]
            current_cc = total_comparisons_per_level[i]
            
            if current_bc * previous_cc < self.smoothing_factor *current_cc * previous_cc:
                break
                
        self.max_comparisons_per_block = previous_size
                        
    def _satisfies_threshold(self, block: Block) -> bool:
        return block.get_cardinality(seld.data.is_dirty_er) <= self.max_comparisons_per_block
