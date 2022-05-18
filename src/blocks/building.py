import logging
import os
import sys
import numpy as np
import tqdm
from tqdm import tqdm

from typing import Dict, List

info = logging.info
error = logging.error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.tokenizer import Tokenizer
from src.core.entities import AttributeClusters

class AbstractBlockBuilding:
    '''
    Abstract class for the block building method
    '''

    # Dirty block / Clean-Clean 1
    blocks_dict_1: dict = dict()

    # Clean-Clean 2
    blocks_dict_2 = dict()

    num_of_blocks_1 = 0
    num_of_blocks_2 = 0

    def __init__(self) -> any:
        pass

    def build_blocks(self, data: list) -> dict:
        pass

    def __str__(self) -> str:
        pass


class StandardBlocking(AbstractBlockBuilding):

    _method_name = "Standard Blocking"
    _method_info = _method_name + ": it creates one block for every token in the attribute \
                                    values of at least two entities."

    def __init__(self) -> any:
        super().__init__()

    def build_blocks(self, data_1: np.array, data_2: np.array = None) -> any:
        
        for i in tqdm(range(0, len(data_1), 1), desc="Standard block building"):
            for token in data_1[i]:
                # TODO: maybe move split to the initial stage /
                # build a list of lists as the input
                if token not in self.blocks_dict_1.keys():
                    self.blocks_dict_1[token] = set()
                self.blocks_dict_1[token].add(i)

        if data_2 is not None:
            for i in range(0, data_2.shape[0], 1):
                for token in data_2[i]:
                    if token not in self.blocks_dict_2.keys():
                        self.blocks_dict_2[token] = set()
                    self.blocks_dict_2[token].add(i)
        else:
            return self.blocks_dict_1

        return (self.blocks_dict_1, self.blocks_dict_2)

class QGramsBlocking(AbstractBlockBuilding):
    
    def __init__(
        self,
        ngrams=None,
        is_char_tokenization=None,
        text_cleaning_method=None
    ) -> any:
        super().__init__()

        self.ngrams = ngrams
        self.is_char_tokenization = is_char_tokenization
        self.text_cleaning_method = text_cleaning_method

    def build_blocks(self, data_1: np.array, data_2: np.array = None) -> any:

            for i in range(0, data_1.shape[0], 1):
                for token in data_1[i].split():
                    # TODO: maybe move split to the initial stage /
                    # build a list of lists as the input
                    if token not in self.blocks_dict_1.keys():
                        self.blocks_dict_1[token] = set()
                    self.blocks_dict_1[token].add(i)

            if data_2 is not None:
                for i in range(0, data_2.shape[0], 1):
                    for token in data_2[i]:
                        if token not in self.blocks_dict_2.keys():
                            self.blocks_dict_2[token] = set()
                        self.blocks_dict_2[token].add(i)
            else:
                return self.blocks_dict_1

            return (self.blocks_dict_1, self.blocks_dict_2)        

class SuffixArraysBlocking(AbstractBlockBuilding):
    pass

class LSHSuperBitBlocking(AbstractBlockBuilding):
    pass


class LSHMinHashBlocking(LSHSuperBitBlocking):
    pass


    

    