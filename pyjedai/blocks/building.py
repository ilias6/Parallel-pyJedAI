'''
Blocking methods
---

One block is consisted of 1 set if Dirty ER and
2 sets if Clean-Clean ER.

TODO: Change dict instertion like cleaning or use method insert_to_dict
TODO: ids to CC as 0...n-1 and n..m can be merged in one set, no need of 2 sets?
'''
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from operator import methodcaller
import os
import sys

import pandas as pd
import nltk
import numpy as np
# nltk.download('punkt')
import tqdm
from tqdm.notebook import tqdm

import math
import re
import time

from typing import Dict, List, Callable

info = logging.info
error = logging.error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datamodel import Block, Data
from blocks.utils import drop_single_entity_blocks, drop_big_blocks_by_size


class AbstractBlockBuilding:
    '''
    Abstract class for the block building method
    '''

    _method_name: str
    _method_info: str

    def __init__(self) -> any:
        self.blocks: dict

    def build_blocks(
            self, data: Data,
            attributes_1: list=None,
            attributes_2: list=None,
    ) -> dict:
        '''
        Main method of Standard Blocking
        ---
        Input: Dirty/Clean-1 dataframe, Clean-2 dataframe
        Returns: dict of token -> Block
        '''
        start_time = time.time()
        self.blocks: dict = dict()
        self.attributes_1 = attributes_1
        self.attributes_2 = attributes_2
        self._progress_bar = tqdm(total=data.num_of_entities, desc=self._method_name)        
        for i in range(0, data.num_of_entities_1, 1):
            record = data.dataset_1.iloc[i, attributes_1] if attributes_1 else data.entities_d1.iloc[i] 
            for token in self._tokenize_entity(record):
                self.blocks.setdefault(token, Block())
                self.blocks[token].entities_D1.add(i)
            self._progress_bar.update(1)
        if not data.is_dirty_er:
            for i in range(0, data.num_of_entities_2, 1):
                record = data.dataset_2.iloc[i, attributes_2] if attributes_2 else data.entities_d2.iloc[i]
                for token in self._tokenize_entity(record):
                    self.blocks.setdefault(token, Block())
                    self.blocks[token].entities_D2.add(data.dataset_limit+i)
                self._progress_bar.update(1)
        self.blocks = drop_single_entity_blocks(self.blocks, data.is_dirty_er)
        self.execution_time = time.time() - start_time
        self._progress_bar.close()
        
        return self.blocks

    def _tokenize_entity(self, entity: str) -> list:
        pass

    def __str__(self) -> str:
        pass

class StandardBlocking(AbstractBlockBuilding):
    '''
    Standard Blocking
    ---
    Creates one block for every token in the attribute values of at least two entities.
    '''

    _method_name = "Standard Blocking"
    _method_info = _method_name + ": it creates one block for every token in the attribute \
                                    values of at least two entities."

    def __init__(self) -> any:
        super().__init__()

    def _tokenize_entity(self, entity) -> set:
        return set(filter(None, re.split('[\\W_]', entity.lower())))

    def _clean_blocks(self, blocks: dict) -> dict:
        pass

class QGramsBlocking(StandardBlocking):
    '''
    Q-Grams Blocking
    ---
    Creates one block for every q-gram that is extracted from any token in the attribute \
    values of any entity. The q-gram must be shared by at least two entities.
    '''

    _method_name = "Q-Grams Blocking"
    _method_info = _method_name + ": it creates one block for every q-gram that is extracted \
                from any token in the attribute values of any entity.\n" + \
                "The q-gram must be shared by at least two entities."

    def __init__(
            self,
            qgrams: int=6,
    ) -> any:
        super().__init__()
        self.qgrams = qgrams

    def _tokenize_entity(self, entity) -> set:
        keys = set()
        tokens = super()._tokenize_entity(entity)
        for token in tokens:
            for qgrams in nltk.ngrams(token, n=self.qgrams):
                keys.add(''.join(qgrams))
        return keys
    
    def _clean_blocks(self, blocks: dict) -> dict:
        pass


class SuffixArraysBlocking(StandardBlocking):
        
    _method_name = "Suffix Arrays Blocking"
    _method_info = _method_name + ": it creates one block for every suffix that appears in the attribute value tokens of at least two entities."

    def __init__(
            self, suffix_length: int = 6
    ) -> any:
        super().__init__()
        self.suffix_length = suffix_length

    def _tokenize_entity(self, entity) -> set:
        keys = set()
        tokens = super()._tokenize_entity(entity)
        for token in tokens:
            if len(token) < self.suffix_length:
                keys.add(token)
            else:
                for length in range(0, len(token) - self.suffix_length + 1):
                    keys.add(token[:length])
        return keys
    
    def _clean_blocks(self, blocks: dict) -> dict:
        return drop_big_blocks_by_size(blocks)

    
class ExtendedSuffixArraysBlocking(StandardBlocking):
    _method_name = "Extended Suffix Arrays Blocking"
    _method_info = _method_name + ": it creates one block for every substring (not just suffix) that appears in the tokens of at least two entities."

    def __init__(
            self, suffix_length: int = 3,
    ) -> any:
        super().__init__()
        self.suffix_length = suffix_length


    def _tokenize_entity(self, entity) -> set:
        tokens = []
        for word in entity.split():
            if len(word) > self.suffix_length:
                for token in list(nltk.ngrams(word,n=self.suffix_length)):
                    tokens.append("".join(token))
            else:
                tokens.append("".join(word))
        return tokens
    
    def _clean_blocks(self, blocks: dict) -> dict:
        return drop_big_blocks_by_size(blocks)
    
class ExtendedQGramsBlocking(StandardBlocking):
    
    _method_name = "Extended QGramsBlocking"
    _method_info = _method_name + ": it creates one block for every substring (not just suffix) that appears in the tokens of at least two entities."
    
    def __init__(
        self, qgrams: int = 6, threshold: float = 0.95
    ) -> any:
        super().__init__()
        self.threshold: float = threshold
        self.MAX_QGRAMS: int = 15

    def _tokenize_entity(self, entity) -> set:
        keys = {}
        for token in super()._tokenize_entity(entity):
            qgrams = [''.join(qgram) for qgram in nltk.ngrams(word, n=self.qgrams)]
            if len(qgrams) == 1:
                keys.add(qgrams)
            else:
                if len(qgrams) > self.MAX_QGRAMS:
                    qgrams = qgrams[:self.MAX_QGRAMS]

                minimum_length = math.floor(len(qgrams) * self.threshold)

                for i in range(minimum_length, len(qgrams)):
                    keys.add(self._qgrams_combinations(qgrams, i))
        
        return keys
    
    def _qgrams_combinations(self, sublists: list, sublist_length: int) -> set:
        
        if not sublists or len(sublists) < sublist_length:
            return []
        
        remaining_elements = sublists.copy()
        last_sublist = remaining_elements.pop(len(sublists)-1)
        combinations_exclusive_x = self._qgrams_combinations(remaining_elements, sublist_length)
        combinations_inclusive_x = self._qgrams_combinations(remaining_elements, sublist_length-1)
        
        resulting_combinations = combinations_exclusive_x.copy()
        if not resulting_combinations:
            resulting_combinations.append(last_sublist)
        else:
            for combination in combinations_inclusive_x:
                resulting_combinations.append(combination+last_sublist)
            
        return resulting_combinations
    
    def _clean_blocks(self, blocks: dict) -> dict:
        pass