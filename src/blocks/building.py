'''
Blocking methods
---

One block is consisted of 1 set if Dirty ER and
2 sets if Clean-Clean ER.

TODO: Change dict instertion like cleaning or use method insert_to_dict
TODO: ids to CC as 0...n-1 and n..m can be merged in one set, no need of 2 sets?
'''

import logging
from operator import methodcaller
import os
import sys

import pandas as pd
import nltk
import numpy as np
# nltk.download('punkt')
import tqdm
from tqdm import tqdm

from typing import Dict, List, Callable

info = logging.info
error = logging.error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.tokenizer import Tokenizer
from src.core.entities import Block


class AbstractBlockBuilding:
    '''
    Abstract class for the block building method
    '''

    _method_name = None
    _method_info = None

    blocks_dict: dict = dict()

    _num_of_blocks = 0
    _is_dirty_er: bool = False

    text_cleaning_method: Callable = None

    def __init__(self) -> any:
        pass

    def build_blocks(self, entities_df_1: pd.DataFrame, entities_df_2: pd.DataFrame = None) -> any:
        '''
        Main method of Standard Blocking
        ---
        Input: Dirty/Clean-1 dataframe, Clean-2 dataframe
        Returns: dict of token -> Block
        '''

        entities_D1 = entities_df_1.apply(" ".join, axis=1)
        entities_D1_size = len(entities_D1)

        if entities_df_2 is not None:
            entities_D2 = entities_df_2.apply(" ".join, axis=1)
            tqdm_desc_1 = self._method_name + " - Clean-Clean ER (1)"
            tqdm_desc_2 = self._method_name + " - Clean-Clean ER (2)"
            self._is_dirty_er = False
        else:
            tqdm_desc_1 = self._method_name + " - Dirty ER"
            self._is_dirty_er = True

        for i in tqdm(range(0, entities_D1_size, 1), desc=tqdm_desc_1):
            record = self.text_cleaning_method(entities_D1[i]) if self.text_cleaning_method is not None else entities_D1[i]
            for token in self.tokenize_entity(record):
                if token not in self.blocks_dict.keys():
                    self.blocks_dict[token] = Block(token, self._is_dirty_er)
                self.blocks_dict[token].entities_D1.add(i)

        if entities_df_2 is not None:
            for i in tqdm(range(0, len(entities_D2), 1), desc=tqdm_desc_2):
                record = self.text_cleaning_method(entities_D2[i]) if self.text_cleaning_method is not None else entities_D2[i]
                for token in self.tokenize_entity(record):
                    if token not in self.blocks_dict.keys():
                        self.blocks_dict[token] = Block(token, self._is_dirty_er)
                    self.blocks_dict[token].entities_D2.add(entities_D1_size+i)

        self.drop_single_entity_blocks()

        return self.blocks_dict

    def drop_single_entity_blocks(self):
        '''
        Removes one-size blocks for DER and empty for CCER
        '''
        all_keys = list(self.blocks_dict.keys())
        # print("All keys before: ", len(all_keys))
        for key in all_keys:
            if self._is_dirty_er:
                if len(self.blocks_dict[key].entities_D1) == 1:
                    self.blocks_dict.pop(key)
            else:
                if (len(self.blocks_dict[key].entities_D1) == 0 and len(self.blocks_dict[key].entities_D2) != 0) or \
                    (len(self.blocks_dict[key].entities_D1) != 0 and len(self.blocks_dict[key].entities_D2) == 0):
                    self.blocks_dict.pop(key)
        # print("All keys after: ", len(self.blocks_dict.keys()))
  
    def tokenize_entity(self, entity: str) -> list:
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

    def __init__(self, text_cleaning_method=None) -> any:
        super().__init__()
        self.text_cleaning_method = text_cleaning_method

    def tokenize_entity(self, entity) -> list:
        return nltk.word_tokenize(entity)


class QGramsBlocking(AbstractBlockBuilding):
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
            qgrams=None,
            text_cleaning_method=None
    ) -> any:
        super().__init__()

        self.qgrams = qgrams
        self.text_cleaning_method = text_cleaning_method

    def tokenize_entity(self, entity) -> list:
        return [' '.join(grams) for grams in nltk.ngrams(entity, n=self.qgrams)]


class SuffixArraysBlocking(AbstractBlockBuilding):
    pass

class ExtendedSuffixArraysBlocking(AbstractBlockBuilding):
    pass

class ExtendedQGramsBlocking(AbstractBlockBuilding):
    pass

class LSHSuperBitBlocking(AbstractBlockBuilding):
    pass


class LSHMinHashBlocking(LSHSuperBitBlocking):
    pass
