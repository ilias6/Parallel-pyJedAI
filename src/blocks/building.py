import logging
import os
import sys

import pandas as pd
import nltk
import numpy as np
# nltk.download('punkt')
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

    cardinality_1: List[int]
    cardinality_2: List[int] = None

    is_dirty_er: bool = False

    def __init__(self) -> any:
        pass

    def build_blocks(self, df_1: pd.DataFrame, df_2: pd.DataFrame = None) -> dict:
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

    def build_blocks(self, df_1: pd.DataFrame, df_2: pd.DataFrame = None) -> any:

        data_1 = df_1.apply(" ".join, axis=1)

        if df_2 is not None:
            data_2 = df_2.apply(" ".join, axis=1)
            tqdm_desc_1 = self._method_name + " - Clean-Clean ER (1)"
            tqdm_desc_2 = self._method_name + " - Clean-Clean ER (2)"
        else:
            tqdm_desc_1 = self._method_name + " - Dirty ER"
            self.is_dirty_er = True
        
        for i in tqdm(range(0, len(data_1), 1), desc=tqdm_desc_1):
            if self.text_cleaning_method is not None:
                record = self.text_cleaning_method(data_1[i])
            else:
                record = data_1[i]

            for token in nltk.word_tokenize(record):
                if token not in self.blocks_dict_1.keys():
                    self.blocks_dict_1[token] = set()
                    self.num_of_blocks_1 += 1

                self.blocks_dict_1[token].add(i)
                
        if df_2 is not None:
            for i in tqdm(range(0, len(data_2), 1), desc=tqdm_desc_2):
                if self.text_cleaning_method is not None:
                    record = self.text_cleaning_method(data_2[i])
                else:
                    record = data_2[i]

                for token in nltk.word_tokenize(record):
                    if token not in self.blocks_dict_2.keys():
                        self.blocks_dict_2[token] = set()
                        self.num_of_blocks_2 += 1
                    self.blocks_dict_2[token].add(i)
        else:
            return self.blocks_dict_1

        return (self.blocks_dict_1, self.blocks_dict_2)

class QGramsBlocking(AbstractBlockBuilding):
    '''
    Q-Grams Blocking
    ---
    Creates one block for every q-gram that is extracted from any token in the attribute values of any entity.
    The q-gram must be shared by at least two entities.
    '''

    _method_name = "Q-Grams Blocking"
    _method_info = _method_name + ": it creates one block for every q-gram that is extracted from any token in the attribute values of any entity.\n" + \
                "The q-gram must be shared by at least two entities."

    def __init__(
        self,
        qgrams=None,
        is_char_tokenization=None,
        text_cleaning_method=None
    ) -> any:
        super().__init__()

        self.qgrams = qgrams
        self.is_char_tokenization = is_char_tokenization
        self.text_cleaning_method = text_cleaning_method

    def build_blocks(self, df_1: pd.DataFrame, df_2: pd.DataFrame = None) -> any:

        data_1 = df_1.apply(" ".join, axis=1)

        if df_2 is not None:
            data_2 = df_2.apply(" ".join, axis=1)
            tqdm_desc_1 = self._method_name + " - Clean-Clean ER (1)"
            tqdm_desc_2 = self._method_name + " - Clean-Clean ER (2)"
        else:
            tqdm_desc_1 = self._method_name + " - Dirty ER"
            self.is_dirty_er = True
        
        for i in tqdm(range(0, len(data_1), 1), desc=tqdm_desc_1):
            if self.text_cleaning_method is not None:
                record = self.text_cleaning_method(data_1[i])
            else:
                record = data_1[i]

            if self.is_char_tokenization:
                record = [' '.join(grams) for grams in nltk.ngrams(record, n=self.qgrams)]
            else:
                word_tokenized_record = nltk.word_tokenize(record)
                word_tokenized_record_size = len(word_tokenized_record)

                if word_tokenized_record_size > self.qgrams:
                    record = [' '.join(grams) for grams in nltk.ngrams(word_tokenized_record, n=self.qgrams)]
                else:
                    record = [' '.join(grams) for grams in  nltk.ngrams(word_tokenized_record, n=word_tokenized_record_size)]

            for token in record:
                if token not in self.blocks_dict_1.keys():
                    self.blocks_dict_1[token] = set()
                self.blocks_dict_1[token].add(i)

        if df_2 is not None:
            for i in tqdm(range(0, len(data_2), 1), desc=tqdm_desc_2):
                if self.text_cleaning_method is not None:
                    record = self.text_cleaning_method(data_2[i])
                else:
                    record = data_2[i]

                if self.is_char_tokenization:
                    record = [' '.join(grams) for grams in nltk.ngrams(record, n=self.qgrams)]
                else:
                    word_tokenized_record = nltk.word_tokenize(record)
                    word_tokenized_record_size = len(word_tokenized_record)

                    if word_tokenized_record_size > self.qgrams:
                        record = [' '.join(grams) for grams in nltk.ngrams(word_tokenized_record, n=self.qgrams)]
                    else:
                        record = [' '.join(grams) for grams in  nltk.ngrams(word_tokenized_record, n=word_tokenized_record_size)]

                for token in record:
                    if token not in self.blocks_dict_2.keys():
                        self.blocks_dict_2[token] = set()
                    self.blocks_dict_2[token].add(i)
        else:
            return self.blocks_dict_1

        return (self.blocks_dict_1, self.blocks_dict_2)

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


    

    