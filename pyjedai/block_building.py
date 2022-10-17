import itertools
from typing import Tuple
import nltk
import math
import re
import time
import logging as log
from tqdm.notebook import tqdm
import numpy as np
from collections import defaultdict

from .datamodel import Block, Data
from .utils import drop_big_blocks_by_size, drop_single_entity_blocks

class AbstractBlockBuilding:
    """Abstract class for the block building method
    """

    _method_name: str
    _method_info: str

    def __init__(self):
        self.blocks: dict
        self._progress_bar: tqdm
        self.attributes_1: list
        self.attributes_2: list
        self.execution_time: float
        self.data: Data

    def build_blocks(
            self,
            data: Data,
            attributes_1: list = None,
            attributes_2: list = None,
            tqdm_disable: bool = False
    ) -> Tuple[dict, dict]:
        """Main method of Blocking in a dataset

            Args:
                data (Data): Data module that contaiins the processed dataset
                attributes_1 (list, optional): Attribute columns of the dataset 1 \
                    that will be processed. Defaults to None. \
                    If not provided, all attributes are slected.
                attributes_2 (list, optional): Attribute columns of the dataset 2. \
                    Defaults to None. If not provided, all attributes are slected.
                tqdm_disable (bool, optional): Disables all tqdm at processing. Defaults to False.

            Returns:
                Tuple[dict, dict]: Dictionary of blocks, Dict of entity index (reversed blocks).
        """

        _start_time = time.time()
        self.data, self.attributes_1, self.attributes_2 = data, attributes_1, attributes_2
        self._progress_bar = tqdm(
            total=data.num_of_entities, desc=self._method_name, disable=tqdm_disable
        )

        # TODO Text process function can be applied in this step (.apply)
        self._entities_d1 = data.dataset_1[attributes_1 if attributes_1 else data.attributes_1] \
                            .apply(" ".join, axis=1) \
                            .apply(self._tokenize_entity) \
                            .values.tolist()
                        # if attributes_1 else data.entities_d1.apply(self._tokenize_entity)

        self._all_tokens = set(itertools.chain.from_iterable(self._entities_d1))

        if not data.is_dirty_er:
            self._entities_d2 = data.dataset_2[attributes_2 if attributes_2 else data.attributes_2] \
                    .apply(" ".join, axis=1) \
                    .apply(self._tokenize_entity) \
                    .values.tolist()
            self._all_tokens.union(set(itertools.chain.from_iterable(self._entities_d2)))

        entity_id = itertools.count()
        blocks = {}
        entity_index = defaultdict(set)

        for entity in self._entities_d1:
            eid = next(entity_id)
            for token in entity:
                blocks.setdefault(token, Block())
                blocks[token].entities_D1.add(eid)
                entity_index[eid].add(token)
            self._progress_bar.update(1)

        if not data.is_dirty_er:
            for entity in self._entities_d2:
                eid = next(entity_id)
                for token in entity:
                    blocks.setdefault(token, Block())
                    blocks[token].entities_D2.add(eid)
                    entity_index[eid].add(token)
                self._progress_bar.update(1)

        self.blocks = self._clean_blocks(blocks)
        self.entity_index = dict(filter(lambda x: len(x[1]) > 0, entity_index.items()))
        self.execution_time = time.time() - _start_time
        self._progress_bar.close()

        return self.blocks, self.entity_index

    def method_configuration(self) -> dict:
        return {
            "name" : self._method_name,
            "parameters" : self._configuration(),
            "runtime": self.execution_time
        }

    def report(self) -> None:
        """Prints Block Building method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            ("\nParameters: \n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in self._configuration().items()]) if self._configuration().items() else "\nParameters: Parameter-Free method\n") +
            "Attributes from D1:\n\t" + ', '.join(c for c in (self.attributes_1 if self.attributes_1 is not None \
                else self.data.dataset_1.columns)) +
            ("\nAttributes from D2:\n\t" + ', '.join(c for c in (self.attributes_2 if self.attributes_2 is not None \
                else self.data.dataset_2.columns)) if not self.data.is_dirty_er else "") +
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )

class StandardBlocking(AbstractBlockBuilding):
    """ Creates one block for every token in \
        the attribute values of at least two entities.
    """

    _method_name = "Standard Blocking"
    _method_info = "Creates one block for every token in " + \
        "the attribute values of at least two entities."

    def __init__(self) -> any:
        super().__init__()

    def _tokenize_entity(self, entity: str) -> list:
        """Produces a list of workds of a given string

        Args:
            entity (str): String representation  of an entity

        Returns:
            list: List of words
        """
        return list(set(filter(None, re.split('[\\W_]', entity.lower()))))

    def _clean_blocks(self, blocks: dict) -> dict:
        """No cleaning"""
        return drop_single_entity_blocks(blocks, self.data.is_dirty_er)

    def _configuration(self) -> dict:
        """No configuration"""
        return {}

class QGramsBlocking(StandardBlocking):
    """ Creates one block for every q-gram that is extracted \
        from any token in the attribute values of any entity. \
            The q-gram must be shared by at least two entities.
    """

    _method_name = "Q-Grams Blocking"
    _method_info = "Creates one block for every q-gram that is extracted " + \
                    "from any token in the attribute values of any entity. " + \
                    "The q-gram must be shared by at least two entities."

    def __init__(
            self, qgrams: int = 6
    ) -> any:
        super().__init__()
        self.qgrams = qgrams

    def _tokenize_entity(self, entity) -> set:
        keys = set()
        for token in super()._tokenize_entity(entity):
            if len(token) < self.qgrams:
                keys.add(token)
            else:
                keys.update(''.join(qg) for qg in nltk.ngrams(token, n=self.qgrams))
        return keys

    def _clean_blocks(self, blocks: dict) -> dict:
        return drop_single_entity_blocks(blocks, self.data.is_dirty_er)

    def _configuration(self) -> dict:
        return {
            "Q-Gramms" : self.qgrams
        }

class SuffixArraysBlocking(StandardBlocking):
    """ It creates one block for every suffix that appears \
        in the attribute value tokens of at least two entities.
    """

    _method_name = "Suffix Arrays Blocking"
    _method_info = "Creates one block for every suffix that appears in the " + \
        "attribute value tokens of at least two entities."

    def __init__(
            self,
            suffix_length: int = 6,
            max_block_size: int = 53
    ) -> any:
        super().__init__()
        self.suffix_length, self.max_block_size = suffix_length, max_block_size

    def _tokenize_entity(self, entity) -> set:
        keys = set()
        for token in super()._tokenize_entity(entity):
            if len(token) < self.suffix_length:
                keys.add(token)
            else:
                for length in range(0, len(token) - self.suffix_length + 1):
                    keys.add(token[length:])
        return keys

    def _clean_blocks(self, blocks: dict) -> dict:
        return drop_big_blocks_by_size(blocks, self.max_block_size, self.data.is_dirty_er)

    def _configuration(self) -> dict:
        return {
            "Suffix length" : self.suffix_length,
            "Maximum Block Size" : self.max_block_size
        }

class ExtendedSuffixArraysBlocking(StandardBlocking):
    """ It creates one block for every substring \
        (not just suffix) that appears in the tokens of at least two entities.
    """

    _method_name = "Extended Suffix Arrays Blocking"
    _method_info = "Creates one block for every substring (not just suffix) " + \
        "that appears in the tokens of at least two entities."

    def __init__(
            self,
            suffix_length: int = 6,
            max_block_size: int = 39
    ) -> any:
        super().__init__()
        self.suffix_length, self.max_block_size = suffix_length, max_block_size

    def _tokenize_entity(self, entity) -> set:
        keys = set()
        for token in super()._tokenize_entity(entity):
            keys.add(token)
            if len(token) > self.suffix_length:
                for current_size in range(self.suffix_length, len(token)): 
                    for letters in list(nltk.ngrams(token, n=current_size)):
                        keys.add("".join(letters))
        return keys

    def _clean_blocks(self, blocks: dict) -> dict:
        return drop_big_blocks_by_size(blocks, self.max_block_size, self.data.is_dirty_er)

    def _configuration(self) -> dict:
        return {
            "Suffix length" : self.suffix_length,
            "Maximum Block Size" : self.max_block_size
        }

class ExtendedQGramsBlocking(StandardBlocking):
    """It creates one block for every combination of q-grams that represents at least two entities.
    The q-grams are extracted from any token in the attribute values of any entity.
    """

    _method_name = "Extended QGramsBlocking"
    _method_info = "Creates one block for every substring (not just suffix) " + \
        "that appears in the tokens of at least two entities."

    def __init__(
            self,
            qgrams: int = 6,
            threshold: float = 0.95
    ) -> any:
        super().__init__()
        self.threshold: float = threshold
        self.MAX_QGRAMS: int = 15
        self.qgrams = qgrams

    def _tokenize_entity(self, entity) -> set:
        keys = set()
        for token in super()._tokenize_entity(entity):
            if len(token) < self.qgrams:
                keys.add(token)
            else:   
                qgrams = [''.join(qgram) for qgram in nltk.ngrams(token, n=self.qgrams)]
                if len(qgrams) == 1:
                    keys.update(qgrams)
                else:
                    if len(qgrams) > self.MAX_QGRAMS:
                        qgrams = qgrams[:self.MAX_QGRAMS]

                    minimum_length = max(1, math.floor(len(qgrams) * self.threshold))
                    for i in range(minimum_length, len(qgrams) + 1):
                        keys.update(self._qgrams_combinations(qgrams, i))

        return keys

    def _qgrams_combinations(self, sublists: list, sublist_length: int) -> list:
        if sublist_length == 0 or len(sublists) < sublist_length:
            return []

        remaining_elements = sublists.copy()
        last_sublist = remaining_elements.pop(len(sublists)-1)

        combinations_exclusive_x = self._qgrams_combinations(remaining_elements, sublist_length)
        combinations_inclusive_x = self._qgrams_combinations(remaining_elements, sublist_length-1)

        resulting_combinations = combinations_exclusive_x.copy() if combinations_exclusive_x else []

        if not combinations_inclusive_x: # is empty
            resulting_combinations.append(last_sublist)
        else:
            for combination in combinations_inclusive_x:
                resulting_combinations.append(combination+last_sublist)

        return resulting_combinations

    def _clean_blocks(self, blocks: dict) -> dict:
        return drop_single_entity_blocks(blocks, self.data.is_dirty_er)

    def _configuration(self) -> dict:
        return {
            "Q-Gramms" : self.qgrams,
            "Threshold" : self.threshold
        }
