'''
TODO info
'''

import logging
import os
import sys

import pandas as pd
import nltk
import numpy as np
import tqdm
from tqdm import tqdm

from typing import Dict, List, Callable

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.entities import Block, Data
from src.blocks.utils import drop_single_entity_blocks

class Evaluation:

    def __init__(self) -> None:
        self.F1: float
        self.recall: float
        self.precision: float
        self.accuracy: float
        self.num_of_comparisons: int
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def report(self, predicted: list, data: Data) -> None:

        gt = data.ground_truth

        self._inverse_groups(predicted)

        for pair in gt.itterrows():
            id1 = pair['id1']
            id2 = pair['id2']

            if (id2 in self._inverted_index and id1 in self._inverted_index[id2]) or \
                (id1 in self._inverted_index and id2 in self._inverted_index[id1]):
                self.true_positives += 1
                
    def _inverse_groups(self, groups: list) -> dict:

        self._inverted_index = dict()
        for group, group_id in (groups, range(0, len(groups))):
            for id in group:
                self._inverted_index.setdefault(id, set())
                self._inverted_index[id].add(group_id)

        return self._inverted_index
