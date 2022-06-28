'''
TODO
'''

from html import entities
import strsimpy
from strsimpy.levenshtein import Levenshtein
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from strsimpy.weighted_levenshtein import WeightedLevenshtein
from strsimpy.damerau import Damerau
from strsimpy.optimal_string_alignment import OptimalStringAlignment
from strsimpy.jaro_winkler import JaroWinkler
from strsimpy.longest_common_subsequence import LongestCommonSubsequence
from strsimpy.metric_lcs import MetricLCS
from strsimpy.ngram import NGram
from strsimpy.qgram import QGram
from strsimpy.overlap_coefficient import OverlapCoefficient
from strsimpy.cosine import Cosine
from strsimpy.jaccard import Jaccard
from strsimpy.sorensen_dice import SorensenDice
from strsimpy import SIFT4


import gensim
from gensim import corpora
from pprint import pprint

import pandas as pd
import tqdm
from tqdm import tqdm
import networkx
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.entities import Block, Data
from blocks.utils import drop_single_entity_blocks, create_entity_index, print_blocks
from utils.constants import EMBEDING_TYPES


class EntityMatching:
    '''
    TODO
    '''
    _method_name: str = "Entity Matching"
    _method_info: str = ": Calculates similarity from 0. to 1. for all blocks"

    def __init__(self, metric: str, ngram: int = 2, embedings: str = None, attributes: list = None, similarity_threshold: float = None) -> None:
        self.data: Data
        self.pairs: networkx.Graph
        self.metric = metric
        self.ngram: int = 2
        self.embedings: str = embedings
        self.attributes: list = attributes
        self.similarity_threshold = similarity_threshold
        self.entities_d1: pd.DataFrame
        self.entities_d2: pd.DataFrame = None
        self.entities: pd.DataFrame
        self._progress_bar: tqdm

        if self.metric == 'levenshtein' or self.metric == 'edit_distance':
            self._metric = Levenshtein().distance
        elif self.metric == 'nlevenshtein':
            self._metric = NormalizedLevenshtein().distance
        elif self.metric == 'jaro_winkler':
            self._metric = JaroWinkler().distance
        elif self.metric == 'metric_lcs':
            self._metric = MetricLCS().distance
        elif self.metric == 'ngram':
            self._metric = NGram(self.ngram).distance
        # elif self.metric == 'cosine':
        #     cosine = Cosine(self.ngram)
        #     self._metric = cosine.similarity_profiles(cosine.get_profile(entity_1), cosine.get_profile(entity_2))
        elif self.metric == 'jaccard':
            self._metric = Jaccard(self.ngram).distance
        elif self.metric == 'sorensen_dice':
            self._metric = SorensenDice().distance
        elif self.metric == 'overlap_coefficient':
            self._metric = OverlapCoefficient().distance

    def predict(self, blocks: dict, data: Data) -> networkx.Graph:
        '''
        TODO
        '''
        if len(blocks) == 0:
            # TODO: Error
            return None
        self.data = data
        self.pairs = networkx.Graph()
        all_blocks = list(blocks.values())
        self._progress_bar = tqdm(total=len(all_blocks), desc=self._method_name+" ("+self.metric+")")

        if self.attributes:
            self.entities_d1 = data.dataset_1[[self.attributes]]
            if not data.is_dirty_er:
                self.entities_d2 = data.dataset_2[[self.attributes]]
        else:
            self.entities_d1 = data.entities_d1
            self.entities_d2 = data.entities_d2
        
        self.entities = data.entities_d1 if data.is_dirty_er else pd.concat([data.entities_d1,  data.entities_d2])

        if isinstance(all_blocks[0], Block):
            self._predict_raw_blocks(blocks)
        elif isinstance(all_blocks[0], set):
            self._predict_prunned_blocks(blocks)
        else:
            # TODO: Error
            pass

        # if self.embedings in EMBEDING_TYPES:
        # TODO: Add GENSIM

        return self.pairs

    def _predict_raw_blocks(self, blocks: dict) -> None:
        '''
        TODO comment
        '''
        if self.data.is_dirty_er:
            for _, block in blocks.items():
                entities_array = list(block.entities_D1)
                for entity_id_1 in range(0, len(entities_array), 1):
                    for entity_id_2 in range(entity_id_1+1, len(entities_array), 1):
                        similarity = self._similarity(
                            self.entities_d1[entity_id_1],
                            self.entities_d1[entity_id_2]
                        )
                        self._insert_to_graph(entity_id_1, entity_id_2, similarity)
                        self._progress_bar.update(1)
        else:
            for _, block in blocks.items():
                for entity_id_1 in block.entities_D1:
                    for entity_id_2 in block.entities_D2:
                        similarity = self._similarity(
                            self.entities_d1[entity_id_1],
                            self.entities_d2[entity_id_2 - self.data.dataset_limit]
                        )
                        self._insert_to_graph(entity_id_1, entity_id_2, similarity)
                        self._progress_bar.update(1)

    def _predict_prunned_blocks(self, blocks: dict) -> None:
        '''
        TODO comment
        '''
        for entity_id, candidates in blocks.items():
            for candidate_id in candidates:
                similarity = self._similarity(
                    self.entities[entity_id],
                    self.entities[candidate_id]
                )
                self._insert_to_graph(entity_id, candidate_id, similarity)
                self._progress_bar.update(1)

    def _insert_to_graph(self, entity_id_1, entity_id_2, similarity):
        if self.similarity_threshold is None or \
            (self.similarity_threshold and similarity > self.similarity_threshold):
            self.pairs.add_edge(entity_id_1, entity_id_2, weight=similarity)

    def _similarity(self, entity_id1: int, entity_id2: int) -> float:

        similarity: float = 0.0

        if isinstance(self.attributes, dict):
            for attribute, weight in self.attributes.items():
                similarity += weight*self._metric(
                    self.entities.iloc[entity_id1][attribute],
                    self.entities.iloc[entity_id2][attribute]
                )
        if isinstance(self.attributes, list):
            for attribute in self.attributes:
                similarity += self._metric(
                    self.entities.iloc[entity_id1][attribute],
                    self.entities.iloc[entity_id2][attribute]
                )
        else:
            similarity = self._metric(
                self.entities.iloc[entity_id1].apply(" ".join, axis=1),
                self.entities.iloc[entity_id2].apply(" ".join, axis=1)
            )

        return similarity
